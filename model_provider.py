import os
import logging
import re
import soundfile as sf
import numpy as np
import torch
from azure.core.exceptions import AzureError
from pathlib import Path
from inference import init_tts
from azure.storage.blob import ContainerClient

_LOGGER = logging.getLogger(__name__)

STYLETTS_MODELS_SAS_URL=os.getenv("STYLETTS_MODELS_SAS_URL")
SAMPLE_RATE = 24000
CHUNK_PAUSE_MS = 120


def _is_valid_checkpoint(path: str) -> bool:
    try:
        torch.load(path, map_location="cpu")
        return True
    except Exception as exc:
        _LOGGER.warning("Invalid checkpoint at %s: %s", path, exc)
        return False

def get_available_styletts_models():

    container_client = ContainerClient.from_container_url(STYLETTS_MODELS_SAS_URL)
    available_models = []
    for blob in container_client.list_blobs():
        if blob.name.endswith(".pth"):
            basename = os.path.splitext(blob.name)[0]
            available_models.append(basename)
    return available_models


def download_model_if_needed(model_name, local_dir):
    """
    Downloads a blob from Azure Blob Storage into styletts_models/
    only if it does not already exist locally.

    Args:
        filename: Full blob name inside container (e.g. "models/a.pth" or "a.pth")
        container_sas_url: Full container SAS URL including ?sv=... token

    Returns:
        Local file path
    """

    filename = f"{model_name}.pth"
    os.makedirs(local_dir, exist_ok=True)

    # Save locally using just the basename
    local_path = os.path.join(local_dir, os.path.basename(filename))
    temp_path = f"{local_path}.part"

    if os.path.exists(local_path):
        if _is_valid_checkpoint(local_path):
            print(f"File already exists: {local_path}")
            return local_path
        _LOGGER.warning("Removing corrupt checkpoint and re-downloading: %s", local_path)
        os.remove(local_path)

    if os.path.exists(temp_path):
        _LOGGER.warning("Removing stale partial checkpoint: %s", temp_path)
        os.remove(temp_path)

    # Create container client from SAS URL
    container_client = ContainerClient.from_container_url(STYLETTS_MODELS_SAS_URL)

    # Get blob client
    blob_client = container_client.get_blob_client(filename)

    print(f"Downloading {filename}...")

    try:
        with open(temp_path, "wb") as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())

        if not _is_valid_checkpoint(temp_path):
            raise RuntimeError(f"Downloaded checkpoint is invalid: {temp_path}")

        os.replace(temp_path, local_path)
    except (AzureError, OSError, RuntimeError) as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Failed to download model checkpoint {filename}: {exc}") from exc

    print(f"Downloaded to {local_path}")

    return local_path
 

class StyleTTSProvider:
    def __init__(
        self,
        model_name,
        model_dir="Models",
        config_dir="Configs"
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _LOGGER.info(
            f"Device: {self.device}",
        )
        config_path = f"{config_dir}/config_sts.yml"
        language = model_name[:2]
        model_path = download_model_if_needed(model_name, model_dir)
        self.model = init_tts(model_path=model_path, device=self.device, config_path=config_path, language=language)
        _LOGGER.info(
            "Initializing StyleTTS2Provider",
        )

    def _split_sentences(self, text: str) -> list[str]:
        pieces = re.split(r"(?:\n+|(?<=[.!?…])\s+)", text.strip())
        return [re.sub(r"\s+", " ", piece).strip() for piece in pieces if piece.strip()]

    def _pack_parts(self, parts: list[str], token_limit: int) -> list[str]:
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = part if not current else f"{current} {part}"
            if self.model.get_token_count(candidate) <= token_limit:
                current = candidate
                continue

            if current:
                chunks.append(current)
                current = ""

            if self.model.get_token_count(part) > token_limit:
                raise ValueError(
                    f"Unable to fit text fragment within token limit ({token_limit}): {part[:80]!r}"
                )

            current = part

        if current:
            chunks.append(current)

        return chunks

    def _split_oversized_sentence(self, sentence: str, token_limit: int) -> list[str]:
        for pattern in (r"(?<=[,;:])\s+", r"\s+"):
            parts = [part.strip() for part in re.split(pattern, sentence) if part.strip()]
            if len(parts) <= 1:
                continue

            try:
                return self._pack_parts(parts, token_limit)
            except ValueError:
                continue

        raise ValueError(
            f"Unable to split sentence under token limit ({token_limit}): {sentence[:120]!r}"
        )

    def _chunk_text(self, text: str) -> list[str]:
        token_limit = self.model.get_token_limit()
        sentence_chunks = self._split_sentences(text)
        if not sentence_chunks:
            return []

        chunks: list[str] = []
        current = ""

        for sentence in sentence_chunks:
            sentence_token_count = self.model.get_token_count(sentence)
            if sentence_token_count > token_limit:
                oversized_parts = self._split_oversized_sentence(sentence, token_limit)
            else:
                oversized_parts = [sentence]

            for part in oversized_parts:
                candidate = part if not current else f"{current} {part}"
                if self.model.get_token_count(candidate) <= token_limit:
                    current = candidate
                    continue

                if current:
                    chunks.append(current)
                current = part

        if current:
            chunks.append(current)

        return chunks


    def synthesize(
        self,
        text: str,
        out_wav_path: Path,
        voice: str = "af",
    ):
        """
        Synthesizes `text` into a single WAV file at `out_wav_path`.
        Splits long text into token-safe chunks and concatenates the audio.
        """
        _LOGGER.info("StyleTTS2Provider: Synthesizing text %s, voice=%s", text, voice)

        chunks = self._chunk_text(text)
        _LOGGER.info("StyleTTS2Provider: split text into %s chunk(s)", len(chunks))

        audio_segments = []
        pause = np.zeros(int(SAMPLE_RATE * (CHUNK_PAUSE_MS / 1000.0)), dtype=np.float32)

        for index, chunk in enumerate(chunks):
            _LOGGER.info(
                "StyleTTS2Provider: synthesizing chunk %s/%s (%s tokens)",
                index + 1,
                len(chunks),
                self.model.get_token_count(chunk),
            )
            audio_segments.append(self.model.run(chunk, diffusion_steps=10, embedding_scale=2))
            if index < len(chunks) - 1:
                audio_segments.append(pause)

        audio = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
        sf.write(str(out_wav_path), audio, samplerate=24000)
        _LOGGER.info("StyleTTS2 synthesis complete. Saved to %s", out_wav_path)
