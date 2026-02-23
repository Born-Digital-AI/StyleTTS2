from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import random
import yaml
import numpy as np
import torch
import torchaudio
import librosa
import nltk

# Your project imports
from munch import Munch
from models import build_model, load_ASR_models, load_F0_models
from utils import recursive_munch
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

import phonemizer


# -----------------------------
# Utility helpers
# -----------------------------

def set_deterministic(seed: int = 0, deterministic: bool = True) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    # lengths: [B]
    max_len = int(lengths.max().item())
    rng = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # [1, T]
    mask = rng.expand(lengths.shape[0], -1) + 1 > lengths.unsqueeze(1)  # True where padded
    return mask


def safe_load_state_dict(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    try:
        module.load_state_dict(state_dict)
    except RuntimeError:
        # Handle "module." prefix etc.
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state_dict.items():
            new_state[k[7:]] = v if k.startswith("module.") else v
        module.load_state_dict(new_state, strict=False)

# -----------------------------
# Inference class
# -----------------------------

class TTSInference:
    def __init__(
        self,
        model: Any,
        sampler: DiffusionSampler,
        phonemizer_backend: phonemizer.backend.EspeakBackend,
        text_cleaner: TextCleaner,
        device: torch.device,
        *,
        n_mels: int = 80,
        n_fft: int = 2048,
        win_length: int = 1200,
        hop_length: int = 300,
        mel_mean: float = -4.0,
        mel_std: float = 4.0,
    ):
        self.model = model
        self.sampler = sampler
        self.phonemizer = phonemizer_backend
        self.text_cleaner = text_cleaner
        self.device = device

        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=n_mels, n_fft=n_fft, win_length=win_length, hop_length=hop_length
        ).to(device)

    def _preprocess_audio_to_mel(self, wave: np.ndarray) -> torch.Tensor:
        wave_tensor = torch.from_numpy(wave).float().to(self.device)
        mel = self.to_mel(wave_tensor)
        mel = (torch.log(1e-5 + mel).unsqueeze(0) - self.mel_mean) / self.mel_std
        return mel  # [1, n_mels, T]

    def compute_style(self, ref_paths: Dict[str, str]) -> Dict[str, Tuple[torch.Tensor, np.ndarray]]:
        """
        Optional helper if you still need reference embeddings.
        Returns: {key: (ref_embedding, trimmed_audio)}
        """
        out: Dict[str, Tuple[torch.Tensor, np.ndarray]] = {}
        for key, path in ref_paths.items():
            wave, sr = librosa.load(path, sr=24000)
            audio, _ = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                audio = librosa.resample(audio, sr, 24000)

            mel = self._preprocess_audio_to_mel(audio)  # [1, n_mels, T]
            with torch.no_grad():
                ref = self.model.style_encoder(mel.unsqueeze(1))  # matches your original call
            out[key] = (ref.squeeze(1), audio)

        return out


    @torch.no_grad()
    def run(
        self,
        text: str,
        embedding_scale: float = 1.0,
        diffusion_steps: int = 5,
        noise_dim: int = 256,
    ) -> np.ndarray:
        """
        Args:
            text: input string
            embedding_scale: guidance scale
            diffusion_steps: number of diffusion steps
            noise_dim: latent noise dimension (default=256)

        Returns:
            waveform as np.ndarray
        """

        # ---- Text preprocessing ----
        text = text.strip().replace('"', "")
        phones = self.phonemizer.phonemize([text])[0]
        nltk.download('punkt_tab')
        phones = " ".join(nltk.tokenize.word_tokenize(phones))

        tokens = self.text_cleaner(phones)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).unsqueeze(0).to(self.device)

        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
        text_mask = length_to_mask(input_lengths).to(self.device)

        # ---- Text encoders ----
        t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        # ---- Noise generation (internal) ----
        noise = torch.randn(1, 1, noise_dim, device=self.device)

        # ---- Diffusion ----
        s_pred = self.sampler(
            noise,
            embedding=bert_dur[0].unsqueeze(0),
            num_steps=int(diffusion_steps),
            embedding_scale=float(embedding_scale),
        ).squeeze(0)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        # ---- Duration prediction ----
        d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.model.predictor.lstm(d)
        duration = self.model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(dim=-1)

        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        pred_dur[-1] += 5

        total_frames = int(pred_dur.sum().item())
        pred_aln_trg = torch.zeros(
            int(input_lengths.item()),
            total_frames,
            device=self.device,
        )

        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            dur_i = int(pred_dur[i].item())
            pred_aln_trg[i, c_frame:c_frame + dur_i] = 1
            c_frame += dur_i

        # ---- Prosody + decoding ----
        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)
        F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

        out = self.model.decoder(
            (t_en @ pred_aln_trg.unsqueeze(0)),
            F0_pred,
            N_pred,
            ref.squeeze().unsqueeze(0),
        )

        return out.squeeze().cpu().numpy()


# -----------------------------
# Public init function
# -----------------------------

def init_tts(
    model_path: str,
    config_path: str,
    language: str,
    device: str = "cpu",
    seed: int = 0,
    deterministic: bool = True,
) -> TTSInference:
    """
    Initialize and return an inference object that can run TTS.
    """
    set_deterministic(seed=seed, deterministic=deterministic)

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    config = yaml.safe_load(open(config_path, "r"))

    # Load pretrained helper models
    asr_cfg = config.get("ASR_config", False)
    asr_path = config.get("ASR_path", False)
    text_aligner = load_ASR_models(asr_path, asr_cfg, load_params=False)

    f0_path = config.get("F0_path", False)
    pitch_extractor = load_F0_models(f0_path, load_params=False)

    plbert_dir = config.get("PLBERT_dir", False)
    plbert = load_plbert(plbert_dir, load_params=False)

    # Build model
    model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
    # Move submodules + eval
    _ = [model[k].to(dev).eval() for k in model]

    # Load checkpoint
    ckpt = torch.load(model_path, map_location="cpu")

    # Many checkpoints store under different keys; try common patterns.
    # Prefer "model" or "state_dict" if present, otherwise assume it's already a dict of submodules.
    params = ckpt.get("model", ckpt.get("net", ckpt))
    print(params.keys())
    # Your build_model returns a dict-like collection of modules.
    for key in model:
        if key in params:
            print(f"loading {key}")
            safe_load_state_dict(model[key], params[key])
        else:
            print(f"not loading {key}")

    _ = [model[k].eval() for k in model]

    # Diffusion sampler
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )

    phon = phonemizer.backend.EspeakBackend(
        language=language,
        preserve_punctuation=True,
        with_stress=True,
        words_mismatch="ignore",
    )

    text_cleaner = TextCleaner()

    return TTSInference(
        model=model,
        sampler=sampler,
        phonemizer_backend=phon,
        text_cleaner=text_cleaner,
        device=dev,
    )

