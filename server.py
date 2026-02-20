#!/usr/bin/env python3
import argparse
import logging
import subprocess
import uuid
import os
import time
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
import uvicorn
import httpx

from providers.piper import load_piper_model, synthesize as piper_synthesize
from providers.kokoro import KokoroProvider, get_available_kokoro_voices
from providers.styletts2 import StyleTTSProvider, get_available_styletts_models

_LOGGER = logging.getLogger(__name__)
app = FastAPI(
    title="StyleTTS2 TTS",
    description="Server that synthetizes speech using local models trained in the StyleTTS2 framework.",
    version="1.0.0",
)
security = HTTPBasic()

USERS = {
    "piper": "digital-human"
}


def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_password = USERS.get(credentials.username)
    if not correct_password or correct_password != credentials.password:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return credentials.username

@app.get("/docs", dependencies=[Depends(authenticate)])
async def secure_docs():
    pass

class SynthesizeModel(BaseModel):
    """
    Model for incoming TTS request.
    """
    text: str
    model_name: str
    sample_rate: int = 8000

STYLETTS_MODELS: List[str] = []
STYLETTS_PROVIDER: StyleTTSProvider = None


@app.get("/models")
async def list_models():
    """
    Lists available mopdels.
    """

    return {
            "available_models": STYLETTS:MODELS,
    }


@app.post("/synthesize")
async def synthesize_audio(data: SynthesizeModel):
    """
    Synthesize text to WAV at 8 kHz, automatically selecting the model based on the  'model_name' the client provides.
    """
    text = data.text.strip()
    model_name = data.model_name.strip()
    sample_rate =  str(data.sample_rate)


    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    if not model_name:
        raise HTTPException(status_code=400, detail="No model_name provided")

    if (
        model_name not in STYLETTS_MODELS
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_name '{model_name}'. "
                   f"Check /models endpoint for available names."
        )

    unique_id = uuid.uuid4().hex
    output_path = Path(f"/tmp/synth_{unique_id}.wav")
    output_8khz_path = Path(f"/tmp/synth_{unique_id}_8k.wav")

    start_time = time.time()

    try:
        if model_name in STYLETTS_MODELS:
            global STYLETTS_PROVIDER
            if STYLETTS_PROVIDER is None:
                STYLETTS_PROVIDER = StyleTTSProvider()

            STYLETTS_PROVIDER.synthesize(text=text, out_wav_path=output_path, voice=model_name)
        subprocess.run(
            ["ffmpeg", "-i", str(output_path), "-ar", sample_rate, "-y", str(output_8khz_path)],
            check=True
        )

        duration = time.time() - start_time
        _LOGGER.info("Synthesis with model '%s' took %.2f sec", model_name, duration)

        return FileResponse(
            output_8khz_path,
            media_type="audio/wav",
            filename="synthesized_8khz.wav",
            background=BackgroundTask(lambda: cleanup_temp_files([output_path, output_8khz_path]))
        )

    except subprocess.CalledProcessError as e:
        _LOGGER.error("Error during ffmpeg conversion: %s", e)
        raise HTTPException(status_code=500, detail="Error converting to 8kHz")
    except Exception as e:  
        _LOGGER.exception("Synthesis error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def cleanup_temp_files(paths):
    """Removes temporary WAV files after sending response."""
    for p in paths:
        if p.exists():
            _LOGGER.info("Removing temp file: %s", p)
            os.remove(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host")
    parser.add_argument("--port", type=int, default=3333, help="HTTP server port")
    parser.add_argument("--cuda", action="store_true", help="Use GPU")
    parser.add_argument("--data-dir", action="append", default=["./models/styletts"],
                        help="Directory for trained  models")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    global args
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug("Parsed args: %s", args)

    if not args.download_dir:
        args.download_dir = args.data_dir[0]

    global STYLETTS_MODELS
    STYLETTS_MODELS = get_available_styletts_models()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
