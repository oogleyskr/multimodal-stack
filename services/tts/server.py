"""
TTS (Text-to-Speech) microservice using Kokoro-82M.

Runs hexgrad/Kokoro-82M on the RTX 3090 (~0.5GB VRAM).
Accepts text via POST /speak and returns WAV audio.

Port: 8103
"""

import io
import os
import time
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts")

app = FastAPI(title="Local TTS Service", version="1.0.0")

# Global pipeline reference — loaded once at startup
pipeline = None

# Available Kokoro voices (American English by default)
VOICES = {
    "af_heart": "American Female (Heart) — default",
    "af_bella": "American Female (Bella)",
    "af_nicole": "American Female (Nicole)",
    "af_sarah": "American Female (Sarah)",
    "af_sky": "American Female (Sky)",
    "am_adam": "American Male (Adam)",
    "am_michael": "American Male (Michael)",
    "bf_emma": "British Female (Emma)",
    "bf_isabella": "British Female (Isabella)",
    "bm_george": "British Male (George)",
    "bm_lewis": "British Male (Lewis)",
}
DEFAULT_VOICE = "af_heart"


class SpeakRequest(BaseModel):
    """Request body for the /speak endpoint."""
    text: str = Field(..., description="Text to synthesize into speech")
    voice: str = Field(default=DEFAULT_VOICE, description="Voice ID (see /voices for options)")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier (0.5-2.0)")


@app.on_event("startup")
async def load_model():
    """Load the Kokoro TTS pipeline on startup."""
    global pipeline
    from kokoro import KPipeline

    logger.info("Loading Kokoro TTS pipeline")
    start = time.time()
    # 'a' = American English; use 'b' for British
    pipeline = KPipeline(lang_code="a")
    elapsed = time.time() - start
    logger.info(f"TTS pipeline loaded in {elapsed:.1f}s")


@app.get("/health")
async def health():
    """Health check — returns 200 if pipeline is loaded."""
    return {"status": "ok" if pipeline is not None else "loading", "service": "tts"}


@app.get("/voices")
async def list_voices():
    """List available voice IDs and their descriptions."""
    return {"voices": VOICES, "default": DEFAULT_VOICE}


@app.post("/speak")
async def speak(req: SpeakRequest):
    """
    Synthesize text to speech.

    Args:
        text: The text to convert to speech.
        voice: Voice ID (default: af_heart). See GET /voices for options.
        speed: Playback speed multiplier (default: 1.0).

    Returns:
        WAV audio file as a streaming response.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline still loading")

    import soundfile as sf

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    voice = req.voice if req.voice in VOICES else DEFAULT_VOICE

    try:
        start = time.time()

        # Generate audio — Kokoro yields (graphemes, phonemes, audio) tuples
        # Concatenate all audio chunks for the full utterance
        audio_chunks = []
        sample_rate = 24000  # Kokoro default sample rate

        for _gs, _ps, audio in pipeline(req.text, voice=voice, speed=req.speed):
            if audio is not None:
                audio_chunks.append(audio)

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        # Concatenate chunks into a single array
        import numpy as np
        full_audio = np.concatenate(audio_chunks)

        # Write to WAV in memory
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, full_audio, sample_rate, format="WAV")
        wav_buffer.seek(0)

        elapsed = time.time() - start
        duration = len(full_audio) / sample_rate
        logger.info(f"Synthesized {len(req.text)} chars → {duration:.1f}s audio in {elapsed:.1f}s (voice={voice})")

        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={
                "X-Audio-Duration": str(round(duration, 3)),
                "X-Processing-Time": str(round(elapsed, 3)),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8103)
