"""
STT (Speech-to-Text) microservice using faster-whisper.

Runs Systran/faster-whisper-large-v3 on the RTX 3090 (~3GB VRAM).
Accepts audio files via POST /transcribe and returns JSON transcripts.

Port: 8101
"""

import io
import os
import time
import tempfile
import logging

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt")

app = FastAPI(title="Local STT Service", version="1.0.0")

# Global model reference — loaded once at startup
model = None
MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"  # FP16 for RTX 3090 efficiency


@app.on_event("startup")
async def load_model():
    """Load the faster-whisper model into GPU memory on startup."""
    global model
    from faster_whisper import WhisperModel

    logger.info(f"Loading faster-whisper model: {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})")
    start = time.time()
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s")


@app.get("/health")
async def health():
    """Health check endpoint — returns 200 if model is loaded."""
    return {"status": "ok" if model is not None else "loading", "service": "stt", "model": MODEL_SIZE}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(default=None),
    prompt: str = Form(default=None),
    word_timestamps: bool = Form(default=False),
):
    """
    Transcribe an audio file to text.

    Args:
        file: Audio file (wav, mp3, m4a, ogg, flac, etc.)
        language: Optional language code (e.g. "en", "es", "fr"). Auto-detected if omitted.
        prompt: Optional context hint to guide transcription.
        word_timestamps: If True, include per-word timestamps in output.

    Returns:
        JSON with transcript text, detected language, and segment details.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    # Write uploaded audio to a temp file (faster-whisper needs a file path)
    suffix = os.path.splitext(file.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        start = time.time()
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            initial_prompt=prompt,
            word_timestamps=word_timestamps,
            beam_size=5,
        )

        # Collect all segments into a list
        result_segments = []
        full_text_parts = []
        for seg in segments:
            seg_data = {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            }
            if word_timestamps and seg.words:
                seg_data["words"] = [
                    {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3), "probability": round(w.probability, 3)}
                    for w in seg.words
                ]
            result_segments.append(seg_data)
            full_text_parts.append(seg.text.strip())

        elapsed = time.time() - start
        full_text = " ".join(full_text_parts)

        logger.info(f"Transcribed {file.filename} ({info.duration:.1f}s audio) in {elapsed:.1f}s — {info.language}")

        return JSONResponse({
            "text": full_text,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 3),
            "processing_time": round(elapsed, 3),
            "segments": result_segments,
        })
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101)
