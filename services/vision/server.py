"""
Vision microservice using Qwen2.5-VL-7B-Instruct-AWQ.

Runs a quantized vision-language model on the RTX 3090 (~5GB VRAM).
Accepts images via POST /describe and returns text descriptions.

Port: 8102
"""

import os
import time
import base64
import logging
import traceback

# Suppress noisy tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision")

app = FastAPI(title="Local Vision Service", version="1.0.0")

# Global model references — loaded once at startup
model = None
processor = None
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"


@app.on_event("startup")
async def load_model():
    """Load the Qwen2.5-VL model and processor into GPU memory."""
    global model, processor
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    logger.info(f"Loading vision model: {MODEL_ID}")
    start = time.time()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    elapsed = time.time() - start
    logger.info(f"Vision model loaded in {elapsed:.1f}s")


@app.get("/health")
async def health():
    """Health check — returns 200 if model is loaded."""
    return {"status": "ok" if model is not None else "loading", "service": "vision", "model": MODEL_ID}


@app.post("/describe")
async def describe(
    file: UploadFile = File(...),
    prompt: str = Form(default="Describe this image in detail."),
    max_tokens: int = Form(default=512),
):
    """
    Describe or analyze an image using the vision-language model.

    Args:
        file: Image file (png, jpg, webp, etc.)
        prompt: Question or instruction about the image.
        max_tokens: Maximum tokens in the response (default 512).

    Returns:
        JSON with the model's text response and metadata.
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    import torch
    from qwen_vl_utils import process_vision_info

    # Read image bytes and encode as base64 data URI
    content = await file.read()
    mime = file.content_type or "image/png"
    b64 = base64.b64encode(content).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"

    # Build chat messages in Qwen2.5-VL format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data_uri},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        start = time.time()

        # Apply chat template and process vision inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_tokens)

        # Decode only the generated tokens (skip the input tokens)
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        elapsed = time.time() - start
        logger.info(f"Described {file.filename} in {elapsed:.1f}s ({len(response_text)} chars)")

        return JSONResponse({
            "text": response_text.strip(),
            "prompt": prompt,
            "processing_time": round(elapsed, 3),
            "tokens_generated": len(generated_ids[0]),
        })
    except Exception as e:
        logger.error(f"Vision inference failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8102)
