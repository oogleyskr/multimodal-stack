"""
Image Generation microservice using SDXL-Turbo.

Runs stabilityai/sdxl-turbo on the RTX 3090 (~5GB VRAM).
Generates images from text prompts in 1-4 inference steps.

Port: 8104
"""

import io
import os
import time
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("imagegen")

app = FastAPI(title="Local Image Generation Service", version="1.0.0")

# Global pipeline reference — loaded once at startup
pipe = None
MODEL_ID = "stabilityai/sdxl-turbo"


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""
    prompt: str = Field(..., description="Text description of the image to generate")
    negative_prompt: str = Field(default="", description="What to avoid in the image")
    steps: int = Field(default=4, ge=1, le=8, description="Number of inference steps (1-8, default 4)")
    guidance_scale: float = Field(default=0.0, ge=0.0, le=10.0, description="CFG scale (0.0 for turbo mode)")
    width: int = Field(default=512, description="Image width in pixels")
    height: int = Field(default=512, description="Image height in pixels")
    seed: int = Field(default=-1, description="Random seed (-1 for random)")


@app.on_event("startup")
async def load_model():
    """Load the SDXL-Turbo pipeline into GPU memory."""
    global pipe
    import torch
    from diffusers import AutoPipelineForText2Image

    logger.info(f"Loading image generation model: {MODEL_ID}")
    start = time.time()

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to("cuda")

    elapsed = time.time() - start
    logger.info(f"ImageGen model loaded in {elapsed:.1f}s")


@app.get("/health")
async def health():
    """Health check — returns 200 if pipeline is loaded."""
    return {"status": "ok" if pipe is not None else "loading", "service": "imagegen", "model": MODEL_ID}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Generate an image from a text prompt.

    Args:
        prompt: Description of the desired image.
        negative_prompt: Things to avoid (default: empty).
        steps: Inference steps (default 4, SDXL-Turbo works well with 1-4).
        guidance_scale: CFG scale (default 0.0 for turbo, increase for more prompt adherence).
        width: Image width (default 512).
        height: Image height (default 512).
        seed: Random seed for reproducibility (-1 for random).

    Returns:
        PNG image as a streaming response.
    """
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    import torch

    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    # Set up the generator for reproducible results
    generator = None
    if req.seed >= 0:
        generator = torch.Generator(device="cuda").manual_seed(req.seed)

    try:
        start = time.time()

        result = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
            generator=generator,
        )

        image = result.images[0]

        # Encode as PNG in memory
        png_buffer = io.BytesIO()
        image.save(png_buffer, format="PNG")
        png_buffer.seek(0)

        elapsed = time.time() - start
        logger.info(f"Generated {req.width}x{req.height} image in {elapsed:.1f}s ({req.steps} steps)")

        return StreamingResponse(
            png_buffer,
            media_type="image/png",
            headers={
                "X-Processing-Time": str(round(elapsed, 3)),
                "X-Image-Size": f"{req.width}x{req.height}",
            },
        )
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8104)
