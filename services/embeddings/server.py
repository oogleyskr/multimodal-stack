"""
Embeddings microservice using nomic-embed-text-v1.5.

Runs nomic-ai/nomic-embed-text-v1.5 on the RTX 3090 (~0.5GB VRAM).
Accepts text(s) via POST /embed and returns embedding vectors.

Port: 8105
"""

import os
import time
import logging
from typing import List, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embeddings")

app = FastAPI(title="Local Embeddings Service", version="1.0.0")

# Global model reference — loaded once at startup
embed_model = None
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768  # nomic-embed-text-v1.5 output dimension


class EmbedRequest(BaseModel):
    """Request body for the /embed endpoint."""
    input: Union[str, List[str]] = Field(..., description="Text or list of texts to embed")
    task_type: str = Field(
        default="search_document",
        description="Task prefix: search_document, search_query, clustering, or classification"
    )


@app.on_event("startup")
async def load_model():
    """Load the nomic embedding model into GPU memory."""
    global embed_model
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embeddings model: {MODEL_ID}")
    start = time.time()

    embed_model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
    embed_model = embed_model.to("cuda")

    elapsed = time.time() - start
    logger.info(f"Embeddings model loaded in {elapsed:.1f}s (dim={EMBEDDING_DIM})")


@app.get("/health")
async def health():
    """Health check — returns 200 if model is loaded."""
    return {"status": "ok" if embed_model is not None else "loading", "service": "embeddings", "model": MODEL_ID}


@app.post("/embed")
async def embed(req: EmbedRequest):
    """
    Generate embeddings for one or more texts.

    The task_type prefix is prepended to each text per nomic-embed best practices:
    - "search_document" — for documents being indexed
    - "search_query" — for search queries
    - "clustering" — for clustering tasks
    - "classification" — for classification tasks

    Args:
        input: A single string or list of strings to embed.
        task_type: Task prefix (default: search_document).

    Returns:
        JSON with embeddings array, dimensions, and token count.
    """
    if embed_model is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    # Normalize input to a list
    texts = [req.input] if isinstance(req.input, str) else req.input

    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    if len(texts) > 128:
        raise HTTPException(status_code=400, detail="Maximum 128 texts per request")

    # Prepend the task type prefix per nomic-embed convention
    prefixed = [f"{req.task_type}: {t}" for t in texts]

    try:
        start = time.time()

        embeddings = embed_model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=True)

        elapsed = time.time() - start
        logger.info(f"Embedded {len(texts)} text(s) in {elapsed:.1f}s")

        # Return in OpenAI-compatible format
        return JSONResponse({
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": emb.tolist(),
                }
                for i, emb in enumerate(embeddings)
            ],
            "model": MODEL_ID,
            "usage": {
                "total_tokens": sum(len(t.split()) for t in texts),  # Approximate token count
            },
            "processing_time": round(elapsed, 3),
            "dimensions": EMBEDDING_DIM,
        })
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8105)
