# Multimodal Stack for RTX 3090

Local AI services running on an RTX 3090 (24GB VRAM), exposed as FastAPI microservices.
Designed to work alongside BillBot (OpenClaw) via local skills.

## Services

| Service | Model | VRAM | Port | Endpoint | Status |
|---------|-------|------|------|----------|--------|
| STT | faster-whisper large-v3 | ~3GB | 8101 | POST /transcribe | **Active** |
| Vision | Qwen2.5-VL-7B-Instruct-AWQ | ~5GB | 8102 | POST /describe | **Active** |
| TTS | Kokoro-82M | ~0.5GB | 8103 | POST /speak | **Active** |
| Image Gen | SDXL-Turbo | ~5GB | 8104 | POST /generate | **Active** |
| Embeddings | nomic-embed-text-v1.5 | ~0.5GB | 8105 | POST /embed | Disabled |
| Doc Utils | CPU-only (pymupdf, etc.) | 0 | 8106 | POST /parse | Disabled |
| Fin Data | CPU-only | 0 | 8107 | POST /query | Disabled |

**Active total: ~14GB VRAM** across 4 services (STT, Vision, TTS, ImageGen).

> **Note:** Embeddings, DocUtils, and FinData are currently disabled to free resources
> for the upcoming Radeon VII migration. Once the Radeon VII is installed, the multimodal
> stack will migrate there, freeing the RTX 3090 entirely.

## Quick Start

```bash
# Start all services
bash scripts/start-all.sh

# Check status
bash scripts/status.sh

# Stop all services
bash scripts/stop-all.sh
```

## Directory Structure

```
multimodal/
├── services/          # FastAPI server.py for each service
│   ├── stt/
│   ├── vision/
│   ├── tts/
│   ├── imagegen/
│   ├── embeddings/
│   └── docutils/
├── scripts/           # Management scripts
│   ├── start-all.sh
│   ├── stop-all.sh
│   └── status.sh
├── venvs/             # Per-service Python 3.12 virtual environments
└── models/            # Shared HuggingFace model cache
```

## Requirements

- NVIDIA GPU with 16+ GB VRAM (tested on RTX 3090)
- CUDA 12.x drivers
- Python 3.12
- ffmpeg (for audio processing)

## OpenClaw Skills

Each service has a corresponding OpenClaw skill:

- `/local-stt` — Transcribe audio files
- `/local-vision` — Describe/analyze images
- `/local-tts` — Text-to-speech synthesis
- `/local-imagegen` — Generate images from prompts
- `/local-embeddings` — Generate text embeddings
- `/local-docparse` — Parse PDF, DOCX, XLSX, etc.
