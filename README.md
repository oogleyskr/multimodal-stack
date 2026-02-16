# Multimodal Stack

Seven FastAPI microservices for local AI inference on an NVIDIA RTX 3090 (24GB VRAM). Covers speech-to-text, vision, text-to-speech, image generation, embeddings, document parsing, and financial data -- all running simultaneously with ~14GB total VRAM usage.

Built as the multimodal backend for [BillBot](https://github.com/oogleyskr/billbot), an OpenClaw-based AI assistant.

## Services

| Service | Port | Model | VRAM | Type | Endpoint |
|---------|------|-------|------|------|----------|
| **STT** | 8101 | faster-whisper-large-v3 | ~3 GB | Audio to Text | `POST /transcribe` |
| **Vision** | 8102 | Qwen2.5-VL-7B-Instruct-AWQ | ~5 GB | Image to Text | `POST /describe` |
| **TTS** | 8103 | Kokoro-82M | ~0.5 GB | Text to Audio | `POST /speak` |
| **ImageGen** | 8104 | SDXL-Turbo | ~5 GB | Text to Image | `POST /generate` |
| **Embeddings** | 8105 | nomic-embed-text-v1.5 | ~0.5 GB | Text to Vector | `POST /embed` |
| **DocUtils** | 8106 | pymupdf, python-docx, openpyxl | CPU only | Document to Text | `POST /parse` |
| **FinData** | 8107 | yfinance | CPU only | Financial API | `POST /quote`, etc. |

**Total GPU VRAM: ~14 GB** (leaves ~10 GB headroom on a 24 GB card)

Every service exposes a `GET /health` endpoint that returns `{"status": "ok"}` when the model is loaded and ready.

## API Reference

### STT -- Speech-to-Text (port 8101)

Transcribes audio files using OpenAI's Whisper large-v3 via the faster-whisper engine (CTranslate2, FP16).

```bash
# Basic transcription
curl -X POST http://localhost:8101/transcribe \
  -F "file=@recording.wav"

# With options
curl -X POST http://localhost:8101/transcribe \
  -F "file=@interview.mp3" \
  -F "language=en" \
  -F "prompt=Technical podcast about machine learning" \
  -F "word_timestamps=true"
```

**Parameters:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Audio file (wav, mp3, m4a, ogg, flac, etc.) |
| `language` | string | auto-detect | Language code (en, es, fr, de, ja, zh, ...) |
| `prompt` | string | none | Context hint to guide transcription |
| `word_timestamps` | bool | false | Include per-word start/end times and confidence |

**Response:** JSON with `text`, `language`, `language_probability`, `duration`, `processing_time`, and `segments` array.

---

### Vision -- Image Analysis (port 8102)

Analyzes images using Qwen2.5-VL-7B-Instruct (AWQ 4-bit quantized). Supports custom prompts for targeted analysis.

```bash
# Describe an image
curl -X POST http://localhost:8102/describe \
  -F "file=@photo.jpg"

# Ask a specific question
curl -X POST http://localhost:8102/describe \
  -F "file=@diagram.png" \
  -F "prompt=What programming languages are shown in this diagram?" \
  -F "max_tokens=256"
```

**Parameters:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Image file (png, jpg, webp, etc.) |
| `prompt` | string | "Describe this image in detail." | Question or instruction |
| `max_tokens` | int | 512 | Maximum response length |

**Response:** JSON with `text`, `prompt`, `processing_time`, and `tokens_generated`.

---

### TTS -- Text-to-Speech (port 8103)

Synthesizes speech using Kokoro-82M with 11 built-in voices (American and British English). Returns streaming WAV audio at 24 kHz.

```bash
# Default voice
curl -X POST http://localhost:8103/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}' \
  -o output.wav

# Custom voice and speed
curl -X POST http://localhost:8103/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test.", "voice": "bm_george", "speed": 0.8}' \
  -o output.wav

# List available voices
curl http://localhost:8103/voices
```

**Parameters (JSON body):**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Text to synthesize |
| `voice` | string | af_heart | Voice ID (see table below) |
| `speed` | float | 1.0 | Speed multiplier (0.5 - 2.0) |

**Available voices:**
| ID | Description |
|----|-------------|
| `af_heart` | American Female (Heart) -- default |
| `af_bella` | American Female (Bella) |
| `af_nicole` | American Female (Nicole) |
| `af_sarah` | American Female (Sarah) |
| `af_sky` | American Female (Sky) |
| `am_adam` | American Male (Adam) |
| `am_michael` | American Male (Michael) |
| `bf_emma` | British Female (Emma) |
| `bf_isabella` | British Female (Isabella) |
| `bm_george` | British Male (George) |
| `bm_lewis` | British Male (Lewis) |

**Response:** Streaming `audio/wav` with `X-Audio-Duration` and `X-Processing-Time` headers.

---

### ImageGen -- Image Generation (port 8104)

Generates images from text prompts using Stability AI's SDXL-Turbo (FP16). Optimized for fast generation in 1-4 inference steps.

```bash
# Quick generation (1 step)
curl -X POST http://localhost:8104/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat sitting on a windowsill at sunset"}' \
  -o image.png

# With full options
curl -X POST http://localhost:8104/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "cyberpunk cityscape, neon lights, rain",
    "negative_prompt": "blurry, low quality",
    "steps": 4,
    "width": 512,
    "height": 512,
    "seed": 42
  }' \
  -o image.png
```

**Parameters (JSON body):**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | required | Text description of the desired image |
| `negative_prompt` | string | "" | What to avoid in the image |
| `steps` | int | 4 | Inference steps (1-8; SDXL-Turbo works best with 1-4) |
| `guidance_scale` | float | 0.0 | CFG scale (0.0 for turbo mode) |
| `width` | int | 512 | Image width in pixels |
| `height` | int | 512 | Image height in pixels |
| `seed` | int | -1 | Random seed for reproducibility (-1 for random) |

**Response:** Streaming `image/png` with `X-Processing-Time` and `X-Image-Size` headers.

---

### Embeddings -- Text Embeddings (port 8105)

Generates 768-dimensional embedding vectors using nomic-embed-text-v1.5. Returns results in an OpenAI-compatible format. Supports task-aware prefixing for optimal retrieval performance.

```bash
# Single text
curl -X POST http://localhost:8105/embed \
  -H "Content-Type: application/json" \
  -d '{"input": "The quick brown fox", "task_type": "search_document"}'

# Batch (up to 128 texts)
curl -X POST http://localhost:8105/embed \
  -H "Content-Type: application/json" \
  -d '{"input": ["first text", "second text"], "task_type": "search_query"}'
```

**Parameters (JSON body):**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string or string[] | required | Text(s) to embed (max 128 per request) |
| `task_type` | string | search_document | Prefix: `search_document`, `search_query`, `clustering`, or `classification` |

**Response:** OpenAI-compatible JSON with `data[].embedding` (768-dim float arrays), `model`, `usage`, `dimensions`, and `processing_time`.

---

### DocUtils -- Document Parsing (port 8106)

Extracts text from documents using CPU-only libraries. No GPU required.

```bash
# Parse a PDF
curl -X POST http://localhost:8106/parse \
  -F "file=@report.pdf"

# Parse an Excel file
curl -X POST http://localhost:8106/parse \
  -F "file=@data.xlsx"

# List supported formats
curl http://localhost:8106/formats
```

**Parameters:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Document to parse |
| `pages` | string | "" | Page range for PDFs (e.g., "1-5") |

**Supported formats:** `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.html`, `.htm`, `.txt`, `.md`, `.csv`, `.json`, `.xml`, `.yaml`, `.yml`, `.log`

**Response:** JSON with `full_text`, format-specific metadata (page count, sheet names, slide count, etc.), `filename`, `file_size`, and `processing_time`.

---

### FinData -- Financial Data (port 8107)

Retrieves stock market data via yfinance. All responses are cached for 60 seconds.

```bash
# Real-time quote
curl -X POST http://localhost:8107/quote \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'

# Price history
curl -X POST http://localhost:8107/history \
  -H "Content-Type: application/json" \
  -d '{"ticker": "NVDA", "period": "3mo", "interval": "1d"}'

# Company info
curl -X POST http://localhost:8107/info \
  -H "Content-Type: application/json" \
  -d '{"ticker": "MSFT"}'

# Financial statements
curl -X POST http://localhost:8107/financials \
  -H "Content-Type: application/json" \
  -d '{"ticker": "GOOG", "statement": "income"}'

# News
curl -X POST http://localhost:8107/news \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TSLA"}'

# Analyst recommendations
curl -X POST http://localhost:8107/analyst \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AMZN"}'

# Multi-ticker download
curl -X POST http://localhost:8107/download \
  -H "Content-Type: application/json" \
  -d '{"tickers": "AAPL,MSFT,GOOG", "period": "5d", "interval": "1d"}'
```

**Endpoints:**
| Endpoint | Body | Description |
|----------|------|-------------|
| `POST /quote` | `{"ticker": "AAPL"}` | Real-time price, volume, market cap |
| `POST /history` | `{"ticker", "period", "interval"}` | OHLCV price history |
| `POST /info` | `{"ticker": "AAPL"}` | Full company profile and fundamentals |
| `POST /financials` | `{"ticker", "statement"}` | Income, balance sheet, or cash flow (annual + quarterly) |
| `POST /news` | `{"ticker": "AAPL"}` | Latest news articles (up to 20) |
| `POST /analyst` | `{"ticker": "AAPL"}` | Analyst recommendations and upgrades/downgrades |
| `POST /download` | `{"tickers", "period", "interval"}` | Bulk OHLCV data for up to 20 tickers |

## Getting Started

### Prerequisites

- NVIDIA GPU with 16+ GB VRAM (tested on RTX 3090 24 GB)
- NVIDIA CUDA 12.x drivers
- Python 3.12
- ffmpeg (for audio processing in STT)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/oogleyskr/multimodal-stack.git ~/multimodal
   cd ~/multimodal
   ```

2. Create per-service virtual environments and install dependencies:
   ```bash
   # Example for the STT service
   python3.12 -m venv venvs/stt
   venvs/stt/bin/pip install faster-whisper uvicorn fastapi

   # Example for the Vision service
   python3.12 -m venv venvs/vision
   venvs/vision/bin/pip install torch transformers accelerate qwen-vl-utils uvicorn fastapi

   # Repeat for each service with its respective dependencies
   ```
   Each service uses its own isolated venv under `venvs/` (total ~28 GB on disk).

3. Models are downloaded automatically on first startup to the HuggingFace cache (or the `models/` directory if configured).

4. Start the services:
   ```bash
   bash scripts/start-all.sh
   ```

5. Verify everything is running:
   ```bash
   bash scripts/status.sh
   ```

## Configuration

Services are configured via `services.conf` in the project root. This file controls which services start by default:

```bash
# All 7 services enabled
ENABLED_SERVICES=(stt vision tts imagegen embeddings docutils findata)
```

To run only a subset of services, edit `services.conf` or override at runtime:

```bash
# Start only STT and TTS
bash scripts/start-all.sh stt tts

# Or via environment variable
MULTIMODAL_SERVICES="stt tts embeddings" bash scripts/start-all.sh

# Start everything regardless of config
MULTIMODAL_SERVICES="all" bash scripts/start-all.sh
```

## Management Scripts

All scripts are in the `scripts/` directory.

### Lifecycle

| Script | Description |
|--------|-------------|
| `start-all.sh` | Start enabled services (respects `services.conf`, CLI args, or `MULTIMODAL_SERVICES` env var) |
| `stop-all.sh` | Stop services by PID file or port fallback |
| `status.sh` | Health check all services, show PID/port/status table and GPU memory usage |
| `test-all.sh` | End-to-end smoke tests for each service (generates test inputs, exercises full pipeline) |

### Per-Service Helpers

Convenience scripts for quick command-line usage:

| Script | Service | Example |
|--------|---------|---------|
| `transcribe.sh` | STT | `bash scripts/transcribe.sh recording.wav --language en --word-timestamps` |
| `describe.sh` | Vision | `bash scripts/describe.sh photo.jpg` |
| `speak.sh` | TTS | `bash scripts/speak.sh "Hello world" --voice am_adam --out hello.wav` |
| `generate.sh` | ImageGen | `bash scripts/generate.sh "a sunset over mountains" --steps 4 --out sunset.png` |
| `embed.sh` | Embeddings | `bash scripts/embed.sh "some text to embed"` |
| `parse.sh` | DocUtils | `bash scripts/parse.sh document.pdf` |

### Process Management

Services run as background uvicorn processes. Logs are written to `/tmp/multimodal-<service>.log` and PID files to `/tmp/multimodal-<service>.pid`.

```bash
# View logs for a specific service
tail -f /tmp/multimodal-vision.log

# Stop a single service
bash scripts/stop-all.sh vision

# Restart a single service
bash scripts/stop-all.sh stt && bash scripts/start-all.sh stt
```

## Directory Structure

```
multimodal/
├── services/              # FastAPI server.py for each service
│   ├── stt/server.py          # Speech-to-text (faster-whisper)
│   ├── vision/server.py       # Vision-language model (Qwen2.5-VL)
│   ├── tts/server.py          # Text-to-speech (Kokoro)
│   ├── imagegen/server.py     # Image generation (SDXL-Turbo)
│   ├── embeddings/server.py   # Text embeddings (nomic-embed)
│   ├── docutils/server.py     # Document parsing (CPU-only)
│   └── findata/server.py      # Financial data (yfinance)
├── scripts/               # Management and helper scripts
│   ├── start-all.sh           # Start services
│   ├── stop-all.sh            # Stop services
│   ├── status.sh              # Health check + GPU usage
│   ├── test-all.sh            # End-to-end smoke tests
│   ├── transcribe.sh          # CLI helper for STT
│   ├── describe.sh            # CLI helper for Vision
│   ├── speak.sh               # CLI helper for TTS
│   ├── generate.sh            # CLI helper for ImageGen
│   ├── embed.sh               # CLI helper for Embeddings
│   └── parse.sh               # CLI helper for DocUtils
├── venvs/                 # Per-service Python 3.12 virtual environments (~28 GB)
├── models/                # Shared HuggingFace model cache
├── services.conf          # Configurable list of enabled services
└── README.md
```

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 16 GB | 24 GB (RTX 3090/4090) |
| System RAM | 16 GB | 32 GB |
| Disk (venvs) | ~28 GB | ~28 GB |
| Disk (models) | ~15 GB | ~15 GB |
| CUDA | 12.x | 12.x |
| Python | 3.12 | 3.12 |

All 7 services running concurrently use approximately 14 GB of GPU VRAM, leaving comfortable headroom on a 24 GB card. The CPU-only services (DocUtils, FinData) have no GPU requirement.

## Integration with BillBot

This stack serves as the multimodal backend for [BillBot](https://github.com/oogleyskr/billbot), an OpenClaw fork running on a DGX Spark. BillBot accesses these services through local HTTP calls on the same machine.

Each service maps to an OpenClaw skill:

| Skill | Service | Description |
|-------|---------|-------------|
| `/local-stt` | STT | Transcribe audio files |
| `/local-vision` | Vision | Describe and analyze images |
| `/local-tts` | TTS | Text-to-speech synthesis |
| `/local-imagegen` | ImageGen | Generate images from prompts |
| `/local-embeddings` | Embeddings | Generate text embeddings |
| `/local-docparse` | DocUtils | Parse PDF, DOCX, XLSX, and more |

The FinData service is accessed directly by BillBot's tool-calling interface for stock market queries.

## Related Repos

| Repository | Description |
|------------|-------------|
| [oogleyskr/billbot](https://github.com/oogleyskr/billbot) | OpenClaw fork with DGX Spark integration (primary AI assistant) |
| [oogleyskr/billbot-android](https://github.com/oogleyskr/billbot-android) | Android companion app (Kotlin + Jetpack Compose) |
| [oogleyskr/billbot-mcpjungle](https://github.com/oogleyskr/billbot-mcpjungle) | MCPJungle gateway configs (11 MCP servers, 136 tools) |
| [oogleyskr/billbot-service-manager](https://github.com/oogleyskr/billbot-service-manager) | Service manager MCP server for lifecycle control |
| [oogleyskr/billbot-memory-cortex](https://github.com/oogleyskr/billbot-memory-cortex) | Long-term memory system (SQLite + FTS5) |

## License

MIT
