#!/usr/bin/env bash
# test-all.sh — Smoke test all multimodal service endpoints.
#
# Creates small test inputs and exercises each endpoint to verify
# the full pipeline works (not just health checks).
#
# Usage:
#   bash /home/mferr/multimodal/scripts/test-all.sh

set -uo pipefail

PASS=0
FAIL=0
SKIP=0

test_result() {
    local name="$1"
    local ok="$2"
    local detail="$3"
    if $ok; then
        echo "  PASS  $name — $detail"
        ((PASS++))
    else
        echo "  FAIL  $name — $detail"
        ((FAIL++))
    fi
}

skip_result() {
    local name="$1"
    local reason="$2"
    echo "  SKIP  $name — $reason"
    ((SKIP++))
}

echo "=== Multimodal Service Tests ==="
echo ""

# --- DocParse (port 8106) ---
echo "[docutils :8106]"
if curl -s -m 2 http://localhost:8106/health | grep -q '"ok"'; then
    # Test: parse a text file
    echo "Test content for docparse" > /tmp/mm-test-input.txt
    resp=$(curl -sS -m 10 -X POST http://localhost:8106/parse -F "file=@/tmp/mm-test-input.txt" 2>&1)
    if echo "$resp" | grep -q '"full_text"'; then
        test_result "parse .txt" true "extracted text successfully"
    else
        test_result "parse .txt" false "$resp"
    fi
    rm -f /tmp/mm-test-input.txt
else
    skip_result "docutils" "service not running"
fi

echo ""

# --- Embeddings (port 8105) ---
echo "[embeddings :8105]"
if curl -s -m 2 http://localhost:8105/health | grep -q '"ok"'; then
    resp=$(curl -sS -m 30 -X POST http://localhost:8105/embed \
        -H "Content-Type: application/json" \
        -d '{"input": "hello world test", "task_type": "search_query"}' 2>&1)
    if echo "$resp" | grep -q '"embedding"'; then
        dim=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['dimensions'])" 2>/dev/null || echo "?")
        test_result "embed text" true "got ${dim}-dim vector"
    else
        test_result "embed text" false "$resp"
    fi
else
    skip_result "embeddings" "service not running"
fi

echo ""

# --- STT (port 8101) ---
echo "[stt :8101]"
if curl -s -m 2 http://localhost:8101/health | grep -q '"ok"'; then
    # Generate a short silent WAV for testing (0.5s, 16kHz, mono)
    python3 -c "
import struct, wave
f = wave.open('/tmp/mm-test-audio.wav', 'w')
f.setnchannels(1)
f.setsampwidth(2)
f.setframerate(16000)
f.writeframes(struct.pack('<' + 'h' * 8000, *([0] * 8000)))
f.close()
" 2>/dev/null
    if [[ -f /tmp/mm-test-audio.wav ]]; then
        resp=$(curl -sS -m 30 -X POST http://localhost:8101/transcribe \
            -F "file=@/tmp/mm-test-audio.wav" 2>&1)
        if echo "$resp" | grep -q '"text"'; then
            test_result "transcribe wav" true "transcription returned (silent audio = empty text, expected)"
        else
            test_result "transcribe wav" false "$resp"
        fi
        rm -f /tmp/mm-test-audio.wav
    else
        test_result "transcribe wav" false "could not generate test audio"
    fi
else
    skip_result "stt" "service not running"
fi

echo ""

# --- TTS (port 8103) ---
echo "[tts :8103]"
if curl -s -m 2 http://localhost:8103/health | grep -q '"ok"'; then
    curl -sS -m 30 -X POST http://localhost:8103/speak \
        -H "Content-Type: application/json" \
        -d '{"text": "Testing one two three.", "voice": "af_heart", "speed": 1.0}' \
        -o /tmp/mm-test-tts.wav 2>&1
    if [[ -f /tmp/mm-test-tts.wav ]] && head -c 4 /tmp/mm-test-tts.wav | grep -q "RIFF"; then
        size=$(stat -c%s /tmp/mm-test-tts.wav 2>/dev/null || echo "?")
        test_result "speak text" true "got WAV file (${size} bytes)"
    else
        test_result "speak text" false "invalid or missing output"
    fi
    rm -f /tmp/mm-test-tts.wav
else
    skip_result "tts" "service not running"
fi

echo ""

# --- Vision (port 8102) ---
echo "[vision :8102]"
if curl -s -m 2 http://localhost:8102/health | grep -q '"ok"'; then
    # Generate a tiny 32x32 red PNG for testing
    python3 -c "
import struct, zlib
def make_png(w, h, r, g, b):
    raw = b''
    for y in range(h):
        raw += b'\x00' + bytes([r, g, b]) * w
    compressed = zlib.compress(raw)
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', compressed) + chunk(b'IEND', b'')
with open('/tmp/mm-test-image.png', 'wb') as f:
    f.write(make_png(32, 32, 255, 0, 0))
" 2>/dev/null
    if [[ -f /tmp/mm-test-image.png ]]; then
        resp=$(curl -sS -m 60 -X POST http://localhost:8102/describe \
            -F "file=@/tmp/mm-test-image.png" \
            -F "prompt=What color is this image?" \
            -F "max_tokens=64" 2>&1)
        if echo "$resp" | grep -q '"text"'; then
            answer=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['text'][:80])" 2>/dev/null || echo "?")
            test_result "describe image" true "response: $answer"
        else
            test_result "describe image" false "$resp"
        fi
        rm -f /tmp/mm-test-image.png
    else
        test_result "describe image" false "could not generate test image"
    fi
else
    skip_result "vision" "service not running"
fi

echo ""

# --- ImageGen (port 8104) ---
echo "[imagegen :8104]"
if curl -s -m 2 http://localhost:8104/health | grep -q '"ok"'; then
    curl -sS -m 60 -X POST http://localhost:8104/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "a solid blue square", "steps": 1, "width": 256, "height": 256}' \
        -o /tmp/mm-test-gen.png 2>&1
    if [[ -f /tmp/mm-test-gen.png ]] && file /tmp/mm-test-gen.png | grep -q "PNG"; then
        size=$(stat -c%s /tmp/mm-test-gen.png 2>/dev/null || echo "?")
        test_result "generate image" true "got PNG (${size} bytes, 256x256)"
    else
        test_result "generate image" false "invalid or missing output"
    fi
    rm -f /tmp/mm-test-gen.png
else
    skip_result "imagegen" "service not running"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed, $SKIP skipped ==="

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
