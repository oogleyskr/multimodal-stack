#!/usr/bin/env bash
# start-all.sh — Start all multimodal services on the RTX 3090.
#
# Each service runs as a background uvicorn process with logs in /tmp/multimodal-*.log.
# PID files are written to /tmp/multimodal-*.pid for clean shutdown.
#
# Usage:
#   bash /home/mferr/multimodal/scripts/start-all.sh          # Start all
#   bash /home/mferr/multimodal/scripts/start-all.sh stt tts   # Start specific services

set -euo pipefail

BASE="/home/mferr/multimodal"
VENVS="$BASE/venvs"
SERVICES="$BASE/services"
LOG_DIR="/tmp"
PID_DIR="/tmp"

# Service definitions: name port
declare -A SERVICE_PORTS=(
    [stt]=8101
    [vision]=8102
    [tts]=8103
    [imagegen]=8104
    [embeddings]=8105
    [docutils]=8106
    [findata]=8107
)

# Ordered list for startup (lighter services first to avoid GPU contention during load)
ALL_SERVICES=(findata docutils embeddings stt tts imagegen vision)

# If specific services are requested, use those; otherwise start all
if [[ $# -gt 0 ]]; then
    REQUESTED=("$@")
else
    REQUESTED=("${ALL_SERVICES[@]}")
fi

start_service() {
    local name="$1"
    local port="${SERVICE_PORTS[$name]}"
    local venv="$VENVS/$name"
    local server="$SERVICES/$name/server.py"
    local log="$LOG_DIR/multimodal-${name}.log"
    local pidfile="$PID_DIR/multimodal-${name}.pid"

    # Check if already running
    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "  ✓ $name already running (PID $(cat "$pidfile")) on port $port"
        return 0
    fi

    # Verify venv and server exist
    if [[ ! -f "$venv/bin/python3.12" ]]; then
        echo "  ✗ $name: venv not found at $venv"
        return 1
    fi
    if [[ ! -f "$server" ]]; then
        echo "  ✗ $name: server.py not found at $server"
        return 1
    fi

    echo -n "  Starting $name on port $port... "

    # Build LD_LIBRARY_PATH from the venv's nvidia packages (needed for CUDA libs).
    # Falls back to another venv's nvidia libs if this venv doesn't have them (e.g. STT).
    local nvidia_lib=""
    if [[ -d "$venv/lib/python3.12/site-packages/nvidia/cublas/lib" ]]; then
        nvidia_lib="$venv/lib/python3.12/site-packages/nvidia/cublas/lib"
    elif [[ -d "$VENVS/tts/lib/python3.12/site-packages/nvidia/cublas/lib" ]]; then
        nvidia_lib="$VENVS/tts/lib/python3.12/site-packages/nvidia/cublas/lib"
    fi

    # Launch with nohup, redirect output to log
    LD_LIBRARY_PATH="${nvidia_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    nohup "$venv/bin/python3.12" -m uvicorn server:app \
        --host 0.0.0.0 \
        --port "$port" \
        --app-dir "$SERVICES/$name" \
        > "$log" 2>&1 &

    local pid=$!
    echo "$pid" > "$pidfile"
    echo "PID $pid (log: $log)"
}

echo "=== Starting Multimodal Services ==="
echo ""

for svc in "${REQUESTED[@]}"; do
    if [[ -z "${SERVICE_PORTS[$svc]+x}" ]]; then
        echo "  ✗ Unknown service: $svc"
        echo "    Available: ${ALL_SERVICES[*]}"
        continue
    fi
    start_service "$svc"
done

echo ""
echo "=== Startup complete ==="
echo "Run 'bash $BASE/scripts/status.sh' to check health."
