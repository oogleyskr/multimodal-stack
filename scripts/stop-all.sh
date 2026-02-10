#!/usr/bin/env bash
# stop-all.sh — Stop all multimodal services.
#
# Reads PID files from /tmp/multimodal-*.pid and sends SIGTERM.
# Falls back to killing by port if PID file is missing.
#
# Usage:
#   bash /home/mferr/multimodal/scripts/stop-all.sh          # Stop all
#   bash /home/mferr/multimodal/scripts/stop-all.sh stt tts   # Stop specific services

set -euo pipefail

PID_DIR="/tmp"

declare -A SERVICE_PORTS=(
    [stt]=8101
    [vision]=8102
    [tts]=8103
    [imagegen]=8104
    [embeddings]=8105
    [docutils]=8106
    [findata]=8107
)

ALL_SERVICES=(stt vision tts imagegen embeddings docutils findata)

if [[ $# -gt 0 ]]; then
    REQUESTED=("$@")
else
    REQUESTED=("${ALL_SERVICES[@]}")
fi

stop_service() {
    local name="$1"
    local port="${SERVICE_PORTS[$name]}"
    local pidfile="$PID_DIR/multimodal-${name}.pid"

    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "  ✓ $name stopped (PID $pid)"
        else
            echo "  - $name was not running (stale PID $pid)"
        fi
        rm -f "$pidfile"
    else
        # Try to find by port as fallback
        local pid
        pid=$(lsof -ti :"$port" 2>/dev/null || true)
        if [[ -n "$pid" ]]; then
            kill "$pid"
            echo "  ✓ $name stopped (found PID $pid on port $port)"
        else
            echo "  - $name was not running"
        fi
    fi
}

echo "=== Stopping Multimodal Services ==="
echo ""

for svc in "${REQUESTED[@]}"; do
    if [[ -z "${SERVICE_PORTS[$svc]+x}" ]]; then
        echo "  ✗ Unknown service: $svc"
        continue
    fi
    stop_service "$svc"
done

echo ""
echo "=== Shutdown complete ==="
