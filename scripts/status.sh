#!/usr/bin/env bash
# status.sh — Check health of all multimodal services.
#
# Hits /health on each port and reports status.
# Shows which services are enabled vs disabled per services.conf.
# Also shows GPU memory usage if nvidia-smi is available.
#
# Usage:
#   bash /home/mferr/multimodal/scripts/status.sh

set -euo pipefail

BASE="/home/mferr/multimodal"

declare -A SERVICE_PORTS=(
    [stt]=8101
    [vision]=8102
    [tts]=8103
    [imagegen]=8104
    [embeddings]=8105
    [docutils]=8106
    [findata]=8107
)

# Ordered for display
ALL_SERVICES=(stt vision tts imagegen embeddings docutils findata)

PID_DIR="/tmp"

# Load enabled services from config
ENABLED_SERVICES=()
if [[ -f "$BASE/services.conf" ]]; then
    source "$BASE/services.conf"
fi

# Helper: check if a service is in the enabled list
is_enabled() {
    local name="$1"
    # If no config loaded, treat everything as enabled
    if [[ ${#ENABLED_SERVICES[@]} -eq 0 ]]; then
        return 0
    fi
    for s in "${ENABLED_SERVICES[@]}"; do
        [[ "$s" == "$name" ]] && return 0
    done
    return 1
}

echo "=== Multimodal Service Status ==="
echo "  Config: $BASE/services.conf"
if [[ ${#ENABLED_SERVICES[@]} -gt 0 ]]; then
    echo "  Enabled: ${ENABLED_SERVICES[*]}"
fi
echo ""
printf "%-12s %-6s %-8s %-10s %-9s %s\n" "SERVICE" "PORT" "PID" "STATUS" "ENABLED" "DETAILS"
printf "%-12s %-6s %-8s %-10s %-9s %s\n" "-------" "----" "---" "------" "-------" "-------"

for svc in "${ALL_SERVICES[@]}"; do
    port="${SERVICE_PORTS[$svc]}"
    pidfile="$PID_DIR/multimodal-${svc}.pid"

    # Check enabled
    if is_enabled "$svc"; then
        enabled="yes"
    else
        enabled="no"
    fi

    # Check PID
    pid="-"
    if [[ -f "$pidfile" ]]; then
        pid=$(cat "$pidfile")
        if ! kill -0 "$pid" 2>/dev/null; then
            pid="dead"
        fi
    fi

    # Check health endpoint
    status="DOWN"
    details=""
    response=$(curl -s -m 2 "http://localhost:${port}/health" 2>/dev/null || true)
    if [[ -n "$response" ]]; then
        svc_status=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null || echo "?")
        if [[ "$svc_status" == "ok" ]]; then
            status="UP"
        elif [[ "$svc_status" == "loading" ]]; then
            status="LOADING"
        fi
        details="$response"
    fi

    printf "%-12s %-6s %-8s %-10s %-9s %s\n" "$svc" "$port" "$pid" "$status" "$enabled" "$details"
done

echo ""

# Show GPU memory if nvidia-smi is available
if command -v nvidia-smi &>/dev/null; then
    echo "=== GPU Memory ==="
    nvidia-smi --query-gpu=name,memory.used,memory.total,memory.free,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' '{printf "  %s: %s/%s MiB used (%s MiB free), %s%% util\n", $1, $2, $3, $4, $5}'
    echo ""
fi
