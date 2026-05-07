#!/bin/bash
# ============================================================================
# vllm_utils.sh — Shared vLLM server lifecycle management
#
# Source this file in run_*.sh scripts:
#   source "$(dirname "$0")/scripts/vllm_utils.sh"
#
# Provides:
#   start_vllm <config_yaml> <log_name>   — Start vLLM in background
#   wait_for_vllm <api_base>              — Poll until server is ready
#   stop_vllm                             — Kill vLLM process
#   port_is_busy <port>                   — Check if port is in use
# ============================================================================

VLLM_PID=""
VLLM_PORT=""
VLLM_LOG_FILE=""
VLLM_LOG_DIR="logs/vllm"
VLLM_POLL_INTERVAL=3   # seconds between health checks

# ──────────────────────────────────────────────────────────────
# start_vllm <config_yaml> <log_name>
#
# Launches vLLM server in background, redirecting output to a log file.
# Sets VLLM_PID for later cleanup.
# ──────────────────────────────────────────────────────────────
start_vllm() {
    local config="$1"
    local log_name="$2"

    if [ ! -f "$config" ]; then
        echo "❌ vLLM config not found: $config"
        exit 1
    fi

    mkdir -p "$VLLM_LOG_DIR"
    local log_file="${VLLM_LOG_DIR}/${log_name}_$(date +%Y%m%d_%H%M%S).log"
    VLLM_LOG_FILE="$log_file"  # expose for wait_for_vllm
    VLLM_PORT=$(get_vllm_port "$config")  # expose for stop_vllm orphan cleanup

    echo ""
    echo "🚀 [vLLM] Starting server..."
    echo "   Config: $config"
    echo "   Log:    $log_file"

    python src/start_vllm.py --config "$config" > "$log_file" 2>&1 &
    VLLM_PID=$!
    echo "   PID:    $VLLM_PID"
}

# ──────────────────────────────────────────────────────────────
# wait_for_vllm <api_base>
#
# Polls the /models endpoint every VLLM_POLL_INTERVAL seconds.
# Fails immediately if the vLLM process dies (no wasted waiting).
# ──────────────────────────────────────────────────────────────
wait_for_vllm() {
    local api_base="$1"
    local url="${api_base}/models"
    local elapsed=0
    local last_log_lines=0

    echo "⏳ [vLLM] Waiting for server at $url ..."
    echo "   (polling every ${VLLM_POLL_INTERVAL}s, will fail immediately if process dies)"
    echo "   ─── vLLM startup log ───"

    while true; do
        # Check if vLLM process is still alive
        if [ -n "$VLLM_PID" ] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo ""
            echo "❌ [vLLM] Process died unexpectedly (PID: $VLLM_PID)"
            # Show last 20 lines of log for debugging
            if [ -n "$VLLM_LOG_FILE" ] && [ -f "$VLLM_LOG_FILE" ]; then
                echo "   ─── Last 20 lines of log ───"
                tail -20 "$VLLM_LOG_FILE" | sed 's/^/   │ /'
            fi
            # NOTE: Do NOT clear VLLM_PID here — let stop_vllm (via trap)
            # handle cleanup of orphaned child processes (EngineCore, etc.)
            exit 1
        fi

        # Show new log lines (real-time progress)
        if [ -n "$VLLM_LOG_FILE" ] && [ -f "$VLLM_LOG_FILE" ]; then
            local current_lines
            current_lines=$(wc -l < "$VLLM_LOG_FILE" 2>/dev/null || echo 0)
            if [ "$current_lines" -gt "$last_log_lines" ]; then
                tail -n +$((last_log_lines + 1)) "$VLLM_LOG_FILE" | head -n $((current_lines - last_log_lines)) | sed 's/^/   │ /'
                last_log_lines=$current_lines
            fi
        fi

        # Try health check
        if curl -s --connect-timeout 2 "$url" > /dev/null 2>&1; then
            echo "   ───────────────────────"
            echo "✅ [vLLM] Server is ready! (took ${elapsed}s)"
            return 0
        fi

        sleep "$VLLM_POLL_INTERVAL"
        elapsed=$((elapsed + VLLM_POLL_INTERVAL))
    done
}

# ──────────────────────────────────────────────────────────────
# stop_vllm
#
# Stops only the vLLM instance managed by THIS script (by PID).
# Safe to call multiple times. Safe to run in parallel with
# other run_*.sh scripts on the same node.
# ──────────────────────────────────────────────────────────────
stop_vllm() {
    if [ -z "$VLLM_PID" ] && [ -z "$VLLM_PORT" ]; then
        echo "🛑 [vLLM] No vLLM instance tracked — nothing to stop."
        return 0
    fi

    echo "🛑 [vLLM] Stopping vLLM (PID: ${VLLM_PID:-n/a}, Port: ${VLLM_PORT:-n/a})..."

    # Kill ONLY the process tree we started (safe for shared nodes).
    # vLLM forks EngineCore sub-processes that `pkill -P` misses because they
    # are grandchildren of the tracked PID, not direct children.
    if [ -n "$VLLM_PID" ]; then
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            # Collect entire process tree rooted at VLLM_PID
            local tree_pids
            tree_pids=$(_get_process_tree "$VLLM_PID")

            # Graceful SIGTERM to all processes in tree
            echo "$tree_pids" | xargs kill 2>/dev/null || true
            sleep 3

            # Force SIGKILL any survivors
            for pid in $tree_pids; do
                kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
            done
            sleep 1
        fi
    fi

    VLLM_PID=""

    # Wait for OUR GPU memory to be released (observation only, no force-kill).
    # On shared nodes, other scripts may be using GPU memory — we must NOT
    # kill their processes.
    if command -v nvidia-smi &>/dev/null; then
        echo "   Waiting for GPU memory release..."
        sleep 5  # Give CUDA contexts time to release
    fi

    echo "   Done."
}

# Helper: recursively collect all PIDs in a process tree
_get_process_tree() {
    local parent_pid="$1"
    local children
    children=$(ps -o pid= --ppid "$parent_pid" 2>/dev/null || true)
    echo "$parent_pid"
    for child in $children; do
        _get_process_tree "$child"
    done
}

# ──────────────────────────────────────────────────────────────
# port_is_busy <port>
#
# Returns 0 (true) if the port is already in use.
# Supports both macOS (lsof) and Linux (ss) environments.
# ──────────────────────────────────────────────────────────────
port_is_busy() {
    local port="$1"
    if command -v lsof &>/dev/null; then
        lsof -i ":$port" -sTCP:LISTEN &>/dev/null
    elif command -v ss &>/dev/null; then
        ss -tlnp 2>/dev/null | grep -q ":${port} "
    else
        # Fallback: try to connect
        curl -s --connect-timeout 1 "http://localhost:$port" &>/dev/null
    fi
}

# ──────────────────────────────────────────────────────────────
# get_vllm_port <vllm_yaml>
#
# Extract the port number from a vLLM YAML config file.
# ──────────────────────────────────────────────────────────────
get_vllm_port() {
    local yaml_file="$1"
    grep '^port:' "$yaml_file" 2>/dev/null | head -1 | awk '{print $2}'
}
