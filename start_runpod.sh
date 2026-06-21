#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VOLUME_ROOT="${GGUF_VOLUME_ROOT:-/workspace}"
VENV_DIR="${GGUF_VENV_DIR:-$VOLUME_ROOT/.venvs/gguf-rig}"
LOG_FILE="${GGUF_STARTUP_LOG:-$VOLUME_ROOT/logs/gguf-rig/startup.log}"

mkdir -p "$(dirname -- "$LOG_FILE")"
exec > >(tee -i "$LOG_FILE") 2>&1

echo "=== GGUF Rig startup: $(date --iso-8601=seconds) ==="

export GGUF_VOLUME_ROOT="$VOLUME_ROOT"
export GGUF_MODELS_DIR="${GGUF_MODELS_DIR:-$VOLUME_ROOT/models/gguf}"
export GGUF_STATE_DIR="${GGUF_STATE_DIR:-$VOLUME_ROOT/.state/gguf-rig}"
export GGUF_LOG_DIR="${GGUF_LOG_DIR:-$VOLUME_ROOT/logs/gguf-rig}"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$VOLUME_ROOT/llama.cpp/build/bin/llama-server}"
export HF_HOME="${HF_HOME:-$VOLUME_ROOT/.hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$VOLUME_ROOT/.hf/hub}"

: "${GGUF_API_KEY:?Set GGUF_API_KEY as a RunPod secret}"
: "${GGUF_PANEL_USER:?Set GGUF_PANEL_USER as a RunPod secret}"
: "${GGUF_PANEL_PASSWORD:?Set GGUF_PANEL_PASSWORD as a RunPod secret}"

if [[ "${GGUF_SKIP_INSTALL:-0}" != "1" || ! -x "$LLAMA_SERVER_BIN" || ! -x "$VENV_DIR/bin/python" ]]; then
    bash "$SCRIPT_DIR/install_runpod.sh"
fi

exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/app.py"
