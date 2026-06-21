#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VOLUME_ROOT="${GGUF_VOLUME_ROOT:-/workspace}"
LLAMA_DIR="${LLAMA_CPP_DIR:-$VOLUME_ROOT/llama.cpp}"
VENV_DIR="${GGUF_VENV_DIR:-$VOLUME_ROOT/.venvs/gguf-rig}"
PYTHON_EXE="${PYTHON_EXE:-python3}"

info() { printf '\033[0;34m%s\033[0m\n' "$*"; }
success() { printf '\033[0;32m%s\033[0m\n' "$*"; }
error() { printf '\033[0;31m%s\033[0m\n' "$*" >&2; }

on_error() {
    local exit_code=$?
    error "Installation failed at line $1 (exit $exit_code)."
    exit "$exit_code"
}
trap 'on_error "$LINENO"' ERR

if [[ ! -d "$VOLUME_ROOT" ]]; then
    error "Persistent volume root does not exist: $VOLUME_ROOT"
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
info "Installing llama.cpp build prerequisites..."
apt-get update
apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake git libcurl4-openssl-dev ninja-build python3-venv

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    info "Cloning llama.cpp into persistent storage..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
elif [[ "${GGUF_SKIP_UPDATE:-0}" != "1" ]]; then
    info "Updating llama.cpp..."
    git -C "$LLAMA_DIR" pull --ff-only
fi

info "Building CUDA llama-server..."
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -G Ninja \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_DIR/build" --target llama-server --parallel "$(nproc)"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    info "Creating the persistent Python environment..."
    "$PYTHON_EXE" -m venv "$VENV_DIR"
fi

info "Installing control-panel dependencies..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

mkdir -p \
    "$VOLUME_ROOT/models/gguf" \
    "$VOLUME_ROOT/.state/gguf-rig" \
    "$VOLUME_ROOT/logs/gguf-rig" \
    "$VOLUME_ROOT/.hf/hub"

success "GGUF Rig is installed. llama-server: $LLAMA_DIR/build/bin/llama-server"
