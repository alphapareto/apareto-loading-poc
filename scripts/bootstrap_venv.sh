#!/usr/bin/env bash
# Build $APARETO_HOME/venv with the stock vLLM wheel and the exact pins in requirements.lock, then record a SHA-256
# manifest of the installed vLLM tree. Servers check that manifest at boot and refuse a modified vLLM.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/local_env.sh"

VLLM_VERSION="${VLLM_VERSION:-0.27.1}"
PY_VERSION="${PY_VERSION:-3.12}"
VENV="${VENV:-$APARETO_HOME/venv}"
BOOTSTRAP_TAG="${BOOTSTRAP_TAG:-bootstrap_$(date -u +%Y%m%dT%H%M%S_%NZ)}"
ENV_RECORD="$APARETO_HOME/results/${BOOTSTRAP_TAG}_results.json"
STOCK_MANIFEST="${STOCK_MANIFEST:-$APARETO_HOME/vllm_stock_manifest.sha256}"
VLLM_PATH_RECORD="${VLLM_PATH_RECORD:-$APARETO_HOME/vllm_path}"

echo "=== apareto-loading-poc venv bootstrap ==="
echo "repo        : $REPO_DIR"
echo "venv        : $VENV"
echo "python      : $PY_VERSION"
echo "vllm        : $VLLM_VERSION (stock wheel)"
echo

if [ -e "$VENV" ]; then
    echo "!! $VENV already exists."
    echo "   Remove it explicitly if you want a fresh one:  rm -rf $VENV"
    exit 1
fi
for output in "$VLLM_PATH_RECORD" "$STOCK_MANIFEST" "$ENV_RECORD"; do
    if [ -e "$output" ]; then
        echo "!! $output already exists; preserve it and choose fresh environment/output paths."
        exit 1
    fi
done
mkdir -p "$APARETO_HOME/results"

INSTALL_ARGS=("vllm==$VLLM_VERSION" numpy datasets requests huggingface_hub)
if [ -f requirements.lock ]; then
    INSTALL_ARGS=(-r requirements.lock)
fi

if command -v uv >/dev/null 2>&1; then
    uv venv --python "$PY_VERSION" "$VENV"
    uv pip install --python "$VENV/bin/python" \
        "${INSTALL_ARGS[@]}"
else
    "python$PY_VERSION" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install \
        "${INSTALL_ARGS[@]}"
fi

PY="$VENV/bin/python"

echo
echo "=== record exact versions and the stock vLLM tree ==="
"$PY" scripts/record_environment.py --out "$ENV_RECORD" \
    --manifest "$STOCK_MANIFEST" --vllm-path-record "$VLLM_PATH_RECORD" \
    --lock requirements.lock --expected-vllm "$VLLM_VERSION"

if [ -f artifacts/code/think/MANIFEST.sha256 ]; then
    echo
    echo "=== gate: R* bundles ==="
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" tests/check_bundles.py
fi

echo
echo "venv ready: $VENV (vLLM $VLLM_VERSION, manifest $STOCK_MANIFEST)"
