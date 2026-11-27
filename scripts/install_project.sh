#!/usr/bin/env bash
# Install only the repository's adapter package; keep the pinned stock dependencies.
set -euo pipefail
APARETO_INSTALL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APARETO_INSTALL_ROOT"
source scripts/local_env.sh
OUT="$APARETO_HOME/results"
APARETO_INSTALL_TAG="${1:?usage: bash scripts/install_project.sh NEW_TAG}"
if [[ ! "$APARETO_INSTALL_TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'Invalid tag' >&2
    exit 1
fi
for suffix in results.json manifest.sha256 vllm_path.txt requirements.lock; do
    if [ -e "$OUT/${APARETO_INSTALL_TAG}_${suffix}" ]; then
        echo "Output exists: $OUT/${APARETO_INSTALL_TAG}_${suffix}" >&2
        exit 1
    fi
done
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$APARETO_HOME/venv/bin/python" --no-deps --no-build-isolation -e .
else
    "$APARETO_HOME/venv/bin/python" -m pip install --no-deps --no-build-isolation -e .
fi
"$APARETO_HOME/venv/bin/python" scripts/record_environment.py \
  --out "$OUT/${APARETO_INSTALL_TAG}_results.json" \
  --manifest "$OUT/${APARETO_INSTALL_TAG}_manifest.sha256" \
  --vllm-path-record "$OUT/${APARETO_INSTALL_TAG}_vllm_path.txt" \
  --lock "$OUT/${APARETO_INSTALL_TAG}_requirements.lock" --expected-vllm 0.27.1
