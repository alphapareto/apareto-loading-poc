#!/usr/bin/env bash
# Build the expert stores the server reads, from the downloaded checkpoint.
#
#   bash setup/build_stores.sh [--keep-packed]
#
# 1. scripts/build_expert_store.py packs every expert's bytes in expert-id order (stores/experts.apex).
# 2. scripts/build_layer_store.py repacks it layer-contiguous for one read per layer, verifying every record's
#    payload on the way (stores/layer_store/experts.apkl, ~18 GB).
# 3. scripts/write_store_header.py keeps the packed store's header (stores/expert_store_header.json); the server
#    checks the served checkpoint against it at every boot.
# 4. The packed store is deleted unless --keep-packed is given.
#
# Needs about 40 GB free while it runs (both stores at once), 18 GB afterwards. Put stores/ on NVMe: the host
# tier reads from it during prefill.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
. scripts/local_env.sh
export APARETO_HOME="${APARETO_HOME:-$HOME/.apareto}"
PY="$APARETO_HOME/venv/bin/python"
MODEL="${APARETO_MODEL:-$APARETO_HOME/models/Qwen3.6-35B-A3B-NVFP4}"
STORES="${APARETO_STORES:-$APARETO_HOME/stores}"
KEEP=0; [ "${1:-}" = "--keep-packed" ] && KEEP=1

[ -d "$MODEL" ] || { echo "Model not found at $MODEL (run setup/install.sh first, or set APARETO_MODEL)"; exit 1; }
for out in "$STORES/layer_store/experts.apkl" "$STORES/expert_store_header.json"; do
  [ -e "$out" ] && { echo "$out already exists; remove it to rebuild"; exit 1; }
done
mkdir -p "$STORES/layer_store" "$STORES/reports"
stamp=$(date -u +%Y%m%dT%H%M%SZ)

echo "== packed store"
[ -e "$STORES/experts.apex" ] || "$PY" scripts/build_expert_store.py --model "$MODEL" --out "$STORES/experts.apex" \
    --report "$STORES/reports/expert_store_$stamp.json"
echo "== layer store (verifies every record)"
"$PY" scripts/build_layer_store.py --store "$STORES/experts.apex" --out "$STORES/layer_store/experts.apkl" \
    --report "$STORES/reports/layer_store_$stamp.json"
echo "== header"
"$PY" scripts/write_store_header.py --store "$STORES/experts.apex" --out "$STORES/expert_store_header.json"
if [ "$KEEP" = 0 ]; then rm -f "$STORES/experts.apex"; echo "removed the packed store"; fi
echo "stores ready in $STORES"
