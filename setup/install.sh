#!/usr/bin/env bash
# One-shot setup: environment, plugin, checkpoint, expert stores, and (optionally) the benchmark runners.
#
#   bash setup/install.sh                    # everything the server needs
#   bash setup/install.sh --with-runners     # also the LiveCodeBench harness venv
#
# Options:
#   --model-dir DIR    use (or download into) DIR instead of $APARETO_HOME/models/Qwen3.6-35B-A3B-NVFP4
#   --stores-dir DIR   build the expert stores in DIR instead of $APARETO_HOME/stores
#   --skip-model       do not download the checkpoint (it must already be at --model-dir)
#   --skip-stores      do not build the expert stores
#   --no-prebuild      do not compile the JIT kernels now (the first server boot compiles them instead)
#   --with-runners     run runners/lcb/install.sh
#
# Every step is skipped when its output already exists, so the script can be re-run after a failure.
# The repository holds code and R* bundles only: the environments, caches, run outputs, checkpoint and stores go
# under APARETO_HOME (default ~/.apareto), or wherever APARETO_MODEL / APARETO_STORES point.
# The setup never uses a GPU: the kernel builds only compile.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_REPO=RedHatAI/Qwen3.6-35B-A3B-NVFP4
MODEL_REVISION=a1ff8c8967426709e5528307c14c52c17616b008
export APARETO_HOME="${APARETO_HOME:-$HOME/.apareto}"
MODEL="${APARETO_MODEL:-$APARETO_HOME/models/Qwen3.6-35B-A3B-NVFP4}"
STORES="${APARETO_STORES:-$APARETO_HOME/stores}"
SKIP_MODEL=0 SKIP_STORES=0 PREBUILD=1 RUNNERS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model-dir) MODEL="$(realpath -m "$2")"; shift 2 ;;
    --stores-dir) STORES="$(realpath -m "$2")"; shift 2 ;;
    --skip-model) SKIP_MODEL=1; shift ;;
    --skip-stores) SKIP_STORES=1; shift ;;
    --no-prebuild) PREBUILD=0; shift ;;
    --with-runners) RUNNERS=1; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
export APARETO_MODEL="$MODEL" APARETO_STORES="$STORES"
. scripts/local_env.sh
VENV="$APARETO_HOME/venv"
PY="$VENV/bin/python"
MANIFEST="$APARETO_HOME/vllm_stock_manifest.sha256"
step() { echo; echo "=== $*"; }
die() { echo "!! $*" >&2; exit 1; }

step "1/7 prerequisites"
command -v uv >/dev/null || command -v python3.12 >/dev/null || die "need uv (recommended) or python3.12"
command -v nvidia-smi >/dev/null || die "need an NVIDIA driver (nvidia-smi not found)"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader | sed 's/^/  gpu /'
if ! nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -q '^12\.0'; then
  echo "  warning: no compute capability 12.0 (SM120) GPU found; the MoE kernels are built for SM120 only"
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
[ -x "$CUDA_HOME/bin/nvcc" ] || die "need the CUDA 13 toolkit (nvcc) at \$CUDA_HOME ($CUDA_HOME) for the CPU-expert kernels"
echo "  nvcc $("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
echo "  host RAM $(awk '/MemTotal/ {printf "%.0f GiB", $2/1048576}' /proc/meminfo)"
mkdir -p "$(dirname "$STORES")"
echo "  free disk at stores: $(df -h --output=avail "$(dirname "$STORES")" | tail -1 | tr -d ' ') (needs ~40 GB while building, ~18 GB after)"

step "2/7 Python environment ($VENV, stock vLLM 0.27.1)"
if [ -x "$PY" ] && [ -f "$MANIFEST" ]; then
  echo "  $VENV exists; keeping it"
else
  [ -e "$VENV" ] && die "$VENV exists but $MANIFEST does not; remove $VENV and re-run"
  bash scripts/bootstrap_venv.sh
fi
if ! cmp -s "$MANIFEST" config/vllm-0.27.1-stock.sha256; then
  die "the installed vLLM tree differs from config/vllm-0.27.1-stock.sha256 (diff it with $MANIFEST); rebuild $VENV"
fi
echo "  vLLM tree matches the reference manifest"

step "3/7 apareto-loading-poc plugin (editable, no dependency changes)"
if "$PY" -c 'import importlib.metadata as m; m.distribution("apareto-loading-poc")' 2>/dev/null; then
  echo "  installed"
else
  bash scripts/install_project.sh "install_$(date -u +%Y%m%dT%H%M%SZ)"
fi
"$PY" - <<'EOF'
from importlib.metadata import entry_points
names = [e.name for e in entry_points(group='vllm.general_plugins')]
assert 'apareto' in names, names
print('  vLLM plugin entry point registered:', names)
EOF

step "4/7 JIT kernels (FlashInfer MoE slot kernel, CPU-expert server)"
if [ "$PREBUILD" = 1 ]; then
  MAX_JOBS="${MAX_JOBS:-4}" TORCH_CUDA_ARCH_LIST=12.0 "$PY" - <<'EOF'
import time
t = time.time()
from apareto_loader.vllm027 import moe_slots
moe_slots.load()
print(f'  moe_slots built ({time.time() - t:.0f} s)')
t = time.time()
from apareto_loader.vllm027 import cpu_experts
cpu_experts.module()
print(f'  cpu_experts built ({time.time() - t:.0f} s)')
EOF
else
  echo "  skipped (--no-prebuild); the first boot compiles them"
fi

step "5/7 checkpoint $MODEL_REPO @ ${MODEL_REVISION:0:8}"
if [ -f "$MODEL/config.json" ] && [ -f "$MODEL/model.safetensors.index.json" ]; then
  echo "  present at $MODEL"
elif [ "$SKIP_MODEL" = 1 ]; then
  die "--skip-model given but no checkpoint at $MODEL"
else
  "$VENV/bin/hf" download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$MODEL" \
      --include '*.json' '*.jinja' '*.safetensors' '*.yaml' 'README.md'
fi
"$PY" - "$MODEL" <<'EOF'
import json, sys
c = json.load(open(f'{sys.argv[1]}/config.json'))
t = c.get('text_config', c)
layers, experts = t.get('num_hidden_layers'), t.get('num_experts')
assert (layers, experts) == (40, 256), f'geometry {layers} layers x {experts} experts; this release needs 40 x 256'
print(f'  geometry {layers} layers x {experts} experts')
EOF

step "6/7 expert stores in $STORES"
if [ -f "$STORES/layer_store/experts.apkl" ] && [ -f "$STORES/expert_store_header.json" ]; then
  echo "  present"
elif [ "$SKIP_STORES" = 1 ]; then
  echo "  skipped (--skip-stores); run setup/build_stores.sh before serving"
else
  bash setup/build_stores.sh
fi

step "7/7 R* bundles"
ART="${APARETO_ARTIFACTS:-$ROOT/artifacts}"
if [ -f "$ART/code/think/MANIFEST.sha256" ]; then
  APARETO_ARTIFACTS="$ART" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" tests/check_bundles.py
else
  echo "  FATAL: no R* bundles in $ART (docs/INSTALL.md, 'R* bundles')"; exit 1
fi

if [ "$RUNNERS" = 1 ]; then
  step "runners"
  if [ -x "$APARETO_HOME/venv-lcb/bin/lighteval" ]; then echo "  lighteval installed"; else bash runners/lcb/install.sh; fi
fi

cat <<EOF

Setup done.
  Serve:   python3 scripts/serve.py --profile code       (LiveCodeBench settings)
  State:   $APARETO_HOME (environments, caches, results)
  Model:   $MODEL
  Stores:  $STORES
Export APARETO_MODEL / APARETO_STORES if you used non-default locations.
EOF
