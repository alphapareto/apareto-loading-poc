#!/usr/bin/env bash
# LiveCodeBench v6 with Red Hat's harness against the apareto server: one seed, one GPU.
#
#   bash runners/lcb/run.sh <seed> [device] [port]        e.g.  bash runners/lcb/run.sh 42 0 8000
#
# Boots scripts/serve.py --profile code, waits for it, runs
#   lighteval endpoint litellm litellm_config.yaml "lcb:codegeneration_v6|0" --output-dir ... --save-details
# exactly as on the model card, then stops the server. Everything lands in $APARETO_HOME/results/lcb/<tag>/ plus
# the server's own $APARETO_HOME/results/<tag>_server_* files (memory samples, per-request selections and timings).
# Red Hat's published number is the mean over seeds 42, 1234 and 4158: see runners/lcb/run_all.sh.
#
#   SERVER_ARGS='--budget 4608'   options for scripts/serve.py after `--` (override the code profile)
#   CPUS=0-3                      the cores this run's server may use (run_all.sh splits the machine per GPU)
#   LCB_MAX_SAMPLES=6             smoke runs only: lighteval's first N problems (not a benchmark result)
#
# lighteval's concurrent_requests is the server's max_num_seqs (3 in the code profile), read from ready.json.
set -uo pipefail
SEED="${1:?usage: run.sh <seed> [device] [port]}"; DEVICE="${2:-0}"; PORT="${3:-$((8000 + DEVICE))}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT" || exit 1
. scripts/local_env.sh
LIGHTEVAL="$APARETO_HOME/venv-lcb/bin/lighteval"; PY="$APARETO_HOME/venv/bin/python"
[ -x "$LIGHTEVAL" ] || { echo "run runners/lcb/install.sh first"; exit 1; }

TAG="lcb_s${SEED}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$APARETO_HOME/results/lcb/$TAG"; mkdir -p "$OUT"
STAG="${TAG}_server"; READY="$APARETO_HOME/results/${STAG}_ready.json"
echo "RUN_START $TAG device=$DEVICE port=$PORT $(date -Is)"

# shellcheck disable=SC2086
${CPUS:+taskset -c "$CPUS"} "$PY" scripts/serve.py --profile code --device "$DEVICE" --port "$PORT" \
    --tag "$STAG" ${SERVER_ARGS:+-- $SERVER_ARGS} > "$OUT/server_stdout.log" 2>&1 &
SRV=$!
stop_server() {
  if kill -0 "$SRV" 2>/dev/null; then
    kill -TERM "$SRV"
    for _ in $(seq 1 90); do kill -0 "$SRV" 2>/dev/null || break; sleep 2; done
    kill -0 "$SRV" 2>/dev/null && kill -9 "$SRV"
  fi
}
trap stop_server EXIT; trap 'exit 143' INT TERM

for _ in $(seq 1 600); do
  [ -s "$READY" ] && break
  kill -0 "$SRV" 2>/dev/null || { echo "FATAL: server exited during boot"; tail -20 "$OUT/server_stdout.log"; exit 1; }
  sleep 5
done
[ -s "$READY" ] || { echo "FATAL: server not ready"; exit 1; }
echo "SERVER READY $(cat "$READY")"
CONC="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("max_num_seqs", 1))' "$READY")"

sed -e "s|@SERVED_MODEL_NAME@|qwen3.6-35b|" -e "s|@BASE_URL@|http://127.0.0.1:${PORT}/v1|" \
    -e "s|@CONCURRENT_REQUESTS@|${CONC}|" -e "s|@SEED@|${SEED}|" -e "s|@CACHE_DIR@|${OUT}/lighteval_cache|" \
    "$HERE/litellm_config.template.yaml" > "$OUT/litellm_config.yaml"

# From the run's own directory: litellm keeps its response cache (.litellm_cache) in the working directory, and a
# shared one would answer a later run's identical requests from this run's generations.
cd "$OUT" || exit 1
"$LIGHTEVAL" endpoint litellm "$OUT/litellm_config.yaml" "lcb:codegeneration_v6|0" \
    --output-dir "$OUT" --save-details ${LCB_MAX_SAMPLES:+--max-samples "$LCB_MAX_SAMPLES"} 2>&1 | tee "$OUT/lighteval.log"
rc=${PIPESTATUS[0]}
echo "RUN_END $TAG lighteval rc=$rc $(date -Is)"
exit "$rc"
