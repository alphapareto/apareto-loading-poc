#!/usr/bin/env bash
# LiveCodeBench v6, one seed spread over several GPUs: one apareto server per GPU and one lighteval run against all.
#
#   bash runners/lcb/run_multi.sh <seed> "<devices>" [port]      e.g.  bash runners/lcb/run_multi.sh 42 "0 1"
#
# Boots scripts/serve.py --profile code on every listed GPU (ports 8000 + device), each on an equal share of the
# machine's cores, and runners/lcb/balance.py on [port] (default 8100), which sends every request to the server with
# the fewest requests in flight. lighteval runs once, as in runners/lcb/run.sh, with concurrent_requests set to the
# sum of the servers' max_num_seqs. The run takes about 1/n of the time of run.sh on n GPUs.
#
# Everything lands in $APARETO_HOME/results/lcb/<tag>/ (plus balance.jsonl: which server served each request) and
# each server's own $APARETO_HOME/results/<tag>_server_gpu<device>_* files.
#
#   SERVER_ARGS='--budget 4608'   options for scripts/serve.py after `--` (override the code profile)
#   LCB_MAX_SAMPLES=6             smoke runs only: lighteval's first N problems (not a benchmark result)
set -uo pipefail
SEED="${1:?usage: run_multi.sh <seed> \"<devices>\" [port]}"
read -r -a DEVICES <<< "${2:?usage: run_multi.sh <seed> \"<devices>\" [port]}"
PORT="${3:-8100}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT" || exit 1
. scripts/local_env.sh
LIGHTEVAL="$APARETO_HOME/venv-lcb/bin/lighteval"; PY="$APARETO_HOME/venv/bin/python"
[ -x "$LIGHTEVAL" ] || { echo "run runners/lcb/install.sh first"; exit 1; }

TAG="lcb_s${SEED}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$APARETO_HOME/results/lcb/$TAG"; mkdir -p "$OUT"
echo "RUN_START $TAG devices=${DEVICES[*]} port=$PORT $(date -Is)"

mapfile -t CORES < <(python3 -c 'import os; print("\n".join(map(str, sorted(os.sched_getaffinity(0)))))')
per=$(( ${#CORES[@]} / ${#DEVICES[@]} ))
PIDS=(); READYS=(); URLS=()
for i in "${!DEVICES[@]}"; do
  d=${DEVICES[$i]}
  cpus=$(IFS=,; echo "${CORES[*]:$((i * per)):$per}")
  stag="${TAG}_server_gpu${d}"
  # shellcheck disable=SC2086
  taskset -c "$cpus" "$PY" scripts/serve.py --profile code --device "$d" --port "$((8000 + d))" \
      --tag "$stag" ${SERVER_ARGS:+-- $SERVER_ARGS} > "$OUT/server_gpu${d}_stdout.log" 2>&1 &
  PIDS+=($!); READYS+=("$APARETO_HOME/results/${stag}_ready.json"); URLS+=("http://127.0.0.1:$((8000 + d))")
done
BAL=
stop_all() {
  [ -n "$BAL" ] && kill -TERM "$BAL" 2>/dev/null
  for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && kill -TERM "$p"; done
  for p in "${PIDS[@]}"; do
    for _ in $(seq 1 90); do kill -0 "$p" 2>/dev/null || break; sleep 2; done
    kill -0 "$p" 2>/dev/null && kill -9 "$p"
  done
}
trap stop_all EXIT; trap 'exit 143' INT TERM

for _ in $(seq 1 600); do
  ready=0
  for i in "${!PIDS[@]}"; do
    [ -s "${READYS[$i]}" ] && ready=$((ready + 1))
    kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "FATAL: the server on GPU ${DEVICES[$i]} exited during boot"
                                           tail -20 "$OUT/server_gpu${DEVICES[$i]}_stdout.log"; exit 1; }
  done
  [ "$ready" = "${#PIDS[@]}" ] && break
  sleep 5
done
[ "$ready" = "${#PIDS[@]}" ] || { echo "FATAL: servers not ready"; exit 1; }
CONC="$("$PY" -c 'import json,sys; print(sum(json.load(open(p)).get("max_num_seqs", 1) for p in sys.argv[1:]))' "${READYS[@]}")"
echo "SERVERS READY ${URLS[*]} concurrent_requests=$CONC"

"$PY" "$HERE/balance.py" --port "$PORT" --log "$OUT/balance.jsonl" "${URLS[@]}" > "$OUT/balance_stdout.log" 2>&1 &
BAL=$!
for _ in $(seq 1 30); do
  "$PY" -c 'import sys,urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)' \
      "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null && break
  kill -0 "$BAL" 2>/dev/null || { echo "FATAL: the balancer exited"; cat "$OUT/balance_stdout.log"; exit 1; }
  sleep 1
done

sed -e "s|@SERVED_MODEL_NAME@|qwen3.6-35b|" -e "s|@BASE_URL@|http://127.0.0.1:${PORT}/v1|" \
    -e "s|@CONCURRENT_REQUESTS@|${CONC}|" -e "s|@SEED@|${SEED}|" -e "s|@CACHE_DIR@|${OUT}/lighteval_cache|" \
    "$HERE/litellm_config.template.yaml" > "$OUT/litellm_config.yaml"

# From the run's own directory, as in run.sh: litellm keeps its response cache in the working directory.
cd "$OUT" || exit 1
"$LIGHTEVAL" endpoint litellm "$OUT/litellm_config.yaml" "lcb:codegeneration_v6|0" \
    --output-dir "$OUT" --save-details ${LCB_MAX_SAMPLES:+--max-samples "$LCB_MAX_SAMPLES"} 2>&1 | tee "$OUT/lighteval.log"
rc=${PIPESTATUS[0]}
echo "RUN_END $TAG lighteval rc=$rc $(date -Is)"
exit "$rc"
