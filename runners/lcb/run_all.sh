#!/usr/bin/env bash
# Red Hat's three seeds (42, 1234, 4158), spread over the listed GPUs; each GPU runs its seeds one after another,
# one fresh server per seed. The machine's cores are split evenly between the GPUs' servers (the CPU expert threads
# of each take the last cores of its share).
#
#   bash runners/lcb/run_all.sh            # GPU 0 only
#   DEVICES="0 1" bash runners/lcb/run_all.sh
#   SERVER_ARGS='--budget 4608' DEVICES="0 1" bash runners/lcb/run_all.sh
#
# Then: python3 runners/lcb/summarize.py ~/.apareto/results/lcb/lcb_s*     ($APARETO_HOME/results/lcb)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
read -r -a DEVICES <<< "${DEVICES:-0}"
SEEDS=(${SEEDS:-42 1234 4158})
declare -A QUEUE
for i in "${!SEEDS[@]}"; do d=${DEVICES[$((i % ${#DEVICES[@]}))]}; QUEUE[$d]+="${SEEDS[$i]} "; done
mapfile -t CORES < <(python3 -c 'import os; print("\n".join(map(str, sorted(os.sched_getaffinity(0)))))')
per=$(( ${#CORES[@]} / ${#DEVICES[@]} ))
pids=()
for i in "${!DEVICES[@]}"; do
  d=${DEVICES[$i]}
  cpus=$(IFS=,; echo "${CORES[*]:$((i * per)):$per}")
  ( for s in ${QUEUE[$d]}; do CPUS="$cpus" bash "$HERE/run.sh" "$s" "$d"; done ) &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
