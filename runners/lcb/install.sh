#!/usr/bin/env bash
# Install the LiveCodeBench harness Red Hat used: lighteval at commit 6f0f351 with its litellm endpoint, pinned by
# runners/lcb/requirements.lock, in its own venv ($APARETO_HOME/venv-lcb), separate from the server's.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
. "$ROOT/scripts/local_env.sh"
VENV="$APARETO_HOME/venv-lcb"
[ -e "$VENV" ] && { echo "$VENV exists; remove it to reinstall"; exit 1; }
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -r "$HERE/requirements.lock"
"$VENV/bin/lighteval" --help >/dev/null
echo "lighteval installed in $VENV"
