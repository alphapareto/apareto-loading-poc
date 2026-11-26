#!/usr/bin/env python3
"""Start the apareto server for one serving profile.

    python3 scripts/serve.py --profile code [--device 0] [--port 8000] [-- --budget 4608 ...]

The client-facing surface is the one Red Hat's evaluation serving of RedHatAI/Qwen3.6-35B-A3B-NVFP4 exposes
(`vllm serve ... --reasoning-parser qwen3 --language-model-only`): an OpenAI-compatible /v1 with chat completions,
the qwen3 reasoning parser, and here the served name `qwen3.6-35b`. Benchmark harnesses therefore run unmodified
against it. Only thinking requests are served (LiveCodeBench runs with thinking on); others are refused.

A profile (config/profiles.json) fixes the server options and which R* bundle fills each phase slot. Paths:

    --model      APARETO_MODEL      default models/Qwen3.6-35B-A3B-NVFP4
    --stores     APARETO_STORES     default stores/   (expert_store_header.json, layer_store/experts.apkl)
    --artifacts  APARETO_ARTIFACTS  default artifacts/

The server writes $APARETO_HOME/results/<tag>_* (server log, memory samples, selections, ready.json, results.json) and runs
until SIGINT/SIGTERM. `ready.json` appears once the server has answered a warm-up request.
Any option after `--` is passed to scripts/serve_bank.py unchanged and overrides the profile's value.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / 'config' / 'profiles.json'
sys.path.insert(0, str(ROOT))
from apareto_loader import paths  # noqa: E402  (standard library only)


def in_local_env():
    return os.environ.get('TMPDIR') == str(paths.cache() / 'tmp')


def reexec_in_local_env():
    """The server keeps every cache under APARETO_HOME (scripts/local_env.sh); start again under it."""
    script = f'. {ROOT}/scripts/local_env.sh && exec {paths.venv()}/bin/python {Path(__file__).resolve()} "$@"'
    os.execvp('bash', ['bash', '-c', script, 'serve.py', *sys.argv[1:]])


def option(name, value):
    flag = '--' + name.replace('_', '-')
    if value is True:
        return [flag]
    if value is False or value is None:
        return []
    return [flag, str(value)]


def main():
    profiles = {k: v for k, v in json.loads(PROFILES.read_text()).items() if not k.startswith('_')}
    argv, passthrough = (sys.argv[1:], [])
    if '--' in argv:
        i = argv.index('--')
        argv, passthrough = argv[:i], argv[i + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', required=True, choices=sorted(profiles))
    parser.add_argument('--model', type=Path, default=paths.model())
    parser.add_argument('--stores', type=Path, default=paths.stores())
    parser.add_argument('--artifacts', type=Path, default=Path(os.environ.get('APARETO_ARTIFACTS', ROOT / 'artifacts')))
    parser.add_argument('--device', type=int, default=int(os.environ.get('APARETO_DEVICE', 0)))
    parser.add_argument('--port', type=int, default=int(os.environ.get('APARETO_PORT', 8000)))
    parser.add_argument('--host', default=None, help="bind address (profile default: 127.0.0.1)")
    parser.add_argument('--served-model-name', default='qwen3.6-35b')
    parser.add_argument('--tool-call-parser', default=None,
                        help='a vLLM tool-call parser with auto tool choice (e.g. qwen3_coder); default: none')
    parser.add_argument('--reasoning-parser', default='qwen3')
    parser.add_argument('--tag', default=None, help='$APARETO_HOME/results/<tag>_*; default <profile>_<UTC timestamp>')
    parser.add_argument('--dry-run', action='store_true', help='print the resolved server command line and exit')
    args = parser.parse_args(argv)
    if not args.dry_run and not in_local_env():
        reexec_in_local_env()

    profile = profiles[args.profile]
    tag = args.tag or f"{args.profile}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    stores = args.stores.resolve()
    server = dict(profile['server'])
    if args.host is not None:
        server['host'] = args.host
    cores = os.environ.get('APARETO_CPU_EXPERT_CORES')   # explicit cores for the CPU expert threads, e.g. 6;7
    if cores and server.get('cpu_experts'):
        items = [i for i in server['cpu_experts'].split(',') if not i.startswith('cores=')]
        server['cpu_experts'] = ','.join([f'cores={cores}', *items])

    command = ['--model', str(args.model.resolve()),
               '--expert-store', str(stores / 'expert_store_header.json'),
               '--layer-store', str(stores / 'layer_store' / 'experts.apkl'),
               '--tag', tag, '--device', str(args.device), '--port', str(args.port),
               '--served-model-name', args.served_model_name, '--reasoning-parser', args.reasoning_parser,
               *(['--tool-call-parser', args.tool_call_parser] if args.tool_call_parser else [])]
    for slot, relative in profile['bundles'].items():
        command += option(slot, (args.artifacts / relative).resolve())
    if server.get('ladder') not in (None, 'off'):
        server['ladder'] = str((ROOT / server['ladder']).resolve())   # repository-relative in the profile
    for name, value in server.items():
        command += option(name, value)
    command += passthrough

    missing = [p for p in (args.model, stores / 'expert_store_header.json', stores / 'layer_store' / 'experts.apkl',
                           *(args.artifacts / r for r in profile['bundles'].values())) if not Path(p).exists()]
    if args.dry_run:
        print(' '.join(['scripts/serve_bank.py', *command]))
        for path in missing:
            print(f'missing: {path}', file=sys.stderr)
        return 0
    if missing:
        sys.exit('Missing inputs (see docs/INSTALL.md):\n  ' + '\n  '.join(map(str, missing)))
    paths.results().mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT / 'scripts'))
    import serve_bank
    sys.argv = ['serve_bank.py', *command]
    return serve_bank.main()


if __name__ == '__main__':
    sys.exit(main())
