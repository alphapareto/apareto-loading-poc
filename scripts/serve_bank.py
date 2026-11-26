"""Persistent bank-engine server.

A benchmark harness drives the server over its OpenAI-compatible API for hours. This boots the bank engine and
waits, with the two things a long-lived server needs: the memory guards keep running for the whole session, and a
ready-file tells the harness where to connect.

The serving setup: the ExpertConfig, the project options that install the bank loader, worker, scheduler and
middleware, the per-run paths (run_paths.py), the compilation config, and --no-enable-prefix-caching, which is not a
tuning choice -- the worker refuses to begin serving a request from a cached prefix.

Verification runs swap in an audited worker and scheduler from outside this repository (--worker-cls,
--scheduler-cls) and set --audit, which gives them an audit path and the router's picks; production servers do
neither.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

from server_util import MemorySampler, await_server, check_stock_manifest, require_free_port, stop_owned_process, write_json
from host_memory import HostMemoryGuard
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apareto_loader import paths as home_paths
from apareto_loader.config import ExpertConfig
from apareto_loader.vllm027.run_paths import run_env

# Batch-invariant Inductor kernels: a prompt's prefill features are the same on every boot.
COMPILATION_CONFIG = {'cudagraph_mode': 'FULL_AND_PIECEWISE', 'inductor_compile_config': {'batch_invariant': True}}
WORKER = 'apareto_loader.vllm027.bank_worker.BankWorker'
SCHEDULER = 'apareto_loader.vllm027.scheduler.BankScheduler'
ASYNC_SCHEDULER = 'apareto_loader.vllm027.scheduler.AsyncBankScheduler'
MIDDLEWARE = 'apareto_loader.vllm027.middleware.BankControlMiddleware'


def project_options(expert, worker, scheduler, async_scheduling):
    """vLLM options that install the bank engine. The per-run paths go in the environment (run_paths.py says why)."""
    return ['--load-format', 'apareto_bank', '--hf-overrides', json.dumps({'architectures': ['AparetoBankQwen3_5Moe']}),
            '--additional-config', json.dumps({'apareto': asdict(expert), 'apareto_run_paths_env': True}),
            '--worker-cls', worker, '--scheduler-cls', scheduler, '--middleware', MIDDLEWARE,
            '--async-scheduling' if async_scheduling else '--no-async-scheduling']


def warm_up(url, model, chunk, audit_path):
    """One request before ready.json: a chat prompt longer than one full prefill chunk, so
    the largest chunk runs, and eight decode steps. The startup profile no longer caps KV, so an activation overrun
    has to show here, at boot, not on a client's first long turn. It goes through the chat template with thinking
    on, as chat clients do. Under --audit its audit row must be clean; a production server (audit_path None) writes
    none."""
    body = {'model': model, 'max_tokens': 8, 'temperature': 0.0, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': True},
            'messages': [{'role': 'user', 'content': 'Warm-up. ' + ' '.join(['apple'] * (chunk + 64))}]}
    request = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=1800) as response:
        usage = json.loads(response.read().decode()).get('usage') or {}
    seconds = time.monotonic() - started
    rows = [json.loads(ln) for ln in audit_path.read_text().splitlines() if ln.strip()] \
        if audit_path is not None and audit_path.exists() else []
    if int(usage.get('prompt_tokens') or 0) <= chunk or int(usage.get('completion_tokens') or 0) < 1:
        raise RuntimeError(f'warm-up served {usage}: not more than one {chunk}-token chunk and a decode')
    if audit_path is not None and (not rows or rows[-1].get('error') or rows[-1].get('clean') is False or not rows[-1].get('phases')):
        raise RuntimeError(f'warm-up audit is not clean: {json.dumps(rows[-1:])[:400]}')
    return {'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': usage.get('completion_tokens'),
            'seconds': seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--expert-store', required=True)
    parser.add_argument('--layer-store', required=True, type=Path)
    parser.add_argument('--think-bundle', type=Path, required=True, help='selects the static think cut at prefill; its shallow taps also predict early (speed only)')
    parser.add_argument('--answer-bundle', type=Path, required=True,
                        help='selects the answer mask from a probe at </think>')
    parser.add_argument('--midrun-every', type=int, default=None,
                        help='refresh the think mask every N think tokens from the midrun bundle; 0 or unset: off')
    parser.add_argument('--midrun-bundle', type=Path, default=None)
    parser.add_argument('--midrun-weight', type=float, default=0.5,
                        help='think mask = cut((1 - w) * static field + w * midrun field)')
    parser.add_argument('--midrun-at-prefill', action=argparse.BooleanOptionalAction, default=False,
                        help='the first think block is a midrun selection too, from the prefill read')
    parser.add_argument('--ladder', default=None,
                        help="think-end ladder spec (config/ladder.json) attached to every thinking request; 'off': none")
    parser.add_argument('--routing-log', action='store_true',
                        help='verification only: log every decode step\'s router picks and their fate for '
                             'tests/check_rescue_log.py (slower; not for benchmarks)')
    parser.add_argument('--max-num-seqs', type=int, default=1,
                        help='requests decoding at once (concurrent decode needs paged decode, --gpu-experts)')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--served-model-name', default='qwen3.6-35b', help='what the harness calls the model')
    parser.add_argument('--budget', type=int, default=4301)
    parser.add_argument('--max-model-len', type=int, required=True)
    parser.add_argument('--max-num-batched-tokens', type=int, default=2048)
    parser.add_argument('--gpu-margin-bytes', type=int, default=256 << 20)
    parser.add_argument('--activation-reserve-bytes', type=int, default=1536 << 20,
                        help='load-time activation estimate; the default over-reserves by ~0.57 GB '
                             'against the measured prefill working set, and that is KV capacity')
    parser.add_argument('--host-tier-gib', type=float, default=14.0)
    parser.add_argument('--kv-cache-memory-bytes', type=int, required=True)
    parser.add_argument('--gpu-experts', type=int, default=None,
                        help='paged decode: this many of the budget in VRAM')
    parser.add_argument('--cpu-experts', default=None,
                        help="paged decode's absent experts computed by the CPU: "
                             "'cores=4-7,timeout_us=1000,admit=on'")
    parser.add_argument('--cpu-rescue', default=None,
                        help="the rescue rule: 'x=0.00014,ax=0.12,gout=0.001,cap=1,phases=think+answer', or 'off'")
    parser.add_argument('--async-scheduling', action='store_true')
    parser.add_argument('--boot-seconds', type=int, default=2400)
    parser.add_argument('--enforce-eager', action='store_true',
                        help='Disable inductor compilation. Piecewise compilation inlines RMSNorm into '
                             'a compiled region where module forward hooks do not fire, which is why the '
                             'hl taps capture nothing while kvl does (its qkv_proj sits next to a '
                             'splitting op and runs eagerly).')
    parser.add_argument('--tool-call-parser', default=None,
                        help='a vLLM tool-call parser (e.g. qwen3_coder), with auto tool choice; default: none, as in '
                             "Red Hat's evaluation serving")
    parser.add_argument('--reasoning-parser', default='qwen3')
    parser.add_argument('--audit', action='store_true',
                        help='verification only: give the worker and scheduler an audit path and the router\'s picks '
                             '(--enable-return-routed-experts); for an audited worker and scheduler set below')
    parser.add_argument('--worker-cls', default=WORKER, help='verification only: an audited worker class')
    parser.add_argument('--scheduler-cls', default=None, help='verification only: an audited scheduler class')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--host', default='127.0.0.1', help='0.0.0.0 to reach containers at 172.17.0.1')
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or list(home_paths.results().glob(args.tag + '_*')):
        parser.error('use a fresh filename-safe tag')
    if os.environ.get('TMPDIR') != str(home_paths.cache() / 'tmp'):
        parser.error('source scripts/local_env.sh')
    if args.midrun_every == 0:
        args.midrun_every = None
    if args.midrun_every is None:
        args.midrun_bundle, args.midrun_at_prefill = None, False
    if args.ladder == 'off':
        args.ladder = None
    if args.cpu_rescue == 'off':
        args.cpu_rescue = None
    # cores=auto:N: the CPU expert threads get the last N cores of this process's affinity to themselves; vLLM runs
    # on the others (a shared core preempts the CPU server, which the GPU then sees as a timeout).
    taskset, cpu_experts = [], args.cpu_experts
    auto = re.search(r'cores=auto:(\d+)', cpu_experts or '')
    if auto:
        mine = sorted(os.sched_getaffinity(0))
        n = int(auto.group(1))
        if not 0 < n < len(mine):
            parser.error(f'cores=auto:{n} leaves no core for vLLM out of {len(mine)}')
        cpu_experts = cpu_experts.replace(auto.group(0), 'cores=' + ','.join(map(str, mine[-n:])).replace(',', ';'))
        taskset = ['taskset', '-c', ','.join(map(str, mine[:-n]))]

    base = home_paths.results() / args.tag
    paths = {k: Path(f'{base}_{k}') for k in ('server.log', 'memory.jsonl', 'audit.jsonl', 'patch.jsonl',
                                              'selection', 'ready.json', 'results.json')}
    expert = ExpertConfig(str(Path(args.expert_store).resolve()), args.budget, str(args.layer_store.resolve()),
                          think_bundle=str(args.think_bundle.resolve()),
                          answer_bundle=str(args.answer_bundle.resolve()),
                          host_tier_bytes=int(args.host_tier_gib * (1 << 30)), gpu_margin_bytes=args.gpu_margin_bytes,
                          activation_reserve_bytes=args.activation_reserve_bytes,
                          gpu_experts=args.gpu_experts, cpu_experts=cpu_experts, cpu_rescue=args.cpu_rescue,
                          midrun_every=args.midrun_every, midrun_weight=args.midrun_weight,
                          midrun_at_prefill=args.midrun_at_prefill, routing_log=args.routing_log,
                          ladder=None if args.ladder is None else str(Path(args.ladder).resolve()),
                          midrun_bundle=None if args.midrun_bundle is None else str(args.midrun_bundle.resolve()))
    scheduler = args.scheduler_cls or (ASYNC_SCHEDULER if args.async_scheduling else SCHEDULER)
    command = [*taskset, str(home_paths.venv() / 'bin/vllm'), 'serve', str(Path(args.model).resolve()),
        '--served-model-name', args.served_model_name, '--host', args.host, '--port', str(args.port),
        '--tensor-parallel-size', '1', '--dtype', 'bfloat16', '--language-model-only',
        '--kv-cache-memory-bytes', str(args.kv_cache_memory_bytes),
        '--max-model-len', str(args.max_model_len), '--max-num-seqs', str(args.max_num_seqs),
        '--max-num-batched-tokens', str(args.max_num_batched_tokens), '--no-enable-prefix-caching',
        '--no-enable-flashinfer-autotune', '--moe-backend', 'flashinfer_cutlass',
        *(['--enforce-eager'] if args.enforce_eager else
          ['--compilation-config', json.dumps(COMPILATION_CONFIG)]),
        '--reasoning-parser', args.reasoning_parser,
        *(['--enable-auto-tool-choice', '--tool-call-parser', args.tool_call_parser] if args.tool_call_parser else []),
        *(['--logits-processors', 'apareto_loader.vllm027.ladder:LadderLogitsProcessor'] if args.ladder else []),
        *project_options(expert, args.worker_cls, scheduler, args.async_scheduling),
        *(['--enable-return-routed-experts'] if args.audit else []),
        '--enable-per-request-metrics']
    record = {'status': 'failed', 'kind': 'persistent bank-engine server',
              'claim_type': 'measured', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'command': command, 'expert': vars(expert),
              'max_model_len': args.max_model_len, 'max_num_batched_tokens': args.max_num_batched_tokens,
              'kv_cache_memory_bytes': args.kv_cache_memory_bytes, 'async_scheduling': args.async_scheduling,
              'audit': args.audit, 'worker_cls': args.worker_cls, 'scheduler_cls': scheduler,
              'versions': {n: version(n) for n in ('vllm', 'torch', 'flashinfer-python')},
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for folder in ('apareto_loader', 'scripts')
                                for p in (ROOT / folder).rglob('*.py')},
              'gates': {}}
    sampler = server = host = None
    url = f'http://127.0.0.1:{args.port}'
    stopping = {'flag': False}
    def on_signal(signum, frame):
        stopping['flag'] = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_signal)
    try:
        check_stock_manifest(home_paths.stock_manifest())
        host = HostMemoryGuard(f'apareto-{args.tag}', expert.host_tier_bytes)
        sampler = MemorySampler([args.device], paths['memory.jsonl'], use_v2=True)
        sampler.require_idle()
        require_free_port(args.port)
        budgets = {args.device: sampler.initial[args.device]['addressable_bytes'] - args.gpu_margin_bytes}
        sampler.start()
        def guard():
            host.check()
            sampler.require_within_budget(budgets)
            if server is not None and server.poll() is not None:
                raise RuntimeError(f'Server exited: {server.returncode}')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.device), 'VLLM_USE_V2_MODEL_RUNNER': '0',
               **({'OMP_NUM_THREADS': str(len(taskset[2].split(',')))} if taskset else {}),
               'VLLM_SERVER_DEV_MODE': '1', 'VLLM_ENABLE_STARTUP_PLAN': '0', 'PYTHONPATH': str(ROOT),
               'PATH': f"{home_paths.venv() / 'bin'}:{os.environ['PATH']}", 'TORCHINDUCTOR_COMPILE_THREADS': '1'}
        with paths['server.log'].open('x') as log:
            server = subprocess.Popen(host.wrap(command), cwd=ROOT,
                                      env={**env, **run_env(paths['patch.jsonl'], paths['selection'],
                                                            paths['patch.jsonl'].with_suffix('.prefill.jsonl'),
                                                            paths['audit.jsonl'] if args.audit else None)},
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print(f'bank server pid={server.pid}; log={paths["server.log"]}', flush=True)
            await_server(server, url, args.served_model_name, args.boot_seconds, guard)
            record['warmup'] = warm_up(url, args.served_model_name, args.max_num_batched_tokens,
                                       paths['audit.jsonl'] if args.audit else None)
            guard()
            print('WARMUP OK ' + json.dumps(record['warmup']), flush=True)
            ready = {'url': url, 'host': args.host, 'port': args.port, 'pid': server.pid,
                     'served_model_name': args.served_model_name, 'tag': args.tag,
                     'max_model_len': args.max_model_len, 'budget': args.budget, 'max_num_seqs': args.max_num_seqs,
                     'ready_at_utc': datetime.now(timezone.utc).isoformat()}
            write_json(paths['ready.json'], ready)
            print('SERVER READY ' + json.dumps(ready), flush=True)
            while not stopping['flag']:
                guard()
                time.sleep(5)
            print('stop requested; draining', flush=True)
        record['status'] = 'pass'
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
        print('[FAIL] ' + record['error'], flush=True)
    finally:
        # Each teardown step is isolated: a crash in one would otherwise abort the whole block,
        # so results.json would never be written and the real error would be lost. Whatever
        # happens here, the record must land.
        for name, step in (('stop_server', lambda: stop_owned_process(server) if server is not None else None),
                           ('close_sampler', lambda: record.__setitem__('memory', sampler.close())
                            if sampler is not None else None),
                           ('release_host', lambda: host.stop() if host is not None else None)):
            try:
                step()
            except Exception as teardown_error:
                record.setdefault('teardown_errors', {})[name] = f'{type(teardown_error).__name__}: {teardown_error}'
        if host is not None:
            # Host RAM of the server's memory scope: the peak seen by the guard over the whole run.
            record['host_memory'] = host.record
        record['stopped_at_utc'] = datetime.now(timezone.utc).isoformat()
        paths['ready.json'].unlink(missing_ok=True)
        write_json(paths['results.json'], record)
    print(json.dumps({'status': record['status'], 'error': record.get('error')}), flush=True)
    return record['status'] != 'pass'


if __name__ == '__main__':
    sys.exit(main())
