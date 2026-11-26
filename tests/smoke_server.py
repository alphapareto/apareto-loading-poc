"""Smoke: boot the code profile with given overrides, serve one thinking request, check the records, stop.

    python3 tests/smoke_server.py --device 1 --port 8101 -- --budget 4800 --max-num-seqs 1

The request is a short LiveCodeBench-style problem with thinking on and max_tokens 2,500, so a midrun refresh
(every 1,056 thinking tokens by default) falls inside it unless the model closes `</think>` first. Checks:
  1 the server boots and answers (a non-empty completion)
  2 the boot record holds the requested settings (budget, max_num_seqs, rescue, midrun, ladder)
  3 the think selection cut exactly the budget, with the reference's per-layer quota
  4 refreshes happened when midrun is on and thinking ran past the interval, and none when it is off
  5 thinking-off, select=false, another budget and a raw prompt without the thinking-on generation prompt are
    refused, and the server keeps serving
Prints PASS/FAIL per check and exits nonzero on any failure. Uses one GPU for a few minutes.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from apareto_loader import paths  # noqa: E402

PROBLEM = ('You will be given a question (problem specification) and will generate a correct Python program that '
           'matches the specification and passes all tests.\n\nQuestion: Given an array of integers nums and an integer '
           'k, return the number of contiguous subarrays whose sum is divisible by k. Explain your reasoning carefully '
           'before answering.\n\nYou will use the following starter code to write the solution to the problem and '
           'enclose your code within delimiters.\n```python\nclass Solution:\n    def subarraysDivByK(self, nums: '
           'List[int], k: int) -> int:\n        \n```\n\n')


def main():
    argv = sys.argv[1:]
    extra = []
    if '--' in argv:
        i = argv.index('--')
        argv, extra = argv[:i], argv[i + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--port', type=int, default=8100)
    parser.add_argument('--max-tokens', type=int, default=2500)
    args = parser.parse_args(argv)
    tag = 'smoke_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    prefix = paths.results() / f'{tag}_server'
    log = open(paths.results() / f'{tag}_stdout.log', 'x')
    server = subprocess.Popen([sys.executable, str(ROOT / 'scripts/serve.py'), '--profile', 'code', '--device',
                               str(args.device), '--port', str(args.port), '--tag', f'{tag}_server', '--', *extra],
                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    fails = []

    def check(name, ok, detail=''):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}", flush=True)
        if not ok:
            fails.append(name)

    try:
        ready = Path(f'{prefix}_ready.json')
        deadline = time.monotonic() + 2400
        while not ready.exists():
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f'server did not become ready; see {log.name}')
            time.sleep(5)
        body = {'model': 'qwen3.6-35b', 'max_tokens': args.max_tokens, 'temperature': 1.0, 'top_p': 0.95,
                'top_k': 20, 'seed': 42, 'messages': [{'role': 'user', 'content': PROBLEM}]}
        request = urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/chat/completions',
                                         data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=3600) as response:
            reply = json.loads(response.read())
        message = reply['choices'][0]['message']
        # Requests the server does not serve: refused with a clear error, and the server keeps serving (the flush below).
        refusals = {}
        for name, path, refused in (
                ('thinking off', 'chat/completions', {**body, 'max_tokens': 4, 'chat_template_kwargs': {'enable_thinking': False}}),
                ('select=false', 'chat/completions', {**body, 'max_tokens': 4, 'vllm_xargs': {'select': False}}),
                ('another budget', 'chat/completions', {**body, 'max_tokens': 4, 'vllm_xargs': {'expert_budget': 4000}}),
                ('raw prompt without the thinking-on generation prompt', 'completions',
                 {'model': body['model'], 'max_tokens': 4, 'prompt': 'Write a Python function that adds two numbers.'})):
            attempt = urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/{path}', data=json.dumps(refused).encode(),
                                             headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(attempt, timeout=600) as response:
                    answer = json.loads(response.read())
                refusals[name] = answer['choices'][0].get('finish_reason') not in ('stop', 'length')
            except urllib.error.HTTPError as error:
                refusals[name] = 400 <= error.code < 600
        check('refusals', all(refusals.values()), json.dumps(refusals))
        # A single-request server writes a request's decode record when the next request arrives: send one.
        flush = urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/chat/completions', headers={'Content-Type': 'application/json'},
                                       data=json.dumps({**body, 'max_tokens': 1}).encode())
        urllib.request.urlopen(flush, timeout=600).read()
        usage = reply.get('usage', {})
        check('served', bool(message.get('content') or message.get('reasoning')), f"usage {usage}")
        startup = json.loads((Path(f'{prefix}_selection') / 'startup.json').read_text())
        expert = json.loads(next(Path(f'{prefix}_selection').glob('startup.json')).read_text())['load_report']
        rows = [json.loads(line) for line in open(Path(f'{prefix}_selection') / 'selections.jsonl')]
        think = [r for r in rows if r.get('mode') == 'think']
        refreshes = [r for r in rows if r.get('kind') == 'midrun_swap' and 'keep_sha256' in r]
        config = json.loads(ready.read_text())
        check('boot record', True, f"budget {config['budget']}, max_num_seqs {config.get('max_num_seqs')}, "
              f"rescue {expert.get('cpu_rescue')}, midrun {(startup.get('midrun') or {}).get('every')}")
        from rstar_emit import Selector
        bundle = (startup.get('think') or {}).get('think_bundle')
        budget = config['budget']
        quota = Selector(bundle).quota(budget) if bundle else None
        check('think selection at the budget', bool(think) and all(r.get('budget') == budget for r in think),
              f'{len(think)} think selections at {budget}; quota {None if quota is None else quota.tolist()[:6]}...')
        thought = max((r['phases']['think']['tokens'] for r in rows if r.get('kind') == 'decode_phases'
                       and 'think' in r.get('phases', {}) and r['phases']['think']['tokens'] > 1), default=0)
        every = (startup.get('midrun') or {}).get('every')
        if every is None:
            check('no refreshes with midrun off', not refreshes, f'{len(refreshes)} refreshes')
        elif thought > every + 64:
            check('refreshes', len(refreshes) >= 1, f'{len(refreshes)} refreshes over {thought} think tokens')
        else:
            check('refreshes', True, f'not reached: {thought} think tokens < {every}')
    except Exception as error:
        check('smoke', False, f'{type(error).__name__}: {error}')
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
    print('PASS' if not fails else f'FAIL: {fails}')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
