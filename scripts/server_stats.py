#!/usr/bin/env python3
"""Speed and memory of one server run, from the server's own records.

    python3 scripts/server_stats.py ~/.apareto/results/<tag>   (the prefix of <tag>_server.log etc., under $APARETO_HOME/results)

Reads:
  <tag>_selection/selections.jsonl  per request: selection latency, midrun refreshes, think->answer swap, decode phases
  <tag>_selection/startup_memory.json  (or the server log's startup line): bank, KV, VRAM budget
  <tag>_memory.jsonl                 GPU memory sampled during the run
Prints one JSON object. Decode rates are token-weighted over all requests (tokens / seconds per phase).
"""
import json
from pathlib import Path
import re
import sys


def quantiles(values):
    values = sorted(values)
    if not values:
        return None
    pick = lambda q: values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]
    return {'n': len(values), 'p50': pick(0.5), 'p90': pick(0.9), 'max': values[-1]}


def startup_memory(prefix):
    path = Path(f'{prefix}_selection') / 'startup_memory.json'
    if path.exists():
        return json.loads(path.read_text())
    log = Path(f'{prefix}_server.log')
    if log.exists():
        for line in log.open(errors='replace'):
            match = re.search(r'APARETO startup memory: (\{.*\})', line)
            if match:
                return json.loads(match.group(1))
    return None


def summarize(prefix):
    prefix = str(prefix)
    rows = []
    selections = Path(f'{prefix}_selection') / 'selections.jsonl'
    if selections.exists():
        rows = [json.loads(line) for line in selections.open() if line.strip()]
    phases = {}
    for row in rows:
        for name, phase in (row.get('phases') or {}).items() if row.get('kind') == 'decode_phases' else ():
            total = phases.setdefault(name, {'tokens': 0, 'seconds': 0.0, 'requests': 0})
            total['tokens'] += phase.get('tokens', 0)
            total['seconds'] += phase.get('seconds', 0.0) - phase.get('paused_seconds', 0.0)
            total['requests'] += 1
    for total in phases.values():
        total['tokens_per_second'] = total['tokens'] / total['seconds'] if total['seconds'] else None
    selection = [r[k] for r in rows for k in ('selection_seconds', 'think_selection_seconds', 'answer_selection_seconds')
                 if isinstance(r.get(k), (int, float))]
    swaps = [r['swap_seconds'] for r in rows
             if r.get('kind') != 'midrun_swap' and isinstance(r.get('swap_seconds'), (int, float))]
    refreshes = [r['seconds'] for r in rows if r.get('kind') == 'midrun_swap' and isinstance(r.get('seconds'), (int, float))]
    per_request = {}
    for row in rows:
        for name, phase in (row.get('phases') or {}).items() if row.get('kind') == 'decode_phases' else ():
            if phase.get('tokens', 0) >= 64 and phase.get('tokens_per_second'):
                per_request.setdefault(name, []).append(phase['tokens_per_second'])
    peak = None
    memory = Path(f'{prefix}_memory.jsonl')
    if memory.exists():
        for line in memory.open():
            if line.strip():
                for device in json.loads(line).get('devices', {}).values():
                    peak = max(peak or 0, device.get('used_bytes', 0))
    start = startup_memory(prefix) or {}
    results = Path(f'{prefix}_results.json')
    host = (json.loads(results.read_text()).get('host_memory') or {}) if results.exists() else {}
    gib = lambda b: round(b / 2**30, 3) if isinstance(b, (int, float)) else None
    return {'decode': phases,
            'selection_seconds': quantiles(selection),
            'decode_tokens_per_second_per_request': {name: quantiles(v) for name, v in per_request.items()},
            'think_to_answer_swap_seconds': quantiles(swaps),
            'midrun_refresh_seconds': quantiles(refreshes),
            'memory_gib': {'gpu_peak_used': gib(peak), 'expert_bank': gib(start.get('bank_bytes')),
                           'bank_slots': start.get('bank_slots'), 'kv_cache': gib(start.get('W_kv_requested_bytes')),
                           'vram_budget': gib(start.get('V_bytes')),
                           'host_peak': gib(host.get('peak_scope_memory_bytes') or None),
                           'host_pinned_expert_tier': gib(host.get('pinned_tier_bytes'))}}


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(json.dumps(summarize(sys.argv[1]), indent=2))
