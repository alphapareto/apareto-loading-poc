#!/usr/bin/env python3
"""Summarize LiveCodeBench runs: lighteval's score per seed, their mean, and the server's speed and memory.

    cd ~/.apareto/results    # $APARETO_HOME/results
    python3 <repo>/runners/lcb/summarize.py lcb/lcb_s42_* lcb/lcb_s1234_* lcb/lcb_s4158_*

Each argument is a run directory written by runners/lcb/run.sh. The score is read from lighteval's own
results file; nothing is re-scored here.
"""
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from server_stats import summarize as server_summary
from apareto_loader import paths

TASK = 'lcb:codegeneration_v6|0'


def lighteval_score(run):
    files = sorted(run.rglob('results_*.json'))
    if not files:
        return None, None
    results = json.loads(files[-1].read_text())['results']
    metrics = results.get(TASK) or next((v for k, v in results.items() if k.startswith('lcb:')), {})
    name = next((k for k in metrics if 'pass@1' in k and not k.endswith('_stderr')), None)
    return name, metrics.get(name)


def main(runs):
    rows = []
    for run in map(Path, runs):
        name, score = lighteval_score(run)
        row = {'run': run.name, 'metric': name, 'score': score}
        # run.sh: one server (<tag>_server). run_multi.sh: one per GPU (<tag>_server_gpu<device>).
        several = sorted(paths.results().glob(f'{run.name}_server_gpu*_results.json'))
        if several:
            row['servers'] = {f.name[len(run.name) + 8:-len('_results.json')]:
                              server_summary(paths.results() / f.name[:-len('_results.json')]) for f in several}
        else:
            row['server'] = server_summary(paths.results() / f'{run.name}_server')
        rows.append(row)
    scores = [r['score'] for r in rows if isinstance(r['score'], (int, float))]
    out = {'task': TASK, 'runs': rows, 'n_seeds': len(scores),
           'mean': statistics.mean(scores) if scores else None,
           'stdev': statistics.stdev(scores) if len(scores) > 1 else None,
           'reference': {'source': 'RedHatAI/Qwen3.6-35B-A3B-NVFP4 model card, stock vLLM, 3 seeds', 'mean': 0.7467}}
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
