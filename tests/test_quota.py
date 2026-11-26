"""Gate: per-layer quotas at every budget.

    python3 tests/test_quota.py [bundle]          (default: $APARETO_ARTIFACTS/code/think)

Checks, for a bundle with layer_quota.json:
  1 at every listed budget the reference returns the listed quota exactly
  2 at every budget 1..layers*experts the quota sums to the budget, every layer in 0..experts
  3 at the top budget every layer keeps every expert (the unmasked model)
  4 the runtime selector cuts at the reference's quota (one cut per 64 budgets, on the prior ranking)
Exit code 0 only if all pass.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'scripts'))
sys.path.insert(0, HERE)
from rstar_emit import ART, Selector, interpolate_quota
from apareto_loader.selector import RuntimeSelector

bundle = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ART, 'code', 'think')
fails = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


reference = Selector(bundle)
listed = json.load(open(os.path.join(bundle, 'layer_quota.json')))['quotas']
L, E = reference.L, reference.E
check('listed budgets', all(np.array_equal(reference.quota(int(b)), np.asarray(q)) for b, q in listed.items()),
      f'{sorted(int(b) for b in listed)}')
bad = []
for budget in range(1, L * E + 1):
    q = reference.quota(budget)
    if q.shape != (L,) or int(q.sum()) != budget or q.min() < 0 or q.max() > E:
        bad.append(budget)
check('every budget sums exactly, layers in range', not bad, f'{L * E} budgets, {len(bad)} bad {bad[:5]}')
check('top budget is unmasked', np.array_equal(reference.quota(L * E), np.full(L, E)))
quotas = {int(b): np.asarray(q) for b, q in listed.items()}
check('interpolation is the reference quota',
      all(np.array_equal(interpolate_quota(quotas, b, L, E), reference.quota(b))
          for b in range(1, L * E + 1, 97) if b not in quotas))
runtime = RuntimeSelector(bundle)
field = reference.prior.copy()
mismatch = [b for b in range(64, L * E + 1, 64)
            if not np.array_equal(runtime.cut(field, b).sum(axis=1), reference.quota(b))]
check('runtime cut follows the quota', not mismatch, f'{len(range(64, L * E + 1, 64))} budgets')
print('PASS' if not fails else f'FAIL: {fails}')
sys.exit(1 if fails else 0)
