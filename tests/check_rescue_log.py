"""Gate: every rescue a served run made follows the rescue rule.

    python3 tests/check_rescue_log.py ~/.apareto/results/<tag>     (a server booted with --routing-log)

For each request's routing log (pre-mask router top-8 per decode step and layer, each pick's fate: 1 kept, 0 dropped,
2 dropped and rescued; the request's static think cut), the rule is recomputed with the server's cpu_rescue spec:
think steps score w x (1 inside the static cut, gout outside), answer steps w x (x / ax); per token row and layer the
highest-scoring dropped pick (first k on ties) is rescued when its score is above x (cap 1; fp32 products as the
kernel's). Steps within 2 of `</think>` are skipped (the phase switches per step).

PASS: in both phases at least 99.9% of token-layers agree with the rule exactly, and no think rescue lies outside the
static cut with w x gout <= x (an illegal rescue). The rest of the 0.1% are rows the GPU served after a CPU timeout,
which the log files as not rescued.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from apareto_loader.config import cpu_rescue_spec

prefix = sys.argv[1].rstrip('/')
record = json.load(open(prefix + '_results.json'))
spec = cpu_rescue_spec(record['expert']['cpu_rescue'])
if spec is None or spec['cap'] != 1:
    sys.exit('This check covers the cap-1 rescue rule')
X, AX, GOUT = (np.float32(spec[k]) for k in ('x', 'ax', 'gout'))
L = 40
agg = {p: {'token_layers': 0, 'agree': 0, 'rescues': 0, 'tokens': 0} for p in ('think', 'answer')}
illegal = legal_outside = requests = 0
files = sorted(glob.glob(prefix + '_selection/routing/*.npz'))
for path in files:
    z = np.load(path)
    requests += 1
    static = z['static_keep']
    p0, close, pos = int(z['prompt_len']), int(z['close_position']), z['positions']
    ids, w, state = z['ids'].astype(np.int64), z['weights'].astype(np.float32), z['state']
    lay = np.arange(L)[None, :, None]
    for phase in ('think', 'answer'):
        if phase == 'think':
            m = (pos >= p0) & ((pos < close - 2) if close > 0 else True)
        else:
            m = (pos > close + 2) if close > 0 else np.zeros(len(pos), bool)
        if not m.any():
            continue
        i_, w_, s_ = ids[m], w[m], state[m]
        drop = (s_ != 1) & (i_ >= 0)
        if phase == 'think':
            g = np.where(static[lay, np.clip(i_, 0, None)], np.float32(1), GOUT).astype(np.float32)
        else:
            g = np.full(i_.shape, np.float32(X / AX), np.float32)
        score = np.where(drop, (w_ * g).astype(np.float32), np.float32(-1))
        j = score.argmax(2)
        best = np.take_along_axis(score, j[..., None], 2)[..., 0]
        want = np.zeros(i_.shape, bool)
        np.put_along_axis(want, j[..., None], (best > X)[..., None], 2)
        got = s_ == 2
        a = agg[phase]
        a['token_layers'] += int(np.prod(got.shape[:2]))
        a['agree'] += int((want == got).all(2).sum())
        a['rescues'] += int(got.sum())
        a['tokens'] += int(got.shape[0])
        if phase == 'think':
            outside = got & ~static[lay, np.clip(i_, 0, None)]
            heavy = (w_ * GOUT).astype(np.float32) > X
            illegal += int((outside & ~heavy).sum())
            legal_outside += int((outside & heavy).sum())
ok = requests > 0 and illegal == 0
for phase, a in agg.items():
    a['agreement'] = round(a['agree'] / max(a['token_layers'], 1), 6)
    a['rescues_per_token'] = round(a['rescues'] / max(a['tokens'], 1), 3)
    ok &= a['token_layers'] > 0 and a['agreement'] >= 0.999
    print(f"  {phase}: {a['tokens']} tokens, agreement {a['agreement']}, rescues per token {a['rescues_per_token']}")
print(f'  requests {requests}; think rescues outside the static cut: {illegal} illegal, {legal_outside} by weight')
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
