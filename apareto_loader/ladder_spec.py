"""Build the per-request ladder config from a JSON spec file and the tokenizer.

Token sets: line-end ids
(text ending in a newline), fence ids (text with a 3+ backtick run), blank ids (spaces/tabs only), the injections
encoded without special tokens. The engine needs no tokenizer; the config travels as a JSON string in
vllm_xargs["ladder"] (vllm_xargs takes only scalars and flat lists).
"""
import json
import re

THINK_END_TEXT, THINK_END_ID = '</think>', 248069


def token_sets(tok):
    """(ends_nl_ids, fence_ids, ws_ids) over the whole vocabulary."""
    ends, fences, blanks = [], [], []
    size = len(tok)
    for start in range(0, size, 4096):
        chunk = list(range(start, min(size, start + 4096)))
        for i, text in zip(chunk, tok.batch_decode([[c] for c in chunk])):
            if text.endswith('\n'):
                ends.append(i)
            if re.search(r'`{3,}', text or ''):
                fences.append(i)
            if text and text.strip(' \t') == '' and '\n' not in text:
                blanks.append(i)
    return ends, fences, blanks


def build_config(spec, tok, cap):
    """The engine config (a dict) for a generation cap of `cap` tokens; refuses an inconsistent spec."""
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    nl = enc('\n')
    if len(nl) != 1:
        raise ValueError(f'newline is not one token: {nl}')
    if tok.convert_ids_to_tokens(THINK_END_ID) != THINK_END_TEXT:
        raise ValueError(f'token {THINK_END_ID} is not {THINK_END_TEXT}')
    nudges = [text for _, _, text in spec['gates']] + [spec['ramp']['injection']]
    if any(THINK_END_TEXT in text for text in nudges):
        raise ValueError('only the final injection may carry </think>')
    if THINK_END_TEXT not in spec['final']['injection']:
        raise ValueError('the final injection must carry </think>')
    reserve = cap - spec['final']['at_tok'] - spec['final_wait']
    if reserve < spec['min_answer_reserve']:
        raise ValueError(f'answer reserve {reserve} < {spec["min_answer_reserve"]} at cap {cap}')
    ends, fences, blanks = token_sets(tok)
    if nl[0] not in ends or not fences:
        raise ValueError('line-end or fence token sets are wrong for this tokenizer')
    ramp = spec['ramp']
    return {'fixed': [[at, bar, enc(text)] for at, bar, text in spec['gates']],
            'ramp': {k: ramp[k] for k in ('start', 'thr0', 'decay', 'per', 'floor', 'max_fires')} | {'inj_ids': enc(ramp['injection'])},
            'final': {'at_tok': spec['final']['at_tok'], 'inj_ids': enc(spec['final']['injection'])},
            'tag_id': THINK_END_ID, 'wait_cap': spec['wait_cap'], 'final_wait': spec['final_wait'],
            'repeats': spec['repeats'], 'maxn': spec['maxn'],
            'ends_nl_ids': ends, 'nl_id': nl[0], 'fence_ids': fences, 'fence_close_ids': enc('```\n'), 'ws_ids': blanks,
            'safe_final': spec['safe_final'], 'safe_nudge': spec['safe_nudge'], 'nudge_wait': spec['nudge_wait'],
            'fence0': 0}


def wire(config):
    return json.dumps(config, separators=(',', ':'))
