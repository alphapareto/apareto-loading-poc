"""Thinking mode: phase plan, phase masks and the think-mode read contract.

A thinking request is served in three phases:
- the prompt, prefilled dense;
- think decode on the think mask;
- answer decode on the answer mask.
The switch is at `</think>`. The row whose input is `</think>` (position q) still runs under the think
mask. The worker then loads only the answer experts not yet resident and decodes q+1 onward under the
answer mask.

Each mask is either static or selected:
- static: the same mask for every request (a `disabled_*.json` file: per layer, the disabled expert ids);
- selected: the think network reads the 8 rollback-spacer rows appended to the thinking-on prompt, so the
  think mask is known at the end of prefill. The same network, on its shallow taps, predicts the think mask at
  layer 4's entry from the same rows.
- post-think answer (answer_read='post_think'): the answer R* reads 8 spacer rows appended after
  `</think>` instead. Its bundle says so in `read_variant`; a bundle without the key is a prompt read
  ('think_open'), and a bundle made for one read point is refused for the other.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

THINK_END = 248069                 # '</think>'


def load_phase_mask(path, layers=40, experts=256):
    """A keep [layers, experts] bool from a `disabled_*.json` file (per layer, the disabled expert ids)."""
    path = Path(path)
    disabled = json.loads(path.read_text())
    if not isinstance(disabled, list) or len(disabled) != layers:
        raise ValueError(f'{path.name}: expected {layers} per-layer disabled lists')
    keep = np.ones((layers, experts), dtype=bool)
    for layer, ids in enumerate(disabled):
        if any(type(e) is not int or not 0 <= e < experts for e in ids) or len(set(ids)) != len(ids):
            raise ValueError(f'{path.name}: invalid expert ids at layer {layer}')
        keep[layer, ids] = False
    return keep, hashlib.sha256(path.read_bytes()).hexdigest()


def think_prompt_tail(tokenizer):
    """Token IDs after the user content of a thinking-on generation prompt (`<|im_end|>\\n...<think>\\n`)."""
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': 'x'}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True)
    ids = tokenizer(rendered, add_special_tokens=False).input_ids
    end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    return tuple(ids[len(ids) - 1 - ids[::-1].index(end):])


READ_VARIANTS = ('think_open', 'post_think')


def think_read_contract(bundle, phase, read_variant='think_open'):
    """Spacer token IDs of a think-mode bundle read on 8 spacer tokens appended at token-ID level to a
    thinking-on render, dense prefill. `read_variant` is the read the caller serves: 'think_open' (the prefill
    spacer, every bundle without the key) or 'post_think' (after `</think>`, answer phase only). Refuses a
    bundle made for the other read, and a think_open bundle whose prose places the read after `</think>`."""
    path = Path(bundle) / 'read_contract.json'
    contract = json.loads(path.read_text())
    if read_variant not in READ_VARIANTS:
        raise ValueError(f'Unknown read variant {read_variant!r}')
    have = contract.get('read_variant', 'think_open')
    if have != read_variant:
        raise ValueError(f'{path}: bundle read_variant {have!r}, served as {read_variant!r}; a bundle made for '
                         'one read point cannot serve the other')
    spacer = contract.get('spacer_token_ids')
    render, placement, rows = (str(contract.get(k, '')) for k in ('render', 'placement', 'rows_read'))
    if read_variant == 'think_open':
        read = '</think>' not in placement and 'after the think' not in rows.lower()
    else:
        # The probe rolls back mid-decode, so the contract must say so; think_open bundles may omit the key.
        read = (phase == 'answer' and 'POST-THINK' in str(contract.get('read_point', ''))
                and contract.get('decode') == 'rolled back to COMMIT')
    if (not isinstance(spacer, list) or len(spacer) != 8 or any(type(t) is not int or t < 0 for t in spacer)
            or 'enable_thinking=true' not in render or not placement.startswith('appended after') or not read
            or not str(contract.get('prefill', '')).startswith('dense')
            or contract.get('phase') != phase):
        raise ValueError(f'{path}: not a {phase} bundle read on the {read_variant} spacer rows')
    return tuple(spacer)


READ_ROWS = ('spacer', 'raw')


def think_read_rows(bundle, contract='read_contract.json'):
    """Which 8 rows a think-mode bundle reads: 'spacer' (the appended spacer rows, every bundle without the key) or
    'raw' (the last 8 prompt rows, [P-8, P-1]). The spacer forward runs either way; under causal attention it leaves
    the raw rows unchanged."""
    path = Path(bundle) / contract
    rows = json.loads(path.read_text()).get('read_rows', 'spacer')
    if rows not in READ_ROWS:
        raise ValueError(f'{path}: unknown read_rows {rows!r}')
    return rows


@dataclass(frozen=True)
class ThinkPlan:
    tail: tuple
    think: np.ndarray | None = None          # static masks (None: selected by the bundle)
    answer: np.ndarray | None = None
    think_sha256: str | None = None
    answer_sha256: str | None = None

    @classmethod
    def load(cls, tokenizer, think_keep=None, answer_keep=None):
        if tokenizer.decode([THINK_END]) != '</think>':
            raise ValueError(f'Token {THINK_END} is not </think> in this tokenizer')
        think, think_sha = load_phase_mask(think_keep) if think_keep else (None, None)
        answer, answer_sha = load_phase_mask(answer_keep) if answer_keep else (None, None)
        return cls(think_prompt_tail(tokenizer), think, answer, think_sha, answer_sha)

    def record(self):
        return {'think_keep_sha256': self.think_sha256, 'answer_keep_sha256': self.answer_sha256,
                'think_prompt_tail': list(self.tail), 'think_end': THINK_END}
