"""Prompt-position bookkeeping for the shipped final-eight-row feature contract."""
from dataclasses import dataclass

import numpy as np

TAPS = tuple([f"hl{i}" for i in (2, 4, 8, 14, 20, 27)] +
             [f"kvl{i}" for i in (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)])
DIMS = (2048,) * 6 + (1024,) * 10
ROWS = 8
FEATURES = sum(DIMS) * ROWS


@dataclass
class PromptWindow:
    request_id: str
    prompt_tokens: int
    next_position: int = 0

    def __post_init__(self):
        if not self.request_id or self.prompt_tokens < ROWS:
            raise ValueError("Selected prompts shorter than eight tokens are unsupported")

    def indices(self, start, count):
        if start != self.next_position or count < 1 or start + count > self.prompt_tokens:
            raise ValueError("Missing, repeated or mixed-phase prompt chunk")
        positions = np.arange(self.prompt_tokens - ROWS, self.prompt_tokens)
        local = positions - start
        indices = np.where((local >= 0) & (local < count), local, -1).astype(np.int64)
        self.next_position = start + count
        return indices

    @property
    def complete(self):
        return self.next_position == self.prompt_tokens


def assemble_taps(captures):
    if set(captures) != set(TAPS):
        raise ValueError("Missing or unexpected activation taps")
    arrays = []
    for tap, width in zip(TAPS, DIMS):
        value = np.asarray(captures[tap])
        if value.shape != (ROWS, width) or value.dtype != np.float16 or not np.isfinite(value).all():
            raise ValueError(f"Invalid final prompt rows for {tap}")
        arrays.append(value.reshape(-1))
    return np.concatenate(arrays)
