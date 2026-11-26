"""Explicit project worker-to-scheduler selection metadata; no wheel modification."""
from dataclasses import dataclass, field, fields

import numpy as np
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

from apareto_loader.selector import array_digest


@dataclass
class SelectedModelRunnerOutput(ModelRunnerOutput):
    apareto_updates: list[dict] = field(default_factory=list)
    apareto_phases_applied: bool = False


class DeferredOutput(AsyncModelRunnerOutput):
    """Async scheduling: the step's sampled tokens reach the host later; `finish` runs on them when the engine
    collects the output (before it schedules the step after next)."""
    def __init__(self, inner, finish):
        self.inner, self.finish = inner, finish

    def get_output(self):
        return self.finish(self.inner.get_output())


def attach_updates(output, updates):
    if not isinstance(output, ModelRunnerOutput):
        raise TypeError('Selected serving requires synchronous native model outputs')
    return SelectedModelRunnerOutput(**{f.name:getattr(output,f.name) for f in fields(ModelRunnerOutput)},
                                     apareto_updates=updates)


def selection_update(selection):
    return {**selection.record(), 'kind':'selection',
            'packed_keep':np.packbits(selection.keep.reshape(-1)).tobytes()}


def decode_update(update, request_id, prompt_tokens, layers, experts):
    if update['kind'] != 'selection' or update['request_id'] != request_id:
        raise ValueError('Selection update belongs to another request')
    if array_digest(np.asarray(prompt_tokens,dtype='<i8')) != update['prompt_sha256']:
        raise ValueError('Selection update belongs to another prompt')
    bits = np.frombuffer(update['packed_keep'],dtype=np.uint8)
    if len(bits) != (layers*experts+7)//8:
        raise ValueError('Malformed selection update mask')
    keep = np.unpackbits(bits)[:layers*experts].astype(bool).reshape(layers,experts)
    budget = update['budget']
    quota = np.array([budget//layers+(layer<budget%layers) for layer in range(layers)])
    if (type(budget) is not int or not 1 <= budget <= layers*experts or
            int(keep.sum()) != budget or not np.array_equal(keep.sum(1),quota) or
            array_digest(keep) != update['keep_sha256']):
        raise ValueError('Selection update digest, budget or quotas differ')
    return keep


def phase_update(request_id, phase, first_position, keep):
    """Thinking mode: the mask that serves `phase` from `first_position` on."""
    return {'kind': 'phase', 'request_id': request_id, 'phase': phase, 'first_position': int(first_position),
            'keep_sha256': array_digest(keep), 'packed_keep': np.packbits(keep.reshape(-1)).tobytes()}


def decode_phase(update, request_id, previous, layers, experts, kinds=None):
    """(first position, keep) of a phase update; phases arrive think, then (midrun) any
    further think segments, then answer, at increasing positions. `kinds`: the phases of `previous`, in order."""
    kinds = ['think', 'answer'][:len(previous)] if kinds is None else kinds
    allowed = (update['phase'] == 'think' and 'answer' not in kinds) or \
              (update['phase'] == 'answer' and kinds and kinds[-1] == 'think')
    if update['request_id'] != request_id or len(kinds) != len(previous) or not allowed:
        raise ValueError('Phase update out of order or for another request')
    first = update['first_position']
    if type(first) is not int or (previous and first <= previous[-1][0]):
        raise ValueError('Phase update positions must increase')
    bits = np.frombuffer(update['packed_keep'], dtype=np.uint8)
    if len(bits) != (layers * experts + 7) // 8:
        raise ValueError('Malformed phase update mask')
    keep = np.unpackbits(bits)[:layers * experts].astype(bool).reshape(layers, experts)
    if array_digest(keep) != update['keep_sha256']:
        raise ValueError('Phase update digest differs')
    return first, keep
