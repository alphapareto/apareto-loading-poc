"""Prefill step bookkeeping: which request's prompt rows a step runs, and whether it is a full (dense) prefill step."""
from contextlib import contextmanager
from dataclasses import dataclass

_active_step = None


@dataclass(frozen=True)
class PrefillStep:
    request_id: str
    start: int
    prompt: int
    count: int
    full: bool
    profiling: bool = False


@contextmanager
def step_context(step):
    global _active_step
    previous = _active_step
    _active_step = step
    try:
        yield
    finally:
        _active_step = previous


def full_step():
    return _active_step if _active_step is not None and _active_step.full else None


def request_prefill_step(output, cached_requests):
    """The step's request and prompt position, from CPU request positions, never token-count heuristics. The bank
    worker marks its prefill steps full itself."""
    if not output.total_num_scheduled_tokens:
        return None
    if len(output.num_scheduled_tokens) != 1:
        raise ValueError("A prefill step runs one request")
    rid, count = next(iter(output.num_scheduled_tokens.items()))
    new = next((r for r in output.scheduled_new_reqs if r.req_id == rid), None)
    if new is not None:
        if new.prompt_token_ids is None:
            raise ValueError("Prefill requires token-ID prompts")
        start, prompt = new.num_computed_tokens, len(new.prompt_token_ids)
    else:
        request = cached_requests[rid]
        index = output.scheduled_cached_reqs.req_ids.index(rid)
        start = output.scheduled_cached_reqs.num_computed_tokens[index]
        prompt = len(request.prompt_token_ids)
    if start < prompt < start + count:
        raise ValueError("A request step does not mix prefill and decode rows")
    return PrefillStep(rid, start, prompt, count, False)
