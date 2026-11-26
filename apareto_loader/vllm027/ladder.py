"""The think-end ladder as a vLLM logits processor.

    --logits-processors apareto_loader.vllm027.ladder:LadderLogitsProcessor

The controller lives in apareto_loader/ladder.py. For async scheduling, tokens are observed in apply(), not in
update_state(). Under async scheduling vLLM 0.27.1 hands update_state() an
output list whose last entry is a -1 placeholder; the real id is written in just before sampling
(InputBatch.update_async_output_token_ids, called from GPUModelRunner._sample, and always when a custom logits
processor is loaded). Observing in apply() reads the same tokens in the same order with or without async, and
decides the forced token for the row being sampled.

Per-request config rides SamplingParams.extra_args["ladder"] as a JSON string (scripts/run_eval.py builds it from
the eval set's protocol); a request without it is untouched. Events (build, arm, inject, fence, force_miss) go to
<selection dir>/ladder.jsonl, one line each, flushed.
"""
import json
from pathlib import Path

from vllm.v1.sample.logits_processor.interface import LogitsProcessor, MoveDirectionality

from apareto_loader.ladder import LadderController
from apareto_loader.vllm027.run_paths import run_path


def build_controller(params):
    """The controller for one request, or None when it carries no ladder."""
    extra = getattr(params, 'extra_args', None) or {}
    cfg = extra.get('ladder')
    if not cfg:
        return None
    cfg = dict(json.loads(cfg) if isinstance(cfg, str) else cfg)
    cfg.setdefault('tag_id', 248069)
    cfg.setdefault('wait_cap', 200)
    cfg.setdefault('final_wait', 200)
    cfg.setdefault('nudge_wait', 1500)
    cfg.setdefault('safe_nudge', True)
    cfg.setdefault('fence0', 0)
    cfg['session'] = None              # events go to ladder.jsonl, not stdout (the controller prints when set)
    cfg['fixed'] = [tuple(x) for x in cfg.get('fixed', [])]
    ends = frozenset(cfg['ends_nl_ids'])
    cfg['ends_nl'] = ends.__contains__
    if cfg['nl_id'] not in ends:
        raise ValueError('ladder: nl_id must itself be a line boundary')
    return LadderController(cfg), str(extra.get('session', '?'))


class LadderLogitsProcessor(LogitsProcessor):
    """One instance per engine. State is keyed by the identity of the request's live output list, which survives
    the persistent batch's re-adds (one request can be re-added thousands of times in a minute)."""

    def __init__(self, vllm_config, device, is_pin_memory):
        self.ctrl = {}       # key -> [controller, out_list, seen, idle_updates, session, events_written]
        self.slots = {}      # batch index -> key
        self.no_ladder = {}  # key -> out_list, requests without a ladder config
        # The worker creates the selection directory after the processors are built, so the log opens on first use.
        directory = run_path(vllm_config, 'apareto_selection_dir', required=False)
        self.log_path = Path(directory) / 'ladder.jsonl' if directory else None
        self.log = None

    def is_argmax_invariant(self):
        return False         # greedy requests must still be forced

    def _write(self, st):
        ctrl = st[0]
        if self.log_path is not None and st[5] < len(ctrl.events):
            if self.log is None:
                self.log = self.log_path.open('x')
            for event in ctrl.events[st[5]:]:
                self.log.write(json.dumps({'session': st[4], **event}) + '\n')
            self.log.flush()
        st[5] = len(ctrl.events)

    def update_state(self, batch_update):
        if batch_update:
            for i in batch_update.removed:
                self.slots.pop(i, None)
            for index, params, _prompt_ids, out in batch_update.added:
                key = id(out)
                if self.no_ladder.get(key) is out:
                    self.slots.pop(index, None)
                    continue
                st = self.ctrl.get(key)
                if st is None or st[1] is not out:
                    built = build_controller(params)
                    if built is None:
                        self.no_ladder[key] = out
                        if len(self.no_ladder) > 256:
                            self.no_ladder.clear()
                        self.slots.pop(index, None)
                        continue
                    ctrl, session = built
                    self.ctrl[key] = st = [ctrl, out, 0, 0, session, 0]
                    ctrl.events.append({'event': 'build', 'at_tok': ctrl.t, 'final_at': ctrl.cfg['final']['at_tok']})
                    self._write(st)
                self.slots[index] = key
            for a, b, direct in batch_update.moved:
                ka, kb = self.slots.pop(a, None), self.slots.pop(b, None)
                if ka is not None:
                    self.slots[b] = ka
                if direct == MoveDirectionality.SWAP and kb is not None:
                    self.slots[a] = kb
        live = set(self.slots.values())
        for key, st in list(self.ctrl.items()):
            # Finished requests age out; the grace period keeps a preempted request's ladder.
            st[3] = 0 if key in live else st[3] + 1
            if st[3] > 4096:
                self.ctrl.pop(key, None)

    def apply(self, logits):
        for index, key in self.slots.items():
            st = self.ctrl.get(key)
            if st is None or index >= logits.shape[0]:
                continue
            ctrl, out, seen = st[0], st[1], st[2]
            forced = ctrl.pending[0] if ctrl.pending else None
            while seen < len(out):
                if out[seen] < 0:
                    raise RuntimeError(f'ladder: output token {seen} of {st[4]} is an unresolved placeholder at sampling')
                forced = ctrl.observe(out[seen])
                seen += 1
            st[2] = seen
            self._write(st)
            if forced is None:
                continue
            row = logits[index]
            if not 0 <= forced < row.shape[0]:
                raise ValueError(f'ladder: forced id {forced} outside the vocabulary ({row.shape[0]})')
            row.fill_(float('-inf'))
            row[forced] = 0.0
        return logits
