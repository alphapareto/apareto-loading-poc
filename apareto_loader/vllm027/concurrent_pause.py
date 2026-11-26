"""Pause and resume for concurrent decode: vLLM's preemption, without recompute.

The requests share one KV block pool. When a running request cannot get a block, vLLM preempts one (the last
admitted) and would later prefill all its tokens again; that would recompute masked decode state with every expert
(a masked request's state must come back exactly as it was). Here:
- the scheduler's `_preempt_request` first notes the request's block rows: its attention blocks behind the forwarded
  tokens and its GDN state block. The worker copies them to pinned host pages before anything is written in the
  step this schedule produces (the instruction rides that output, run before the runner's batch update), and only
  then does the native preemption free the blocks;
- at re-admission the computed-blocks seam takes fresh blocks for the attention rows, hands them to the request as
  computed (num_computed_tokens = the forwarded tokens), and the worker onloads the rows into them before the batch
  update (so vLLM's copy-on-write of a partial tail block reads the onloaded bytes); after the update it onloads the
  GDN rows into the state block vLLM gave the request. The request then decodes on from where it stopped;
- a request preempted before its prefill finished has no masked state and is refused (never expected: a prefill runs
  alone and is admitted only when its whole sequence fits).
Not prefix caching: nothing is hashed; `enable_caching` is flipped on the attention managers only so
`save_new_computed_blocks` accepts the handed-back blocks.
"""
import json

import torch
from vllm.v1.kv_cache_interface import MambaSpec

from .kv_offload import HostKVTier


def cdiv(a, b):
    return -(-int(a) // int(b))


class SchedulerPause:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        manager = scheduler.kv_cache_manager
        groups = manager.kv_cache_config.kv_cache_groups
        self.attention = [i for i, g in enumerate(groups) if not isinstance(g.kv_cache_spec, MambaSpec)]
        self.groups = len(groups)
        sizes = {groups[i].kv_cache_spec.block_size for i in self.attention}
        if len(sizes) != 1:
            raise ValueError(f'Pause needs one attention block size, got {sorted(sizes)}')
        self.block = sizes.pop()
        self.manager, self.pool = manager, manager.block_pool
        for index in self.attention:
            manager.coordinator.single_type_managers[index].enable_caching = True
        manager.coordinator.cache_blocks = lambda request, num_computed_tokens: None
        self.native_computed = manager.get_computed_blocks
        manager.get_computed_blocks = self.computed_blocks
        self.native_preempt = scheduler._preempt_request
        scheduler._preempt_request = self.preempt
        self.paused = {}        # request id -> forwarded tokens held on the host
        self.resuming = {}      # request id -> (held, per-group blocks) handed back this schedule
        self.ops = []           # worker instructions for the output this schedule produces
        self.counts = {'pauses': 0, 'resumes': 0, 'resume_waits': 0, 'paused_tokens': 0}

    def preempt(self, request, timestamp, drop_stale_output=False):
        rid = request.request_id
        held = request.num_computed_tokens
        if 0 < held < request.num_prompt_tokens:
            raise RuntimeError(f'{rid} preempted mid-prefill ({held} of {request.num_prompt_tokens} prompt tokens)')
        if held:
            blocks = self.manager.coordinator.get_blocks(rid)
            need = cdiv(held, self.block)
            attention = [b.block_id for g in self.attention for b in blocks[g][:need]]
            if any(len(blocks[g]) < need for g in self.attention):
                raise RuntimeError(f'{rid} holds fewer blocks than its {held} forwarded tokens')
            state = [b.block_id for g in range(self.groups) if g not in self.attention for b in blocks[g]]
            self.ops.append(('pause', rid, held, attention, state))
            self.paused[rid] = held
            self.counts['pauses'] += 1
            self.counts['paused_tokens'] += held
        return self.native_preempt(request, timestamp, drop_stale_output)

    def computed_blocks(self, request):
        rid = request.request_id
        held = self.paused.get(rid)
        if held is None:
            return self.native_computed(request)
        need = cdiv(held, self.block)
        if need * len(self.attention) > self.pool.get_num_free_blocks():
            self.counts['resume_waits'] += 1
            return self.manager.empty_kv_cache_blocks, 0, 0
        blocks = self._split(self.pool.get_new_blocks(need * len(self.attention)), need)
        self.resuming[rid] = (held, blocks)
        return self.manager.create_kv_cache_blocks(blocks), held, 0

    def _split(self, fresh, need):
        out, k = [], 0
        for g in range(self.groups):
            if g in self.attention:
                out.append(list(fresh[k * need:(k + 1) * need]))
                k += 1
            else:
                out.append([])
        return tuple(out)

    def stamp(self, output):
        """After the schedule: resumed requests took their own reference on the handed-back blocks (drop ours) and
        get their onload; a request whose blocks were taken but that was not admitted gives them back."""
        resumed = set(output.scheduled_cached_reqs.resumed_req_ids or ())
        for rid, (held, blocks) in self.resuming.items():
            flat = [b for group in blocks for b in group]
            # schedule() has advanced num_computed_tokens by the step's tokens already.
            start = self.scheduler.requests[rid].num_computed_tokens - output.num_scheduled_tokens.get(rid, 0)
            if rid in resumed and start == held:
                self.ops.append(('resume', rid, held, [b.block_id for g in self.attention for b in blocks[g]]))
                self.paused.pop(rid)
                self.counts['resumes'] += 1
            elif rid in resumed:
                raise RuntimeError(f'{rid} resumed at {start}, held {held}')
            else:
                self.counts['resume_waits'] += 1
            self.pool.free_blocks(flat)
        self.resuming = {}
        output.apareto_pause = self.ops
        if self.ops:
            print('APARETO pause ' + json.dumps({'ops': [op[:3] for op in self.ops], 'paused': sorted(self.paused),
                                                 **self.counts}), flush=True)
        self.ops = []


class WorkerPause:
    """The worker's side: host pages for paused requests' rows (kv_offload.HostKVTier, no snapshot slots: the GDN state
    travels as its own block rows)."""
    def __init__(self, runner, host_bytes, verify=False):
        self.runner = runner
        self.tier = HostKVTier(runner, [{}], verify, 0)
        self.tier.arena = torch.empty(int(host_bytes), dtype=torch.uint8, pin_memory=True)
        self.pending = {}       # request id -> held, waiting for its GDN onload after the batch update
        original = runner._update_states

        def update_states(scheduler_output):
            ops = getattr(scheduler_output, 'apareto_pause', None) or ()
            host = []
            for op in ops:
                if op[0] == 'pause':
                    _, rid, held, attention, state = op
                    host += [('copy', (rid, 'kv'), 0, held, attention), ('copy', (rid, 'gdn'), 0, held, state)]
                else:
                    _, rid, held, attention = op
                    host.append(('onload', (rid, 'kv'), 0, held, attention))
                    self.pending[rid] = held
            self.tier.run(host)
            result = original(scheduler_output)
            for rid in [r for r in self.pending if r in scheduler_output.num_scheduled_tokens]:
                held = self.pending.pop(rid)
                request = runner.requests[rid]
                if request.num_computed_tokens != held:
                    raise RuntimeError(f'{rid} resumes at {request.num_computed_tokens}, its rows hold {held}')
                state = [b for g, ids in enumerate(request.block_ids) if g not in self.attention_groups() for b in ids]
                self.tier.run([('onload', (rid, 'gdn'), 0, held, state)])
                self.tier.run([('drop', (rid, 'kv')), ('drop', (rid, 'gdn'))])
                print('APARETO pause-resumed ' + json.dumps({'request_id': rid, 'held': held, **self.tier.counts}),
                      flush=True)
            return result
        runner._update_states = update_states

    def attention_groups(self):
        groups = self.runner.kv_cache_config.kv_cache_groups
        return {i for i, g in enumerate(groups) if not isinstance(g.kv_cache_spec, MambaSpec)}
