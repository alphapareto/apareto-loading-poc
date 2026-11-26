"""Request-bound selection transport and controlled rejection on the native scheduler."""
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.request import RequestStatus

from apareto_loader.config import ExpertConfig
from apareto_loader.store import Ranking
from apareto_loader.think import think_read_contract
from .concurrent_pause import SchedulerPause
from .selection_output import decode_phase, decode_update


class BankScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        expert = ExpertConfig.from_vllm(self.vllm_config)
        self.geometry = Ranking.expert_order()   # layers x experts
        self.selected_masks = {}
        # Thinking mode: per request, [(first position, keep)] in phase order (think, then answer).
        self.phase_masks = {}
        self.phase_kinds = {}   # rid -> phase of each phase_masks entry (think segments, then answer)
        spacer = think_read_contract(expert.think_bundle, 'think')
        if spacer:
            # KV slots for the rollback spacer at P..P+len-1 must be allocated, never the null block.
            self.num_lookahead_tokens = max(self.num_lookahead_tokens, len(spacer))
            # The worker appends the spacer to the final prompt chunk; leave room in the step's token buffers.
            if self.max_num_scheduled_tokens <= len(spacer):
                raise ValueError('max_num_batched_tokens leaves no room for the selector spacer')
            self.max_num_scheduled_tokens -= len(spacer)
        # Concurrent decode: a prefill chunk runs alone, and a request that
        # runs out of KV blocks is paused (rows to the host) instead of recomputed.
        self.concurrent = self.max_num_running_reqs > 1
        self.pause = SchedulerPause(self) if self.concurrent else None
        self.decode_only = False
        if self.concurrent:
            manager = self.kv_cache_manager
            native_allocate = manager.allocate_slots
            def allocate_slots(request, *args, **kwargs):
                # A decode step admits no new request: its prefill would share the step with decode rows.
                if self.decode_only and request.status == RequestStatus.WAITING:
                    return None
                return native_allocate(request, *args, **kwargs)
            manager.allocate_slots = allocate_slots

    def _fits(self, request):
        """vLLM's own admission of a waiting request (its whole sequence must fit), counted by its coordinator."""
        manager = self.kv_cache_manager
        full = min(request.num_tokens, self.max_model_len)
        blocks = manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id, num_tokens=full, new_computed_blocks=manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0, total_computed_tokens=0, num_local_computed_tokens=0, num_tokens_main_model=full,
            apply_admission_cap=True)
        return blocks + manager.watermark_blocks <= manager.block_pool.get_num_free_blocks()

    def _schedule_concurrent(self, *args, **kwargs):
        """A prefill (a running request still in its prompt, or a new request that fits) is scheduled alone: the
        decoding requests sit out that step. Otherwise every running request decodes and no new one is admitted."""
        decoders = [r for r in self.running if r.num_computed_tokens >= r.num_prompt_tokens]
        prefilling = [r for r in self.running if r.num_computed_tokens < r.num_prompt_tokens]
        head = self.waiting.peek_request() if self.waiting else None
        admit = (not prefilling and head is not None and head.status == RequestStatus.WAITING
                 and len(self.running) < self.max_num_running_reqs and self._fits(head))
        if prefilling or admit:
            saved, self.running = self.max_num_running_reqs, prefilling
            self.max_num_running_reqs = max(len(prefilling), 1)
            try:
                output = super().schedule(*args, **kwargs)
            finally:
                self.max_num_running_reqs = saved
                self.running = decoders + self.running
        else:
            self.decode_only = True
            try:
                output = super().schedule(*args, **kwargs)
            finally:
                self.decode_only = False
        self.pause.stamp(output)
        return output

    def schedule(self, *args, **kwargs):
        if self.concurrent:
            return self._schedule_concurrent(*args, **kwargs)
        return super().schedule(*args, **kwargs)

    def apply_phases(self, scheduler_output, model_runner_output):
        """Phase updates take effect before the step's rows are processed: the answer phase starts at the row
        the update arrives with. Idempotent per output (the audit scheduler applies them before auditing)."""
        if getattr(model_runner_output, 'apareto_phases_applied', False):
            return
        for update in self.live_updates(scheduler_output, model_runner_output):
            if update['kind'] == 'phase':
                rid = update['request_id']
                phases = self.phase_masks.setdefault(rid, [])
                kinds = self.phase_kinds.setdefault(rid, [])
                phases.append(decode_phase(update, rid, phases, self.geometry.layers, self.geometry.experts,
                                           kinds))
                kinds.append(update['phase'])
        if hasattr(model_runner_output, 'apareto_phases_applied'):
            model_runner_output.apareto_phases_applied = True

    def live_updates(self, scheduler_output, model_runner_output):
        """Updates of requests still known. With async scheduling a step can run after its request finished;
        its output, and anything attached, is stale and dropped as vLLM drops its tokens."""
        for update in getattr(model_runner_output, 'apareto_updates', []):
            rid = update['request_id']
            if rid not in scheduler_output.num_scheduled_tokens:
                raise ValueError('Selection metadata does not belong to this scheduled request')
            if rid in self.requests:
                yield update

    def update_from_output(self, scheduler_output, model_runner_output):
        self.apply_phases(scheduler_output, model_runner_output)
        for update in self.live_updates(scheduler_output, model_runner_output):
            rid = update['request_id']
            if update['kind'] == 'rejected':
                if len(scheduler_output.num_scheduled_tokens) != 1:
                    raise ValueError('Single-request rejection received a mixed batch')
                finished = self.finish_requests([rid], RequestStatus.FINISHED_ERROR)
                return {request.client_index: EngineCoreOutputs(outputs=[EngineCoreOutput(
                    request_id=request.request_id, new_token_ids=[], finish_reason=request.get_finished_reason(),
                    events=request.take_events(), trace_headers=request.trace_headers)]) for request in finished}
            if update['kind'] == 'phase':
                continue
            if rid in self.selected_masks:
                raise ValueError('A request received a second selector output')
            self.selected_masks[rid] = decode_update(update, rid, self.requests[rid].prompt_token_ids,
                                                     self.geometry.layers, self.geometry.experts)
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        for batch in outputs.values():
            for output in batch.outputs:
                if output.finish_reason is not None:
                    self.selected_masks.pop(output.request_id, None)
                    self.phase_masks.pop(output.request_id, None)
                    self.phase_kinds.pop(output.request_id, None)
        return outputs

    def finish_requests(self, request_ids, finished_status):
        requests = super().finish_requests(request_ids, finished_status)
        for request in requests:
            self.selected_masks.pop(request.request_id, None)
            self.phase_masks.pop(request.request_id, None)
            self.phase_kinds.pop(request.request_id, None)
        return requests


class AsyncBankScheduler(BankScheduler, AsyncScheduler):
    """Bank engine with async scheduling: the host prepares step n+1 while the GPU runs step n."""
