"""Bank-engine worker: dense prefill through the expert bank, R* selection, decode on the selected set.

Expert handling is the bank engine's:
- every prefill step is dense and loads, per layer, whatever the bank lacks;
- every request is a thinking request (rendered with thinking on): early R* predicts at layer k from the spacer rows
  (one-pass spacer); the think and answer bundles select; the bank loads the difference; decode runs on the
  selected set. A request that cannot be selected (thinking off, select=false, another budget, too long for the
  spacer) is refused: its one prefill step runs on the experts already in the bank and it ends with an error.
Decode must never load: the host-tier byte counter is checked around every decode step. Paged decode
(gpu_experts) is the exception: the bank holds part of the decode set, the kernel
reads the rest from pinned host memory, and each step's misses are admitted between steps.
Thinking mode: a prompt rendered with thinking on is prefilled dense and
decodes on the think mask. Both phase masks are known at the end of prefill: they are static, or selected from
the spacer rows by the think and answer networks, with the think network's early prediction at layer 4's entry. Once the
row whose input is `</think>` has been sampled, the answer experts not yet resident are loaded and decoding
continues on the answer mask. This swap is the one decode-time load.
Post-think answer read (answer_read='post_think'): the prefill read selects
only the think mask; at `</think>` a probe of the post-think spacer runs at q+1.. under the think mask as one extra
forward and is rolled back, and its read selects the answer mask, loaded by the one swap.
Midrun think masks (midrun_every): every midrun_every think tokens the same probe runs under the
live think mask, the midrun bundle reads it, and the think mask becomes the cut of (1 - w) * static field + w * midrun
field; only the experts it adds are loaded. Each refresh is a think segment in the phase updates.
Concurrent decode (max_num_seqs > 1): one state and one keep slot per request; the
scheduler runs a prefill chunk alone, and a decode step carries one row per request, each row dispatched under its own
request's keep row (engine.row_slot, written after the runner's batch update). The bank's decode set is the union of
the live requests' masks; each request's phase events (refreshes, answer swap, probes) run before the step, in turn.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker

from apareto_loader.bank import BankFull
from apareto_loader.config import ExpertConfig
from apareto_loader.selector import RuntimeSelector, SelectionRejected, array_digest
from apareto_loader.think import THINK_END, ThinkPlan, think_read_contract, think_read_rows
from .bank_model import BankRoutedExperts
from .run_paths import run_path
from .prefill import PrefillStep, full_step, request_prefill_step, step_context
from .selection_output import DeferredOutput, attach_updates, phase_update
from .spacer import SpacerRollback
from .tap_capture import TapCapture


_LAYER_ENTRY = {'callback': None}


@torch.library.custom_op("apareto::bank_layer_entry", mutates_args=("marker",))
def bank_layer_entry(marker: torch.Tensor) -> None:
    """Runs the engine's layer-entry callback at run time. A plain Python hook inside the compiled prefill
    forward is evaluated once while tracing and compiled away (the prediction would never run
    at layer 4's entry); an opaque custom op is called on every execution, as the tap capture is."""
    callback = _LAYER_ENTRY['callback']
    if callback is not None:
        callback()


@bank_layer_entry.register_fake
def _bank_layer_entry_fake(marker):
    return None


def decode_phases(totals):
    """Per decode phase (think, answer, or decode for a no-think request): steps, tokens, and the decode time from
    the phase's first decode step to its last sample. Timing only: no routing is read. Under async scheduling a
    step's tokens are counted when the engine collects them, one step late, so the think phase's last sample arrives
    after the answer swap (with its post-think probe) has run; the swap's interval is recorded on the think totals
    ('pauses') and every pause that lies inside the phase's clock is subtracted, so `seconds` is decode only (the
    swap's own time is in its think_swap row)."""
    out = {}
    for phase, t in totals.items():
        paused = sum(b - a for a, b in t.get('pauses', ()) if a >= t['first'] and b <= t['last'])
        seconds = t['last'] - t['first'] - paused
        out[phase] = {'steps': t['steps'], 'tokens': t['tokens'], 'seconds': seconds, 'paused_seconds': paused,
                      'tokens_per_second': t['tokens'] / seconds if seconds > 0 else None,
                      'tokens_per_step': t['tokens'] / t['steps']}
    return out


class BankWorker(Worker):
    def initialize_from_config(self, kv_cache_config):
        # The KV layout vLLM chose: which layers share a group and block ids, and the GDN state shapes. Printed once
        # so offline checks can read it from the log.
        groups = []
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            entry = {'spec': type(spec).__name__, 'block_size': spec.block_size,
                     'page_size_bytes': spec.page_size_bytes, 'layers': list(group.layer_names)}
            for name in ('shapes', 'num_speculative_blocks', 'mamba_cache_mode', 'num_kv_heads', 'head_size'):
                if hasattr(spec, name):
                    entry[name] = getattr(spec, name)
            if hasattr(spec, 'dtypes'):
                entry['dtypes'] = [str(d) for d in spec.dtypes]
            groups.append(entry)
        print('APARETO kv-cache-groups ' + json.dumps({
            'num_blocks': kv_cache_config.num_blocks, 'groups': groups,
            'tensors': [{'size': t.size, 'shared_by': list(t.shared_by)} for t in kv_cache_config.kv_cache_tensors]},
            default=str), flush=True)
        return super().initialize_from_config(kv_cache_config)

    def load_model(self, **kwargs):
        Worker.load_model(self, **kwargs)
        model = self.model_runner.get_model()
        before = torch.cuda.memory_allocated(self.device)
        self.config = ExpertConfig.from_vllm(self.vllm_config)
        self.engine = model.apareto_engine
        if self.config.think_bundle is None:
            raise ValueError('The bank engine serves thinking requests: it needs a think bundle')
        # The prefill read: the think bundle's spacer. The midrun and answer bundles must read the same one.
        self.spacer = SpacerRollback(model, self.vllm_config, think_read_contract(self.config.think_bundle, 'think'))
        # A verification run (an audit path is set): paused requests' KV is read back after every restore.
        self.verifying = run_path(self.vllm_config, 'apareto_audit_path', required=False) is not None
        self._load_think()
        directory = Path(run_path(self.vllm_config, 'apareto_selection_dir'))
        directory.mkdir(exist_ok=False)
        self.selection_directory = directory
        self._load_midrun(directory)
        startup = {'read': 'rollback_spacer', 'spacer_token_ids': list(self.spacer.spacer),
                   'spacer_state_scratch_bytes': self.spacer.scratch_bytes, 'engine': 'bank',
                   'load_report': model.apareto_load_report,
                   'think': self._think_record(), 'midrun': self.midrun_record}
        with (directory / 'startup.json').open('x') as output:
            json.dump(startup, output)
        self.selection_log = (directory / 'selections.jsonl').open('x')
        self.prefill_log = Path(run_path(self.vllm_config, 'apareto_prefill_log')).open('x')
        self.taps = TapCapture(self.device); model.add_module('apareto_taps', self.taps); self.taps.attach(model)
        self._attach_prediction_hook(model)
        self.experts = {m.apareto_layer: m for m in model.modules() if isinstance(m, BankRoutedExperts)}
        self.request_state = None
        self.stager = None
        self.sample_position = None   # position of the token the current step samples
        self.decode_step = None       # (phase, host time the step began) of the current decode step: decode_phases
        self.spacer_rows = 0
        self.pending_updates = []
        self.apareto_added_model_buffer_bytes = torch.cuda.memory_allocated(self.device) - before
        self.model_runner.model_memory_usage += self.apareto_added_model_buffer_bytes
        if os.environ.get('APARETO_STEP_TIMING'):
            self._time_steps(os.environ['APARETO_STEP_TIMING'])
        self.profile = None
        if os.environ.get('APARETO_PROFILE_DECODE'):
            # Measurement only (APARETO_PROFILE_DECODE=<new directory>:<context>,<context>,...): when a decode step
            # first starts at or beyond each context length, the torch profiler records that step and the next 19
            # (forward and sampling) and writes per-kernel CUDA time to profile_<context>.txt.
            directory, contexts = os.environ['APARETO_PROFILE_DECODE'].rsplit(':', 1)
            Path(directory).mkdir(exist_ok=False)
            self.profile = {'directory': Path(directory), 'pending': sorted(int(c) for c in contexts.split(',')),
                            'active': None, 'steps': 0}
        self.memory_snapshots = None
        if os.environ.get('APARETO_MEMORY_SNAPSHOT'):
            # Measurement only (APARETO_MEMORY_SNAPSHOT=<new directory>): the allocator records every GPU allocation's
            # stack from here on; at the end of each of the first 3 requests the live allocations are dumped
            # (torch.cuda.memory._dump_snapshot) and the allocator's counters appended to stats.jsonl.
            self.memory_snapshots = Path(os.environ['APARETO_MEMORY_SNAPSHOT'])
            self.memory_snapshots.mkdir(exist_ok=False)
            self.memory_snapshot_count = 0
            torch.cuda.memory._record_memory_history(max_entries=200000)
        self.rows = self.vllm_config.scheduler_config.max_num_seqs
        self.states, self.slots, self.step_infos, self.row_requests = {}, {}, None, []
        if self.rows > 1:
            self._concurrent_setup()
        if self.engine.rescue is not None and self.engine.rescue.get('gain') == 'static_think':
            if self.rows > 1:
                self.engine.cpu_experts.slot_gain(self.rows, self.engine.device)
            self.rescue_gain_keys = {}


    def _concurrent_setup(self):
        """Concurrent decode: free keep slots, and the hook that writes each token
        row's keep slot after the runner's batch update (rows follow the persistent batch's order)."""
        if not self.engine.paged:
            raise ValueError('Concurrent decode needs paged decode')
        rescue = self.engine.rescue
        if rescue is not None and set(rescue['phases']) != {'think', 'answer'}:
            # The rescue switch is per step; with every decode phase rescuing it never differs between rows.
            raise ValueError('Concurrent decode with CPU rescue needs phases=think+answer')
        self.free_slots = list(range(self.rows))
        runner = self.model_runner
        # Paused requests' KV and GDN rows (concurrent_pause.py): at most rows - 1 paused, each at most the whole pool.
        from .concurrent_pause import WorkerPause
        kv = self.vllm_config.cache_config.kv_cache_memory_bytes
        self.pause = WorkerPause(runner, (self.rows - 1) * int(kv * 1.05), verify=self.verifying)
        tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.slot_staging = [torch.full((tokens,), -1, dtype=torch.int32).pin_memory() for _ in range(4)]
        self.slot_staging_next = 0
        self.async_scheduling = bool(self.vllm_config.scheduler_config.async_scheduling)
        self.known_outputs = {}      # request id -> its sampled tokens collected so far (async: re-adding a row)
        self.step_prev = None        # async: (prev_sampled_token_ids, prev_req_id_to_index) as this step began
        original = runner._update_states
        def update_states(scheduler_output):
            if self.async_scheduling:
                self._restore_dropped(scheduler_output)
            result = original(scheduler_output)
            batch, counts = runner.input_batch, scheduler_output.num_scheduled_tokens
            host = self.slot_staging[self.slot_staging_next % len(self.slot_staging)]
            self.slot_staging_next += 1
            view = host.numpy()
            view[:] = -1
            row, requests = 0, []
            for rid in batch.req_ids[:batch.num_reqs]:
                n = counts.get(rid, 0)
                view[row:row + n] = self.slots.get(rid, -1)
                requests += [rid] * n
                row += n
            self.engine.row_slot.copy_(host, non_blocking=True)
            self.row_requests = requests
            return result
        runner._update_states = update_states

    def _restore_dropped(self, scheduler_output):
        """Async concurrent decode: a probe's batch update drops the other requests from the persistent batch, which
        the scheduler does not know. vLLM re-adds a request missing from the batch from `all_token_ids`, which the
        scheduler sends only for requests it left out itself; here the worker supplies them for the ones a probe
        dropped: the prompt, the outputs collected so far, and the in-flight step's token (on the GPU until its output
        is collected; read from the step's previous-sample buffer, a small sync on probe steps only)."""
        cached = scheduler_output.scheduled_cached_reqs
        batch = self.model_runner.input_batch
        if cached.all_token_ids is None:
            cached.all_token_ids = {}
        for i, rid in enumerate(cached.req_ids):
            if (rid in batch.req_id_to_index or rid in (cached.resumed_req_ids or ()) or rid in cached.all_token_ids
                    or not cached.num_output_tokens[i]):
                continue
            state = self.model_runner.requests[rid]
            known = self.known_outputs.get(rid, [])
            missing = cached.num_output_tokens[i] - len(known)
            tail = []
            if missing == 1:
                sampled, index = self.step_prev if self.step_prev is not None else (None, None)
                if sampled is None or not index or rid not in index:
                    raise RuntimeError(f'{rid}: dropped with a token in flight that the previous step did not sample')
                tail = [int(sampled[index[rid], 0])]
            elif missing != 0:
                raise RuntimeError(f'{rid}: {missing} output tokens unaccounted for when re-adding its row')
            cached.all_token_ids[rid] = list(state.prompt_token_ids) + list(known) + tail

    def _time_steps(self, path):
        """Measurement only (APARETO_STEP_TIMING=<new file>): per step, host time in this worker's own code,
        in vLLM's model step (input preparation + graph launch), in sampling (which waits for the GPU), and
        the gap since the previous sample returned (engine core: output handling, scheduling)."""
        import time
        log = open(path, 'x')
        execute, sample, run = self.execute_model, self.sample_tokens, self._run_model
        state = {'run': 0.0, 'last_sample_end': None}
        def timed_run(scheduler_output):
            start = time.perf_counter()
            try:
                return run(scheduler_output)
            finally:
                state['run'] += time.perf_counter() - start
        def timed_execute(scheduler_output):
            start = time.perf_counter(); state['run'] = 0.0
            result = execute(scheduler_output)
            end = time.perf_counter()
            state['row'] = {'tokens': scheduler_output.total_num_scheduled_tokens,
                            'gap_since_sample_ms': None if state['last_sample_end'] is None else 1e3 * (start - state['last_sample_end']),
                            'execute_ms': 1e3 * (end - start), 'vllm_model_ms': 1e3 * state['run'],
                            'worker_own_ms': 1e3 * (end - start - state['run'])}
            return result
        def timed_sample(grammar_output):
            start = time.perf_counter()
            result = sample(grammar_output)
            end = time.perf_counter(); state['last_sample_end'] = end
            row = state.pop('row', {})
            row['sample_ms'] = 1e3 * (end - start)
            log.write(json.dumps(row) + '\n'); log.flush()
            return result
        self._run_model, self.execute_model, self.sample_tokens = timed_run, timed_execute, timed_sample

    def _profile_begin(self, context):
        profile = self.profile
        if profile['active'] is not None or not profile['pending'] or context < profile['pending'][0]:
            return
        while profile['pending'] and context >= profile['pending'][0]:
            target = profile['pending'].pop(0)
        torch.cuda.synchronize()
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA])
        profiler.__enter__()
        profile.update(active=profiler, steps=0, target=target, context=context, began=time.perf_counter())

    def _profile_step(self):
        profile = self.profile
        profile['steps'] += 1
        if profile['steps'] < 20:
            return
        torch.cuda.synchronize()
        seconds = time.perf_counter() - profile['began']
        profiler, profile['active'] = profile['active'], None
        profiler.__exit__(None, None, None)
        table = profiler.key_averages().table(sort_by='cuda_time_total', row_limit=60, max_name_column_width=90)
        with (profile['directory'] / f'profile_{profile["target"]}.txt').open('x') as output:
            output.write(f'context at first step {profile["context"]}, steps {profile["steps"]}, '
                         f'wall {seconds:.4f} s\n{table}\n')
        profiler.export_chrome_trace(str(profile['directory'] / f'trace_{profile["target"]}.json'))

    def _memory_snapshot(self, request_id):
        index, self.memory_snapshot_count = self.memory_snapshot_count, self.memory_snapshot_count + 1
        torch.cuda.synchronize()
        stats = torch.cuda.memory_stats(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        row = {'index': index, 'request_id': request_id, 'free_bytes': free, 'total_bytes': total,
               **{key: stats[key] for key in ('allocated_bytes.all.current', 'allocated_bytes.all.peak',
                                              'reserved_bytes.all.current', 'reserved_bytes.all.peak')}}
        with (self.memory_snapshots / 'stats.jsonl').open('a') as output:
            output.write(json.dumps(row) + '\n')
        if index < 3:
            torch.cuda.memory._dump_snapshot(str(self.memory_snapshots / f'snapshot_{index}.pickle'))

    def _attach_prediction_hook(self, model):
        """Early R* at layer_cap's entry: a forward hook on that layer's input norm, registered after the
        tap hooks so hl{layer_cap} is already captured. Only dense prefill steps (eager) predict."""
        from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
        cap = self.engine.predictor.layer_cap
        layer = next(m for m in model.modules() if isinstance(m, Qwen3_5DecoderLayer) and m.layer_idx == cap)
        engine = self.engine
        def at_entry():
            step = full_step()
            if step is not None and not step.profiling and engine.stats is not None:
                engine.at_layer_entry(cap)
        _LAYER_ENTRY['callback'] = at_entry
        marker = torch.zeros(1, dtype=torch.int8, device=self.device)
        model.register_buffer('apareto_layer_entry_marker', marker)
        def entry_hook(module, args, output):
            bank_layer_entry(marker)
        layer.input_layernorm.register_forward_hook(entry_hook)

    @torch.inference_mode()
    def determine_available_memory(self):
        """Native memory and graph profiling (decode, plus a dense prefill through the profiling table)."""
        requested = self.cache_config.kv_cache_memory_bytes
        self.cache_config.kv_cache_memory_bytes = None
        original = self.model_runner.profile_run
        def profile_both(*args, **kwargs):
            original(*args, **kwargs)
            with step_context(PrefillStep('profile', 0, 1, 1, True, profiling=True)):
                return original(*args, **kwargs)
        self.model_runner.profile_run = profile_both
        # Exact-byte budget: free memory at startup minus the margin, not a utilization fraction.
        self.requested_memory = self.init_snapshot.free_memory - self.config.gpu_margin_bytes
        try:
            available = Worker.determine_available_memory(self)
        finally:
            self.cache_config.kv_cache_memory_bytes = requested
            self.model_runner.profile_run = original
        kv = requested if requested is not None else available
        estimate = {'V_bytes': self.requested_memory, 'budget': 'free at startup - gpu_margin_bytes',
                    'gpu_margin_bytes': self.config.gpu_margin_bytes, 'native_available_kv_bytes': available,
                    'spare_after_kv_bytes': available - (requested if requested is not None else available),
                    'kv_ceiling': 'none: bank_loader exact-byte check plus the warm-up request',
                    'W_kv_requested_bytes': kv, 'bank_slots': self.engine.capacity,
                    'bank_bytes': self.model_runner.get_model().apareto_load_report['bank_bytes'],
                    'source': 'native startup activation and CUDA graph memory profile with the bank allocated'}
        # No ceiling from the profile: its non-KV term varies between boots of one configuration
        # (0.04-0.97 GiB), so it would refuse fits that work. The exact-byte check is bank_loader's
        # bank + KV + activation reserve + margin <= free; the serve script's warm-up request exercises the
        # largest prefill chunk before ready. The native number stays in the record.
        with (self.selection_directory / 'startup_memory.json').open('x') as output:
            json.dump(estimate, output, indent=2)
        print('APARETO startup memory: ' + json.dumps(estimate), flush=True)
        return kv

    def _features(self):
        # One line, once per request: did each tap's hook actually fire and write its rows? `seen` is
        # set by the capture kernel itself, so a zero here means the hook never ran, while a nonzero
        # seen with NaN values would mean it ran and wrote nothing usable. 'not finite' alone cannot
        # tell those apart.
        if not getattr(self, '_tap_seen_logged', False):
            self._tap_seen_logged = True
            from apareto_loader.taps import TAPS as _T
            counts = self.taps.seen.sum(dim=1).tolist()
            bad = self.taps.srcbad.sum(dim=1).tolist()
            print('APARETO tap seen rows/8: ' + ' '.join(f'{n}={c}' for n, c in zip(_T, counts)), flush=True)
            print('APARETO tap SOURCE nonfinite rows/8: '
                  + ' '.join(f'{n}={b}' for n, b in zip(_T, bad)), flush=True)
        return self.taps.features.float().cpu().numpy()

    def _release_concurrent(self, request_id):
        """A finished request of concurrent decode: its logs, its keep slot back; the engine's decode bookkeeping ends
        with the last live request."""
        state = self.states.pop(request_id)
        self.free_slots.append(self.slots.pop(request_id))
        if self.memory_snapshots is not None:
            # Counters at every release (the peaks run over the whole boot); stack dumps for the first 3 only.
            self._memory_snapshot(request_id)
        if state.get('staging') is not None:
            state['staging'].result()
            self.engine.loader.host.release_held()
        if state['decode']:
            self.selection_log.write(json.dumps({'request_id': request_id, 'kind': 'decode_phases',
                                                 'phases': decode_phases(state['decode'])}) + '\n')
        if self.routing_log is not None and self.engine.decode_stats is not None and state.get('routing_positions'):
            self._write_routing(state)
        if not self.states and self.engine.decode_stats is not None:
            self._log_epochs(final=True)
            self.engine.decode_stats = None
            self.engine.loader.host.release_decode()
        self.selection_log.flush()

    def _log_epochs(self, final=False):
        """Concurrent decode: the engine's paged-decode counters per union epoch (between two changes of the decode
        set), with the requests live in it; per-request counters are not separable when rows share a step."""
        engine = self.engine
        rows = list(engine.decode_stats_history)
        engine.decode_stats_history = []
        if final and engine.decode_stats is not None and engine.decode_stats['steps']:
            stats = engine.decode_stats
            if engine.cpu_experts is not None and engine.cpu_stats_start is not None:
                now = engine.cpu_experts.stats()
                stats['cpu_experts'] = {k: now[k] - engine.cpu_stats_start[k] for k in now}
            rows.append(stats)
        for stats in rows:
            self.selection_log.write(json.dumps({'kind': 'paged_decode_epoch', 'requests': sorted(self.states),
                                                 **stats}) + '\n')

    def _release(self):
        state = self.request_state
        if self.memory_snapshots is not None and state is not None:
            self._memory_snapshot(state['request_id'])     # counters each request; stack dumps for the first 3
        if state is not None and state.get('staging') is not None:
            state['staging'].result()
            self.engine.loader.host.release_held()
        if state is not None and state['decode']:
            self.selection_log.write(json.dumps({'request_id': state['request_id'], 'kind': 'decode_phases',
                                                 'phases': decode_phases(state['decode'])}) + '\n')
            self.selection_log.flush()
        if self.routing_log is not None and state is not None and self.engine.decode_stats is not None:
            self._write_routing(state)
        if self.engine.paged:
            stats, self.engine.decode_stats = self.engine.decode_stats, None
            if stats is not None and self.engine.cpu_experts is not None and self.engine.cpu_stats_start is not None:
                now, start = self.engine.cpu_experts.stats(), self.engine.cpu_stats_start
                stats['cpu_experts'] = {k: now[k] - start[k] for k in now}
            history, self.engine.decode_stats_history = self.engine.decode_stats_history, []
            if state is not None and stats is not None:
                for index, phase in enumerate([*history, stats]):
                    self.selection_log.write(json.dumps({'request_id': state['request_id'], 'kind': 'paged_decode',
                                                         'phase_index': index, **phase}) + '\n')
                self.selection_log.flush()
            # The request has finished, so no queued kernel still reads its host records.
            self.engine.loader.host.release_decode()
        self.taps.release(); self.request_state = None

    def _reject(self, request_id, error, row=None):
        self.pending_updates.append({'kind': 'rejected', 'request_id': request_id, 'error': error})
        self.selection_log.write(json.dumps({**(row or {'request_id': request_id}), 'selection_applied_to_decode': False,
                                             'error': error}) + '\n')
        self.selection_log.flush()
        if self.request_state is not None and self.request_state['request_id'] == request_id:
            self.request_state['rejected'] = True

    def _attach(self, result):
        if result is None:
            return None
        updates, self.pending_updates = self.pending_updates, []
        return attach_updates(result, updates)

    @contextmanager
    def _eager(self):
        original = self.model_runner._determine_batch_execution_and_padding
        def force_eager(*args, **kwargs):
            kwargs['force_eager'] = True
            return original(*args, **kwargs)
        self.model_runner._determine_batch_execution_and_padding = force_eager
        try:
            yield
        finally:
            self.model_runner._determine_batch_execution_and_padding = original

    def _run_model(self, scheduler_output):
        if not self.spacer_rows:
            return Worker.execute_model(self, scheduler_output)
        return self._run_with_spacer(scheduler_output)

    def _run_with_spacer(self, scheduler_output):
        """Append the spacer to this final prompt chunk; sample the first token from the last prompt row."""
        rows = self.spacer_rows
        runner = self.model_runner
        (rid, count), = scheduler_output.num_scheduled_tokens.items()
        commit = len(runner.requests[rid].prompt_token_ids) if rid in runner.requests else None
        output = replace(scheduler_output, num_scheduled_tokens={rid: count + rows},
                         total_num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens + rows)
        original = runner._prepare_inputs
        def prepare(batch_output, num_scheduled_tokens):
            index = runner.input_batch.req_id_to_index[rid]
            prompt = len(runner.requests[rid].prompt_token_ids)
            if commit is not None and prompt != commit:
                raise ValueError('Prompt changed under the spacer step')
            if (runner.input_batch.num_tokens_no_spec[index] != prompt
                    or runner.input_batch.num_computed_tokens_cpu[index] + count != prompt):
                raise ValueError('Spacer must follow the final prompt chunk directly')
            tokens = runner.input_batch.token_ids_cpu
            saved = tokens[index, prompt:prompt + rows].copy()
            tokens[index, prompt:prompt + rows] = self.spacer.spacer
            try:
                logits, spec = original(batch_output, num_scheduled_tokens)
            finally:
                tokens[index, prompt:prompt + rows] = saved
            if spec is not None or logits.numel() != 1:
                raise ValueError('Spacer step must sample one request')
            return logits - rows, spec
        runner._prepare_inputs = prepare
        try:
            return Worker.execute_model(self, output)
        finally:
            del runner._prepare_inputs

    def _publish(self, keep, request_id=None):
        """The keep-set into every layer's keep row: the only row, or (concurrent decode) the request's slot."""
        slot = None if self.rows == 1 else self.slots[request_id]
        for layer, module in self.experts.items():
            row = torch.from_numpy(np.array(keep[layer], dtype=bool))
            (module.apareto_keep if slot is None else module.apareto_keep[slot]).copy_(row)

    def _decode_set(self, state, request_id, keep, row):
        """Load the decode set's missing experts; returns False (request rejected) if the bank cannot."""
        target = keep
        if self.rows > 1:
            # The bank serves the union of the live requests' masks; each row still dispatches under its own.
            target = keep.copy()
            for other, other_state in self.states.items():
                if other != request_id and other_state.get('keep') is not None:
                    target |= other_state['keep']
        try:
            prepare = self.engine.prepare_paged if self.engine.paged else self.engine.prepare_decode
            loaded, events = prepare(target)
        except BankFull as error:
            self._reject(request_id, f'Bank cannot hold the decode set: {error}', row)
            return False
        for event in events:
            torch.cuda.current_stream().wait_event(event)
        self._publish(keep, request_id)
        if self.rows > 1:
            row['union_decode_set'] = int(target.sum())
            self._log_epochs()
        state['keep'] = keep
        state['decode_loaded'] = loaded
        return True

    def _load_midrun(self, directory):
        """Midrun think masks: the midrun bundle's selector, which reads the think spacer; optionally the routing log
        (verification only)."""
        config = self.config
        self.midrun = self.routing_log = None
        self.midrun_record = None
        if config.midrun_every is not None:
            if self.think_selector is None:
                raise ValueError('Midrun masks blend with the selected think field; serve a think bundle')
            spacer = tuple(json.loads((Path(config.midrun_bundle) / 'read_contract.json').read_text())['spacer_token_ids'])
            if spacer != self.spacer.spacer:
                raise ValueError('The midrun bundle reads a different spacer than the think bundle')
            self.midrun = RuntimeSelector(config.midrun_bundle)
            self.midrun.manifest_check()
            self.midrun.check_budget(self.think_budget)
            # Block 0 is cut by the think bundle and later blocks by the midrun bundle (both at the think budget): a
            # per-layer quota (layer_quota.json) must be the same in both, or the resident set changes shape mid-think.
            quota = self.midrun.reference.quota(self.think_budget)
            if not np.array_equal(quota, self.think_selector.reference.quota(self.think_budget)):
                raise ValueError('The midrun and think bundles cut different per-layer quotas at the think budget')
            manifest = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(config.midrun_bundle).iterdir())}
            self.midrun_record = {'every': config.midrun_every, 'weight': config.midrun_weight,
                                  'layer_quota': quota.tolist(),
                                  'bundle': config.midrun_bundle, 'bundle_sha256': manifest,
                                  'top_k': self.midrun.reference.k}
        if self.engine.rescue is not None and self.engine.rescue.get('gain') == 'static_think' and self.think_selector is None:
            raise ValueError('The rescue rule scores against the static think cut: serve a think bundle')
        if config.routing_log:
            self.engine.enable_premask()
            self.routing_log = {'directory': directory / 'routing', 'count': 0}
            self.routing_log['directory'].mkdir()

    def _midrun_refresh(self, state, start):
        """Before the row at `start` (think token m = start - prompt): probe under the live mask, read, blend, cut,
        load the difference. A probe that cannot run or a read the selector rejects keeps the mask and says so."""
        rid, midrun = state['request_id'], state['midrun']
        row = {'request_id': rid, 'kind': 'midrun_swap', 'segment': midrun['segment'] + 1, 'position': start,
               'think_pos': start - len(state['tokens'])}
        began = time.perf_counter()
        midrun['next'] += self.config.midrun_every
        if start + len(self.spacer.spacer) > self.model_runner.max_model_len:
            row['skipped'] = 'probe exceeds max_model_len'
        else:
            features = self._probe_forward(rid, start, self.spacer.spacer)
            row['probe_seconds'] = time.perf_counter() - began
            row['probe_features_sha256'] = hashlib.sha256(features.tobytes()).hexdigest()
            try:
                field = self.midrun.fused(features)
            except SelectionRejected as error:
                row['skipped'] = f'midrun read rejected: {error}'
            else:
                w = self.config.midrun_weight
                keep = self.midrun.cut((1 - w) * midrun['static'] + w * field, self.think_budget)
                previous = state['keep']
                host, tier, before = self.engine.loader.host, dict(self.engine.loader.host.stats), self.engine.loader.bytes
                row.update(previous_keep_sha256=array_digest(previous), keep_sha256=array_digest(keep),
                           added=int((keep & ~previous).sum()), dropped=int((previous & ~keep).sum()),
                           moved_from_static=int((keep & ~midrun['static_keep']).sum()))
                swap = time.perf_counter()
                if not self._decode_set(state, rid, keep, row):
                    state['phase'] = 'rejected'
                    return
                torch.cuda.synchronize()
                row.update(swap_seconds=time.perf_counter() - swap, swap_loaded=state['decode_loaded'],
                           swap_loaded_bytes=self.engine.loader.bytes - before,
                           swap_host_tier={k: host.stats[k] - tier[k] for k in ('hits', 'misses', 'nvme_bytes', 'evictions')})
                midrun['segment'] += 1
                self.pending_updates.append(phase_update(rid, 'think', start, keep))
        row['seconds'] = time.perf_counter() - began
        think = state['decode'].get('think')
        if think is not None:     # decode_phases subtracts it from think's clock, as for the answer swap
            think.setdefault('pauses', []).append((began, time.perf_counter()))
        self.selection_log.write(json.dumps(row) + '\n')
        self.selection_log.flush()

    def _write_routing(self, state):
        """Routing log (verification only): the request's pre-mask router top-8 per decode step, each slot's fate, and the
        request's static think cut (the rescue rule's in-cut set), one file per request."""
        rows = self.engine.take_premask(state['request_id'] if self.rows > 1 else None)
        positions = state.get('routing_positions', [])
        log = self.routing_log
        path = log['directory'] / f"{log['count']:04d}.npz"
        log['count'] += 1
        if len(rows) != len(positions):
            raise RuntimeError(f'Routing log has {len(rows)} steps for {len(positions)} decode launches')
        with path.open('xb') as output:
            static = state.get('static_keep')
            np.savez(output, request_id=np.array(state['request_id']), prompt_len=np.int64(len(state['tokens'])),
                     static_keep=np.zeros((40, 256), bool) if static is None else np.asarray(static, bool),
                     close_position=np.int64(-1 if state['close_position'] is None else state['close_position']),
                     positions=np.asarray(positions, np.int64),
                     ids=np.stack([r[0] for r in rows]) if rows else np.zeros((0, 40, 8), np.int16),
                     weights=np.stack([r[1] for r in rows]).astype(np.float16) if rows else np.zeros((0, 40, 8), np.float16),
                     # per slot: 1 kept, 0 dropped, 2 dropped and rescued (on the CPU, or on the GPU when resident)
                     state=np.stack([r[2] for r in rows]) if rows else np.zeros((0, 40, 8), np.int8),
                     # per slot: L2 norm of a CPU-computed expert output before its routing weight;
                     # -1 not computed on the CPU (kept on the GPU, dropped, or rescued from the bank), -2 timed out,
                     # NaN without the CPU path. Mass of a rescue = weight x norm.
                     norms=np.stack([r[3] for r in rows]).astype(np.float32) if rows else np.zeros((0, 40, 8), np.float32))

    def _load_think(self):
        """Thinking mode: the phase plan, the bundles' selectors (each checked against its manifest) and the early
        prediction (the engine's predictor, bank_loader). Every think-open read is the think bundle's prefill spacer,
        so the one-pass spacer and its rollback serve them all."""
        config = self.config
        self.think = self.think_selector = self.answer_selector = None
        self.think_early = self.engine.predictor
        self.post_think_spacer = None
        self.think_read_raw = False     # think-open reads take the last 8 prompt rows, not the spacer rows
        from transformers import AutoTokenizer
        model_config = self.vllm_config.model_config
        tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer, revision=model_config.tokenizer_revision,
                                                  trust_remote_code=model_config.trust_remote_code)
        self.think = ThinkPlan.load(tokenizer)
        self.think_tests = {}
        rows = {}
        post = True     # the answer bundle reads the post-think probe at `</think>`
        for name, path, phase, variant in (
                ('think_selector', config.think_bundle, 'think', 'think_open'),
                ('answer_selector', config.answer_bundle, 'answer', 'post_think' if post else 'think_open')):
            if path is None:
                continue
            spacer = think_read_contract(path, phase, variant)
            rows[phase] = think_read_rows(path)
            if variant == 'post_think':
                if rows[phase] != 'spacer':
                    raise ValueError('The post-think probe reads its spacer rows; a raw-row answer bundle is not served')
                del rows[phase]
                # The probe writes KV at q+1..q+len; the scheduler's lookahead is sized by the prefill spacer.
                if len(spacer) != len(self.spacer.spacer):
                    raise ValueError('The post-think probe must be as long as the prefill spacer')
                self.post_think_spacer = spacer
            elif spacer != self.spacer.spacer:
                raise ValueError(f'The {phase} bundle reads a different spacer than the think bundle')
            selector = RuntimeSelector(path)
            self.think_tests[phase] = selector.manifest_check()
            setattr(self, name, selector)
        # One prefill read serves every think-open bundle; the early prediction reads the think bundle's shallow taps.
        if len(set(rows.values())) > 1:
            raise ValueError(f'Think-mode bundles read different rows: {rows}')
        self.think_read_raw = set(rows.values()) == {'raw'}
        # Selectors cut at the served budget (--expert-budget, which also sizes the bank); fixed masks must match it.
        budgets = {int(m.sum()) for m in (self.think.think, self.think.answer) if m is not None}
        if self.think_selector is not None or self.answer_selector is not None:
            budgets.add(int(config.budget))
        if len(budgets) != 1:
            raise ValueError(f'Think and answer phases must have one budget, got {sorted(budgets)}')
        self.think_budget = budgets.pop()

    def _think_record(self):
        if self.think is None:
            return None
        return {**self.think.record(), 'budget': self.think_budget, 'think_bundle': self.config.think_bundle,
                'midrun_at_prefill': self.config.midrun_at_prefill, 'answer_bundle': self.config.answer_bundle,
                'answer_read': 'post_think', 'think_read_rows': 'raw' if self.think_read_raw else 'spacer',
                'post_think_spacer': list(self.post_think_spacer) if self.post_think_spacer else None,
                'bundle_checks': self.think_tests}

    def _admit(self, extra, tokens):
        """A new request's rejection, or None. Served: a selected request whose prompt ends in the thinking-on
        generation prompt, at the served budget, with room for the spacer."""
        if extra.get('select', True) not in (True, 1):
            return 'This server selects every request; select=false is not served'
        if tokens[-len(self.think.tail):] != self.think.tail:
            return 'This server serves thinking requests only: render the prompt with thinking on'
        budget = extra.get('expert_budget')
        if budget is not None and budget != self.think_budget:
            return f'Thinking mode serves {self.think_budget}-expert phase masks; got budget {budget}'
        if self.think_selector is not None and len(tokens) + len(self.spacer.spacer) > self.model_runner.max_model_len:
            return 'Prompt plus selector spacer exceeds max_model_len'
        return None

    def _static(self, state, phase):
        """A thinking request's static mask for a phase (None for a selected phase)."""
        return self.think.think if phase == 'think' else self.think.answer

    def _think_select(self, state, request_id):
        """End of a thinking request's prefill: both phase masks (the think mask only, for a post-think answer read);
        the think mask becomes the decode set."""
        row = {'request_id': request_id, 'mode': 'think', 'engine': 'bank', 'budget': self.think_budget}
        features = None
        if state['spacer']:
            try:
                features = self.taps.collect()
            finally:
                self.taps.release()
            row['features_sha256'] = hashlib.sha256(features.tobytes()).hexdigest()
        masks = {}
        post = self.post_think_spacer is not None
        for phase, selector, static in (('think', self.think_selector, self._static(state, 'think')),
                                        ('answer', self.answer_selector, self._static(state, 'answer'))):
            if phase == 'answer' and post:
                masks[phase] = None                  # selected at `</think>` from the probe (_post_think_select)
                row['answer_source'] = 'post_think'
                continue
            if selector is None:
                masks[phase] = static
                row[f'{phase}_source'] = 'static'
                continue
            selection = selector.select(request_id, state['tokens'], features, self.think_budget)
            masks[phase] = selection.keep
            row.update({f'{phase}_source': 'selected', f'{phase}_keep_sha256': selection.keep_sha256,
                        f'{phase}_selection_seconds': selection.seconds})
        state['answer'] = masks['answer']
        # The static think cut: the rescue rule's in-cut set for the whole think phase, refreshes or not.
        state['static_keep'] = masks['think'] if self.think_selector is not None else None
        if self.midrun is not None:
            # Midrun: the static think field the refreshes blend with; its cut must be the think mask just selected.
            static = self.think_selector.fused(features)
            if not np.array_equal(self.think_selector.cut(static, self.think_budget), masks['think']):
                raise RuntimeError('The static think field does not cut to the selected think mask')
            state['midrun'] = {'static': static, 'static_keep': masks['think'], 'segment': 0,
                               'next': len(state['tokens']) + self.config.midrun_every}
            if self.config.midrun_at_prefill:
                # The first block is a midrun selection too, from the same prefill spacer read.
                w = self.config.midrun_weight
                keep = self.midrun.cut((1 - w) * static + w * self.midrun.fused(features), self.think_budget)
                row.update(think_source='midrun_prefill', think_keep_sha256=array_digest(keep), midrun_weight=w,
                           midrun_prefill_moved_from_static=int((keep & ~masks['think']).sum()))
                masks['think'] = keep
        if not post:
            row['answer_only_experts'] = int((masks['answer'] & ~masks['think']).sum())
        if self.engine.predicted is not None and self.think_selector is not None:
            row['early_recall'] = float((self.engine.predicted & masks['think']).sum() / masks['think'].sum())
        if not self._decode_set(state, request_id, masks['think'], row):
            return
        row['decode_loaded_after_prefill'] = state['decode_loaded']
        if post:
            state['staging'] = None
            self.pending_updates.append(phase_update(request_id, 'think', len(state['tokens']), masks['think']))
            self.selection_log.write(json.dumps({**row, 'selection_applied_to_decode': True}) + '\n')
            self.selection_log.flush()
            return
        # The answer experts the swap will load are known now: stage them in host RAM while the think decodes,
        # so the swap copies from RAM and never waits on NVMe.
        staged = np.flatnonzero(masks['answer'].reshape(-1) & (self.engine.bank.index < 0))
        row['answer_staged'] = len(staged)
        if self.stager is None:
            self.stager = ThreadPoolExecutor(1)
        state['staging'] = self.stager.submit(self.engine.loader.host.stage, staged)
        self.pending_updates.append(phase_update(request_id, 'think', len(state['tokens']), masks['think']))
        self.selection_log.write(json.dumps({**row, 'selection_applied_to_decode': True}) + '\n')
        self.selection_log.flush()

    def sample_tokens(self, grammar_output):
        if self.rows > 1:
            return self._sample_concurrent(grammar_output)
        output = Worker.sample_tokens(self, grammar_output)
        updates, self.pending_updates = self.pending_updates, []
        state, position, self.sample_position = self.request_state, self.sample_position, None
        step, self.decode_step = self.decode_step, None
        if self.profile is not None and self.profile['active'] is not None:
            self._profile_step()
        finish = lambda result: self._sampled(result, state, position, updates, step)
        # Async scheduling: the tokens reach the host when the engine collects this output, which is after the
        # next step was launched and before the one after it is.
        return DeferredOutput(output, finish) if isinstance(output, AsyncModelRunnerOutput) else finish(output)

    def _sample_concurrent(self, grammar_output):
        """Concurrent decode: every request of the step counted. Async scheduling: the
        tokens reach the host when the engine collects this output, after the next step was launched and before the
        one after it (as for one request), so each request's `</think>` is known before the row after the next."""
        infos, self.step_infos = self.step_infos or {}, None
        output = Worker.sample_tokens(self, grammar_output)
        updates, self.pending_updates = self.pending_updates, []
        def finish(result):
            if result is None:
                self.pending_updates[:0] = updates
                return None
            for rid, (state, position, step) in infos.items():
                self._sampled(result, state, position, [], step)
                if self.async_scheduling and rid in self.states and rid in result.req_ids:
                    self.known_outputs.setdefault(rid, []).extend(result.sampled_token_ids[result.req_ids.index(rid)])
            return attach_updates(result, updates)
        return DeferredOutput(output, finish) if isinstance(output, AsyncModelRunnerOutput) else finish(output)

    def _sampled(self, output, state, position, updates, step=None):
        """A step's sampled tokens on the host: note where `</think>` lands (the swap happens in execute_model,
        before the row after it), count a decode step's tokens for its phase, and attach the step's updates."""
        if output is None:
            self.pending_updates[:0] = updates
            return None
        sampled = (output.sampled_token_ids[output.req_ids.index(state['request_id'])]
                   if state is not None and state['request_id'] in output.req_ids else None)
        if step is not None and sampled is not None:
            phase, began = step
            totals = state['decode'].setdefault(phase, {'steps': 0, 'tokens': 0, 'first': began, 'last': None})
            totals['steps'] += 1
            totals['tokens'] += len(sampled)
            totals['last'] = time.perf_counter()
        if state is not None and state['think'] and state['phase'] == 'think' and state['close_position'] is None \
                and position is not None and sampled is not None:
            if THINK_END in sampled:
                state['close_position'] = position + sampled.index(THINK_END)
        return attach_updates(output, updates)

    @torch.inference_mode()
    def _answer_swap(self, state, start):
        """Before the row at `start` = q+1 runs: row q (input `</think>`) ran under the think mask."""
        rid = state['request_id']
        row = {'request_id': rid, 'kind': 'think_swap', 'close_position': start - 1, 'answer_first_position': start}
        began = time.perf_counter()
        host = self.engine.loader.host
        if state['staging'] is not None:
            row['staged_from_nvme'] = state['staging'].result()
            row['staging_wait_seconds'] = time.perf_counter() - began
        tier = dict(host.stats)
        before = self.engine.loader.bytes
        if self.post_think_spacer is not None:
            # The probe runs under the think mask still installed, then its read selects the answer mask.
            row['answer_read'] = 'post_think'
            row['probe_mask_keep_sha256'] = hashlib.sha256(state['keep'].tobytes()).hexdigest()
            self._post_think_select(state, rid, start, row)
        if not self._decode_set(state, rid, state['answer'], row):
            state['phase'] = 'rejected'
            return
        loaded = state['decode_loaded']
        torch.cuda.synchronize()
        host.release_held()
        state['staging'] = None
        loaded_bytes = self.engine.loader.bytes - before
        state['decode_loaded'] = loaded
        row.update(answer_loaded=state['decode_loaded'], answer_loaded_bytes=loaded_bytes,
                   record_bytes=host.stride, swap_seconds=time.perf_counter() - began,
                   swap_host_tier={k: host.stats[k] - tier[k] for k in ('hits', 'misses', 'nvme_bytes', 'evictions')})
        if loaded_bytes != state['decode_loaded'] * self.engine.loader.host.stride:
            raise RuntimeError(f'Answer swap moved {loaded_bytes} bytes for {state["decode_loaded"]} experts')
        think = state['decode'].get('think')
        if think is not None:     # decode_phases subtracts it from think's clock if the last sample lands after it
            think.setdefault('pauses', []).append((began, time.perf_counter()))
        state['phase'] = 'answer'
        self.pending_updates.append(phase_update(rid, 'answer', start, state['answer']))
        self.selection_log.write(json.dumps(row) + '\n')
        self.selection_log.flush()

    def _post_think_select(self, state, rid, start, row):
        """Probe at start = q+1 under the think mask, then the post-think answer selection. A probe that cannot run
        or a read the selector rejects keeps the think mask for the answer and says so (answer_source)."""
        began = time.perf_counter()
        count = len(self.post_think_spacer)
        think = state['keep']
        state['answer'] = think
        if start + count > self.model_runner.max_model_len:
            row['answer_source'] = 'think: probe exceeds max_model_len'
            return
        features = self._probe_forward(rid, start)
        row['probe_seconds'] = time.perf_counter() - began
        row['probe_features_sha256'] = hashlib.sha256(features.tobytes()).hexdigest()
        try:
            selection = self.answer_selector.select(rid, state['tokens'], features, self.think_budget)
        except SelectionRejected as error:
            row['answer_source'] = f'think: post-think read rejected: {error}'
            return
        row.update(answer_source='post_think', answer_keep_sha256=selection.keep_sha256,
                   answer_selection_seconds=selection.seconds,
                   answer_moved_from_think=int((selection.keep & ~think).sum()))
        state['answer'] = selection.keep

    def _set_rescue_gain(self, state, request_id):
        """The rescue gain: think steps score w x (1 inside the request's static think cut, gout outside);
        answer steps w x (x / ax), the weight rule at ax. Written in place into the request's keep-slot row (fixed
        address, stream-ordered behind the steps already enqueued) when the slot's request or phase changes."""
        rescue = self.engine.rescue
        if rescue is None or rescue.get('gain') != 'static_think':
            return
        slot = None if self.rows == 1 else self.slots[request_id]
        phase = 'think' if state['phase'] == 'think' and state.get('static_keep') is not None else 'answer'
        key = (id(state), phase)
        if self.rescue_gain_keys.get(slot) == key:
            return
        if phase == 'think':
            gain = np.where(state['static_keep'], 1.0, rescue['gout']).astype(np.float32)
        else:
            gain = np.full((self.engine.layers, self.engine.experts), rescue['x'] / rescue['ax'], np.float32)
        self.engine.cpu_experts.set_rescue_gain(gain, self.engine.device, slot)
        self.rescue_gain_keys[slot] = key
        self.rescue_gain_writes = getattr(self, 'rescue_gain_writes', 0) + 1

    def _set_rescue(self, on):
        """Rescue rule on or off for the decode steps enqueued from here (stream-ordered device switch)."""
        if getattr(self, 'rescue_on', None) != on:
            self.engine.cpu_experts.set_rescue(on)
            self.rescue_on = on

    def _probe_forward(self, rid, start, probe=None):
        """The post-think probe: one forward of the post-think spacer at start..start+7 under the live
        mask, taps on its 8 rows, nothing sampled. Afterwards the token buffer and the recurrent state are put
        back as they were after the `</think>` row. Attention KV written at those positions is never read:
        decode overwrites it position by position."""
        runner = self.model_runner
        batch = runner.input_batch
        # Concurrent decode: the probe's batch update drops the other requests from the persistent batch and condenses
        # it, which can move this request's row (or add it back, if an earlier probe this step dropped it). Its spacer
        # tokens are placed, and its own tokens saved and put back, at the row it holds after that update.
        index = None if self.rows > 1 else batch.req_id_to_index[rid]
        probe = self.post_think_spacer if probe is None else probe
        count = len(probe)
        if self.engine.rescue is not None:
            self._set_rescue(False)        # a read runs under the live mask as served without rescue
        saved = batch.token_ids_cpu[index, start:start + count].copy() if index is not None else None
        slots = self.spacer.snapshot(runner, rid)
        self.taps.begin(rid, start + count, start)
        self.taps.prepare_step(start, count)
        synthetic = replace(self.last_scheduler_output, scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData(req_ids=[rid], resumed_req_ids=set(), new_token_ids=[],
                all_token_ids={}, new_block_ids=[None], num_computed_tokens=[start],
                num_output_tokens=[len(runner.requests[rid].output_token_ids)]),
            num_scheduled_tokens={rid: count}, total_num_scheduled_tokens=count, scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={}, finished_req_ids=set(), free_encoder_mm_hashes=[], preempted_req_ids=None,
            new_block_ids_to_zero=None, kv_cache_block_copies=None)
        # Async scheduling: the token sampled at q+1 lives only on the GPU (prev_sampled_token_ids) and vLLM
        # would scatter it into the probe's input ids. Hide it for the probe, so the inputs come from the CPU
        # buffer; the real step at q+1 then reads it as before.
        previous = batch.prev_sampled_token_ids, batch.prev_req_id_to_index
        batch.prev_sampled_token_ids = batch.prev_req_id_to_index = None
        placed, native_update = {}, runner._update_states
        if index is not None:
            batch.token_ids_cpu[index, start:start + count] = probe
        else:
            def update_states(scheduler_output):
                result = native_update(scheduler_output)
                placed['index'] = row = batch.req_id_to_index[rid]
                placed['saved'] = batch.token_ids_cpu[row, start:start + count].copy()
                batch.token_ids_cpu[row, start:start + count] = probe
                return result
            runner._update_states = update_states
        try:
            with self._eager():
                self._run_probe(synthetic)
            if runner.execute_model_state is None:
                raise RuntimeError('The post-think probe did not run a forward')
        finally:
            runner.execute_model_state = None          # nothing is sampled from the probe
            runner.kv_connector_output = None
            runner._update_states = native_update
            if index is not None:
                batch.token_ids_cpu[index, start:start + count] = saved
            elif placed:
                batch.token_ids_cpu[placed['index'], start:start + count] = placed['saved']
            batch.prev_sampled_token_ids, batch.prev_req_id_to_index = previous
            self.spacer.restore(slots)
        try:
            return self.taps.collect()
        finally:
            self.taps.release()

    def _run_probe(self, synthetic):
        return self.model_runner.execute_model(synthetic)

    @torch.inference_mode()
    def execute_model(self, scheduler_output):
        if self.rows > 1:
            return self._execute_concurrent(scheduler_output)
        return self._execute_one(scheduler_output)

    def _execute_concurrent(self, scheduler_output):
        """Concurrent decode: a prefill chunk runs alone through the single-request
        path with its request's state; a decode step runs each request's phase events, then one forward of all rows."""
        self.last_scheduler_output = scheduler_output
        self.decode_rows = None
        batch = self.model_runner.input_batch
        self.step_prev = (batch.prev_sampled_token_ids, dict(batch.prev_req_id_to_index or {})) \
            if self.async_scheduling else None
        for rid in scheduler_output.finished_req_ids:
            if rid in self.states:
                self._release_concurrent(rid)
            self.known_outputs.pop(rid, None)
        if not scheduler_output.total_num_scheduled_tokens:
            return self._attach(self._run_model(scheduler_output))
        runner = self.model_runner
        steps = {}
        for rid, count in scheduler_output.num_scheduled_tokens.items():
            new = next((r for r in scheduler_output.scheduled_new_reqs if r.req_id == rid), None)
            if new is not None:
                start, prompt = new.num_computed_tokens, len(new.prompt_token_ids)
            else:
                cached = scheduler_output.scheduled_cached_reqs
                start = cached.num_computed_tokens[cached.req_ids.index(rid)]
                prompt = len(runner.requests[rid].prompt_token_ids)
            steps[rid] = (start, prompt, count)
        prefilling = [rid for rid, (start, prompt, _) in steps.items() if start < prompt]
        if prefilling:
            if len(steps) != 1:
                raise RuntimeError(f'Concurrent decode runs a prefill chunk alone; this step mixes {sorted(steps)}')
            rid = prefilling[0]
            self.decode_rows = None
            if rid not in self.slots:
                self.slots[rid] = self.free_slots.pop(0)
            self.request_state = self.states.get(rid)
            try:
                result = self._execute_one(scheduler_output)
            finally:
                if self.request_state is not None:
                    self.states[rid] = self.request_state
                self.request_state = None
            if rid in self.states:
                self.step_infos = {rid: (self.states[rid], self.sample_position, None)}
            self.sample_position = None
            return result
        infos, phases = {}, set()
        for rid, (start, prompt, count) in steps.items():
            state = self.states.get(rid)
            if state is None:
                raise RuntimeError(f'{rid} decodes with no request state')
            if count != 1:
                raise RuntimeError(f'Concurrent decode steps carry one row per request; {rid} has {count}')
            self.request_state = state
            try:
                self._before_decode(state, rid, start)
            finally:
                self.request_state = None
            phases.add(state['phase'] or 'answer')
            self._set_rescue_gain(state, rid)
            infos[rid] = (state, start + 1, (state['phase'] or 'decode', time.perf_counter()))
        if self.engine.rescue is not None:
            self._set_rescue(phases <= set(self.engine.rescue['phases']))
        self.step_infos = infos
        self.decode_rows = {rid: start for rid, (start, _, _) in steps.items()}
        if self.engine.decode_stats is None:
            # No decode set is live: every row is a refused request's step that async scheduling launched before the
            # refusal reached the scheduler. Its output is dropped, and it runs on the experts already resident
            # (_before_decode), so nothing loads and there is no paged-decode bookkeeping to keep.
            return self._attach(self._run_model(scheduler_output))
        self.engine.between_steps()
        result = self._run_model(scheduler_output)
        self.engine.after_decode_launch(self.row_requests)
        if self.routing_log is not None:
            for rid, (start, _, _) in steps.items():
                self.states[rid].setdefault('routing_positions', []).append(start)
        return self._attach(result)

    def _before_decode(self, state, request_id, start):
        """A request's events before its decode row runs: the answer swap after `</think>` and a due midrun refresh;
        returns its keep-set (rejected: the resident experts, so nothing loads)."""
        close = state['close_position']
        if state['think'] and state['phase'] == 'think' and close is not None:
            if start == close + 1:
                self._answer_swap(state, start)
            elif start > close + 1:
                raise RuntimeError(f'Row {start} scheduled before the answer swap at {close + 1}')
        midrun = state.get('midrun')
        if midrun is not None and state['phase'] == 'think' and state['close_position'] is None \
                and start >= midrun['next']:
            if start != midrun['next']:
                raise RuntimeError(f"Midrun refresh due at {midrun['next']}, decode step at {start}")
            self._midrun_refresh(state, start)
        keep = state['keep']
        if keep is None and state['rejected']:
            # Async scheduling: a step scheduled before the rejection reached the scheduler. Its output is
            # dropped; run it on experts already resident so nothing loads.
            keep = state['keep'] = self.engine.bank.index.reshape(self.engine.layers, self.engine.experts) >= 0
            self._publish(keep, request_id)
        if keep is None:
            raise ValueError('Decode scheduled before the decode set was prepared')
        return keep

    def _execute_one(self, scheduler_output):
        self.last_scheduler_output = scheduler_output
        if self.request_state is not None and self.request_state['request_id'] in scheduler_output.finished_req_ids:
            self._release()
        step = request_prefill_step(scheduler_output, self.model_runner.requests)
        if step is None:
            return self._attach(self._run_model(scheduler_output))
        new = next((r for r in scheduler_output.scheduled_new_reqs if r.req_id == step.request_id), None)
        request = new if new is not None else self.model_runner.requests[step.request_id]
        extra = request.sampling_params.extra_args or {}
        rejection = None
        if self.request_state is None:
            # A request's first step starts at position 0; later chunks start where the last ended.
            if step.start != 0:
                raise ValueError('Cannot begin bank serving from a cached prompt prefix')
            tokens = tuple(request.prompt_token_ids)
            rejection = self._admit(extra, tokens)
            think = rejection is None
            self.request_state = {'request_id': step.request_id, 'tokens': tokens,
                                  'select_argument': extra.get('select', True), 'budget': extra.get('expert_budget'),
                                  'keep': None, 'think': think, 'phase': 'think' if think else None,
                                  'close_position': None, 'answer': None, 'rejected': False, 'first': step.start,
                                  'decode': {}}
            state = self.request_state
            # Early R* predicts for every served request (the think bundle selects its think mask).
            state['predict'] = think and self.think_selector is not None
            state['spacer'] = state['predict']
            state['raw'] = think and self.think_read_raw
            if state['spacer']:
                self.taps.begin(step.request_id, len(tokens) + (0 if state['raw'] else len(self.spacer.spacer)), step.start)
        state = self.request_state
        self.sample_position = step.start + step.count    # the step's first sampled token
        if state['request_id'] != step.request_id or state['select_argument'] != extra.get('select', True):
            raise ValueError('Request state or selection mode changed while active')
        prefill = step.start < step.prompt
        final = prefill and step.start + step.count == step.prompt
        spacer_rows = len(self.spacer.spacer) if final and state['spacer'] else 0
        if prefill:
            if state['spacer']:
                self.taps.prepare_step(step.start, step.count + (0 if state['raw'] else spacer_rows))
            else:
                self.taps.disable()
            if state['think']:
                known = None if state['predict'] else self._static(state, 'think')
                budget = self.think_budget
            else:
                # A refused request runs its one step on the experts already in the bank, then ends with its error.
                known = self.engine.bank.index.reshape(self.engine.layers, self.engine.experts) >= 0
                budget = int(known.sum())
            self.engine.begin_prefill(budget, known=known, first_chunk=step.start == state['first'])
            features = self._features if spacer_rows and state['predict'] else None
            for module in self.experts.values():
                object.__setattr__(module, 'apareto_features', features)
            self.engine.features = features
            if spacer_rows:
                self.spacer.arm(self.model_runner, step.request_id, step.count)
                self.spacer_rows = spacer_rows
            full = replace(step, full=True, count=step.count + spacer_rows)
            try:
                with self._eager(), step_context(full):
                    result = self._run_model(scheduler_output)
            except BaseException:
                if spacer_rows:
                    self.spacer.active = None
                raise
            finally:
                self.spacer_rows = 0
                stats = self.engine.end_prefill()
                self.prefill_log.write(json.dumps({'request_id': step.request_id, 'start_position': step.start,
                    'tokens': full.count, 'selected': state['think'], **(stats or {})}) + '\n')
                self.prefill_log.flush()
            if spacer_rows:
                self.spacer.disarm()
        else:
            midrun = state.get('midrun')
            if midrun is not None and state['phase'] == 'think' and state['close_position'] is None \
                    and step.start >= midrun['next'] and step.count != 1:
                raise RuntimeError(f"Midrun refresh due at {midrun['next']}, decode step at {step.start}+{step.count}")
            keep = self._before_decode(state, step.request_id, step.start)
            if self.engine.rescue is not None:     # rescue in the decode phases named
                self._set_rescue((state['phase'] or 'answer') in self.engine.rescue['phases'])
                self._set_rescue_gain(state, step.request_id)
            self.decode_step = (state['phase'] or 'decode', time.perf_counter())
            if self.profile is not None:
                self._profile_begin(step.start)
            if self.engine.paged and self.engine.decode_stats is not None:
                self.engine.between_steps()
                result = self._run_model(scheduler_output)
                self.engine.after_decode_launch()
                if self.routing_log is not None:
                    state.setdefault('routing_positions', []).append(step.start)
            else:
                if np.any(self.engine.bank.index.reshape(keep.shape)[keep] < 0):
                    raise RuntimeError('A decode-set expert is not resident')
                bytes_before = self.engine.loader.bytes
                result = self._run_model(scheduler_output)
                if self.engine.loader.bytes != bytes_before:
                    raise RuntimeError('Decode loaded experts')
        if rejection is not None:
            self._reject(step.request_id, rejection)
        elif final and state['think']:
            try:
                self._think_select(state, step.request_id)
            except SelectionRejected as error:
                self._reject(step.request_id, str(error), {'request_id': step.request_id, 'mode': 'think'})
        return self._attach(result)
