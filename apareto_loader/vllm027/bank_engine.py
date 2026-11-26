"""The expert bank on the GPU and its prefill / decode policy.

What it holds:
- bank tensors [slots, ...] for every store field (any slot holds any expert);
- per layer: a [256] slot table (expert -> slot, -1 if absent), alphas and input-quant scales, read by the
  patched MoE kernel (moe_slots.py), so a layer's call costs the 256-expert amount wherever experts sit;
- the CPU master bookkeeping (apareto_loader.bank.ExpertBank).

Policy:
- Prefill uses all experts. Before layer L's MoE, every expert of layer L not in the bank is loaded from
  the host RAM tier; layer L+1 is loaded while layer L computes.
- Nothing is evicted before early R* says what to evict. Until the prediction, the previous request's decode
  set D stays resident and protected as a provisional prediction (eviction rank: in D = wanted). With
  pre_prediction_keep='layers' (default) the first layer_cap layers also stay protected until the prediction,
  so a bank of D + those layers (4,893 at budget 4,301) evicts nothing before it; with 'decode' only D is
  protected and a computed layer's other experts may be evicted (smaller bank, reloads after the prediction).
- Early R* reads only layers < layer_cap (hl4 is the residual entering layer 4), so it runs at layer
  layer_cap's entry (a hook after the tap capture), before that layer's experts are requested.
- After the prediction: protected = the prediction, the layer computing and the layer loading. Eviction order:
  layers already computed in this pass first (horizon), then worst early-R* rank, then least recently used
  (stamped on use). A request whose decode set is known up front (select=false) uses it from layer 0. "Free" is
  lazy: an evictable expert stays resident until its slot is needed, so leftovers serve the next request.
- After full R* selects, only the selected experts still missing are loaded, and decode runs on them.
Safety: before any load, the loader stream waits for all compute already enqueued, so a slot is never
overwritten while a queued kernel may still read its old expert. Index entries become visible to compute
only through the loader's completion event.

Paged decode (gpu_experts below the bank minimum). The bank holds G < |D|
experts and the MoE kernel reads every other expert of D from its record in pinned host memory (host_table,
moe_slots.py), inside the decode CUDA graph. Between steps the host reads the step's routed ids (copied out
after the graph), stamps recency per routed expert and admits that step's misses, evicting the least recently
routed. Ordering: A1 a slot is published on the loader stream after its copy; A2 a victim is
unpublished on the compute stream before the next step is launched, and its slot is overwritten only after
the loader stream has waited for that; A3 in-flight slots are never victims (ExpertBank.allocate). Prefill is
unchanged except that the prediction is not protected (it may not fit): rank + horizon still choose victims.
"""
import collections
import os
import sys
import time

import numpy as np
import torch

from apareto_loader.bank import ExpertBank
from .host_tier import _device_ids

# Measurement only (APARETO_HOST_PHASES=1): per-request host seconds of each between-step phase, added to
# decode_stats as phase_<name>. Off: no timer calls.
_HOST_PHASES = bool(os.environ.get('APARETO_HOST_PHASES'))

_EARLY_FEATURES = None


def early_selector(path):
    """The early prediction for prefill residency: the think bundle read on its shallow taps (LateShallowSelector)."""
    return LateShallowSelector(path)


SHALLOW_TAPS = ('hl2', 'hl4', 'kvl3')


class LateShallowSelector:
    """The early prediction made by the think bundle on its shallow taps alone: the query and the item embeddings
    keep only the hl2 / hl4 / kvl3 blocks of the bundle's own embedding (each block is unit-normalised there),
    renormalised; scoring, aggregation and cut are the bundle's. Those taps all exist at
    layer 4's entry, so the prediction runs there, during prefill. It decides only which experts stay in the bank
    while prefill runs (speed); the decode keep-set always comes from the full read at the end of prefill."""

    def __init__(self, bundle, layer_cap=4):
        from apareto_loader.selector import load_reference
        s = self.late = load_reference(bundle)    # the think network the worker selects with, not a second copy
        pick = [s.taps.index(t) for t in SHALLOW_TAPS]
        self.taps, self.rows, self.layer_cap = list(SHALLOW_TAPS), s.rows, layer_cap
        self.mu, self.sd, self.P = ([getattr(s, a)[i] for i in pick] for a in ('mu', 'sd', 'P'))
        self.full_slices = [s.slices[i] for i in pick]
        self.slices, o = [], 0
        for i in pick:
            self.slices.append((o, o + s.rows * s.dims[i])); o += s.rows * s.dims[i]
        self.n_features, self.n_full = o, s.n_features
        cols = np.cumsum([0] + [p.shape[1] for p in s.P])
        self.item_emb = self._nz(np.concatenate([s.item_emb[:, cols[i]:cols[i + 1]] for i in pick], 1))
        self.k = s.k

    @staticmethod
    def _nz(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

    def accept_features(self, x):
        x = np.atleast_2d(np.asarray(x, np.float64))
        if x.shape[1] == self.n_features:
            return x
        if x.shape[1] == self.n_full:
            return np.concatenate([x[:, lo:hi] for lo, hi in self.full_slices], 1)
        raise ValueError(f"expected {self.n_features} or {self.n_full} features, got {x.shape[1]}")

    def embed(self, x):
        x = self.accept_features(x)
        return self._nz(np.concatenate([self._nz(((x[:, lo:hi] - mu) / sd) @ P)
                                        for (lo, hi), mu, sd, P in zip(self.slices, self.mu, self.sd, self.P)], 1))

    def retrieve(self, emb):
        sim = emb @ self.item_emb.T
        idx = np.argsort(-sim, axis=1)[:, : self.k]
        return idx, np.take_along_axis(sim, idx, axis=1)

    def fuse(self, idx, sim):
        return self.late.fuse(idx, sim)

    def cut(self, fused, budget):
        return self.late.cut(fused, budget)

    def keep_set(self, features, budget):
        return self.cut(self.fuse(*self.retrieve(self.embed(features))), budget)


class BankEngine(torch.nn.Module):
    paged = False   # set by enable_paged

    cpu_experts = None       # CPU-served misses (cpu_experts.py), paged decode only
    rescue = None            # the CPU rescue rule's spec, with cpu_experts
    rescue_w13 = None        # [layers, experts] host address of every record's w13 field (rescue lookups)
    admit = True             # paged decode: admit a step's misses into the bank between steps
    cpu_stats_start = None   # CPU-path counters at the start of the request's decode

    def __init__(self, capacity, layout, input_global_scales, max_tokens, device, layers=40, experts=256,
                 extra_slots=0, rows=1):
        """extra_slots: slots after `capacity` that the bookkeeping does not own.
        rows: requests decoding at once (max_num_seqs): each has a keep row, and a
        decode step's routed ids and routing log carry one row per request."""
        super().__init__()
        self.layers, self.experts, self.capacity, self.rows = layers, experts, capacity, rows
        dtypes = {'uint8': torch.uint8, 'float8_e4m3fn': torch.float8_e4m3fn, 'float32': torch.float32}
        self.fields = {}
        for name, field in layout.items():
            total = capacity + extra_slots
            shape = (total, *field['shape']) if field['shape'] else (total,)
            tensor = torch.zeros(shape, dtype=dtypes[field['dtype']], device=device)
            if name.startswith('s'):
                tensor.view(torch.uint8).fill_(0x38)   # FP8 1.0: unreachable slots stay finite for profiling
            if name.startswith('global'):
                tensor.fill_(1.0)
            self.register_buffer(f'bank_{name}', tensor)
            self.fields[name] = tensor
        scales = torch.as_tensor(input_global_scales, dtype=torch.float32, device=device)
        if scales.shape != (layers, 2) or not bool((torch.isfinite(scales) & (scales > 0)).all()):
            raise ValueError('Expected per-layer positive finite input-global scales [layers, 2]')
        # Same reciprocal order as the native preparation.
        self.register_buffer('layer_dequant', 1.0 / scales)
        self.register_buffer('layer_quant', 1.0 / self.layer_dequant)
        self.register_buffer('table', torch.full((layers, experts), -1, dtype=torch.int32, device=device))
        # vLLM's memory profiling runs a dense forward where any expert id can appear before anything is
        # loaded; it reads this all-valid table (slot contents are placeholders) instead of `table`.
        self.register_buffer('profile_table', (torch.arange(layers * experts, device=device) % capacity)
                             .to(torch.int32).view(layers, experts))
        self.register_buffer('alpha13', torch.ones((layers, experts), dtype=torch.float32, device=device))
        self.register_buffer('alpha2', torch.ones((layers, experts), dtype=torch.float32, device=device))
        self.register_buffer('a13', self.layer_quant[:, 0:1].expand(layers, experts).contiguous())
        self.register_buffer('a2', self.layer_quant[:, 1:2].expand(layers, experts).contiguous())
        from flashinfer.fused_moe.core import cutlass_fused_moe_workspace_size
        hidden = layout['w2']['shape'][0]
        middle = layout['w2']['shape'][1] * 2
        size = cutlass_fused_moe_workspace_size(max_tokens, hidden, middle, experts, 8, x_dtype=torch.uint8,
                                                weight_dtype=torch.int64, output_dtype=torch.bfloat16,
                                                device=torch.device(device))   # int64: packed FP4 view
        self.register_buffer('workspace', torch.empty(size, dtype=torch.uint8, device=device))
        self.bank = ExpertBank(capacity, layers, experts)
        self.device = device
        self.pre_prediction_keep = 'layers'   # or 'decode'
        self.decode_set = None    # D: the last decode set (warm start: the warm set)
        self.provisional = False  # True while D stands in for the prediction
        self.features = None      # features() of the current step, bound by the worker
        self.moe = None          # patched FlashInfer module, bound at load
        self.loader = None       # host tier -> bank, bound at load
        self.predictor = None    # early R*
        self.stats = None
        self.premask = None      # routing log (enable_premask): host rows of pre-mask top-8 ids and weights
        # Paged decode: per layer, host addresses of each expert's fc1 w / fc1 sf / fc2 w / fc2 sf (0 outside D),
        # and the step's routed ids (masked; -1 dropped), written inside the decode graph.
        self.register_buffer('host_table', torch.zeros((layers, 4, experts), dtype=torch.int64, device=device))
        self.register_buffer('routed', torch.full((layers, rows * 8), -1, dtype=torch.int32, device=device))
        # Each token row's keep slot (its request's row of every layer's apareto_keep), written before each forward
        # by the worker; -1 marks a padding row (CUDA graph batch padding): nothing allowed, nothing rescued.
        self.register_buffer('row_slot', torch.zeros(max_tokens, dtype=torch.int32, device=device))

    def enable_paged(self, layout):
        """Bank smaller than the decode set. Called at load, before any decode."""
        self.paged = True
        self.host_layout = layout
        self.host_offsets = np.array([layout[name]['offset'] for name in ('w13', 's13', 'w2', 's2')], dtype=np.int64)
        self.global_offsets = (layout['global13']['offset'], layout['global2']['offset'])
        # Per step: masked routed ids [layers, rows * 8], in contiguous pinned buffers (a copy into a strided view of
        # pinned memory is not asynchronous).
        self.ring = [torch.empty((self.layers, self.rows * 8), dtype=torch.int32).pin_memory() for _ in range(4)]
        self.ring_rows = [None] * len(self.ring)    # the request of each row of a ring entry's step (routing log)
        self.ring_next = 0
        self.pending_routed = collections.deque()   # (ring index, event) of routed-id copies not yet read
        self.copies = []                            # (event, [(expert, slot)]) admissions in flight
        self.inflight = set()
        self.decode_stats = None
        self.decode_stats_history = []   # earlier phases' counters of the current request

    def _phase(self, name, since):
        """APARETO_HOST_PHASES: add the seconds since `since` to phase_<name>; returns now."""
        now = time.perf_counter()
        key = f'phase_{name}'
        self.decode_stats[key] = self.decode_stats.get(key, 0.0) + now - since
        return now

    # -- binding (after weights are loaded) ------------------------------------------------------------
    def bind(self, loader, predictor, moe):
        from flashinfer.utils import device_support_pdl
        self.loader, self.predictor, self.moe = loader, predictor, moe
        self.loader_stream = loader.stream
        if _HOST_PHASES:
            loader.phase = lambda name, since: self._phase(name, since) if self.decode_stats is not None \
                else time.perf_counter()
        self.enable_pdl = device_support_pdl(torch.device(self.device))

    # -- compute ---------------------------------------------------------------------------------------
    def run(self, layer, hidden, ids, weights, output=None, table=None, host=None):
        """host: optional [layers, 4, 256] int64 host addresses (fc1 w, fc1 sf, fc2 w, fc2 sf) for experts whose slot
        is -1; the kernel then reads them from pinned host memory (moe_slots.py)."""
        from vllm import _custom_ops as ops
        packed, input_sf = ops.scaled_fp4_quant(hidden, self.layer_quant[layer, 0:1])
        if output is None:
            output = torch.empty_like(hidden)
        f = self.fields
        quant = [self.a13[layer], f['s13'].view(torch.int32), self.alpha13[layer],
                 self.a2[layer], f['s2'].view(torch.int32), self.alpha2[layer],
                 (self.table if table is None else table)[layer]]
        if host is not None:
            quant.append(host[layer])
        from flashinfer.fused_moe.core import ActivationType
        self.moe.cutlass_fused_moe(output, packed, ids, weights, f['w13'].view(torch.int64), None,
            f['w2'].view(torch.int64), None, torch.bfloat16, quant, input_sf, None, None, None, True,
            1, 0, 1, 0, 1, 0, use_packed_weights=False, enable_alltoall=False, use_deepseek_fp8_block_scale=False,
            use_w4_group_scaling=False, use_mxfp8_act_scaling=False, min_latency_mode=False,
            tune_max_num_tokens=8192, enable_pdl=self.enable_pdl, activation_type=ActivationType.Swiglu,
            use_fused_finalize=True, profile_ids=None, workspace_buffer=self.workspace)
        return output

    # -- loading ---------------------------------------------------------------------------------------
    def ensure(self, layer, experts=None):
        """Load the layer's experts (all, or the given ids within the layer) that are not in the bank.

        Returns the completion event (or None if nothing was missing). Raises BankFull if the bank cannot
        hold them next to everything protected.
        """
        base = layer * self.experts
        wanted = [base + int(e) for e in (range(self.experts) if experts is None else experts)]
        # The wanted experts already resident must not be evicted to make room for the missing ones:
        # they are not reloaded, so their table entry would go to -1 under a kernel that reads them.
        self.bank.protect([x for x in wanted if self.bank.index[x] >= 0])
        missing = sorted(set(wanted) & set(self.bank.missing(layer)))
        if not missing:
            return None
        before = time.perf_counter()
        allocations = self.bank.allocate(missing)
        evicted = [victim for _, _, victim in allocations if victim >= 0]
        slots = [slot for _, slot, _ in allocations]
        done = self._transfer(layer, base, missing, slots, evicted)
        for expert, slot, _ in allocations:
            self.bank.complete(expert, slot)
        if np.any(self.bank.index[wanted] < 0):
            raise RuntimeError(f'Layer {layer}: a wanted expert is absent after ensure')
        if self.stats is not None:
            self.stats['loaded'] += len(missing)
            self.stats['evicted'] += len(evicted)
            if self.provisional:
                self.stats['evicted_before_prediction'] += len(evicted)
                self.stats['evicted_before_prediction_from_D'] += int(sum(
                    bool(self.predicted.reshape(-1)[x]) for x in evicted))
                self.stats['evicted_before_prediction_early_layers'] += sum(
                    x // self.experts < self.predictor.layer_cap for x in evicted)
            self.stats['issue_seconds'] += time.perf_counter() - before
        return done

    def _transfer(self, layer, base, missing, slots, evicted):
        """GPU side of a load: invalidate evicted table entries, copy the records, publish alphas and the slot
        table on the loader stream. Returns the completion event."""
        # Slots about to be overwritten may still be read by compute already enqueued.
        self.loader_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.loader_stream):
            if evicted:
                index = _device_ids(evicted, self.device)
                self.table.view(-1).index_fill_(0, index, -1)
        self.loader.load(layer, [x - base for x in missing], slots)
        with torch.cuda.stream(self.loader_stream):
            local = _device_ids([x - base for x in missing], self.device)
            slot_tensor = _device_ids(slots, self.device)
            # Reciprocal, then multiply: the same order as the native preparation.
            self.alpha13[layer].index_copy_(0, local, (1.0 / self.fields['global13'][slot_tensor]) * self.layer_dequant[layer, 0])
            self.alpha2[layer].index_copy_(0, local, (1.0 / self.fields['global2'][slot_tensor]) * self.layer_dequant[layer, 1])
            self.table[layer].index_copy_(0, local, slot_tensor.to(torch.int32))
            done = torch.cuda.Event()
            done.record(self.loader_stream)
        return done

    def _sync(self):
        torch.cuda.current_stream().synchronize()

    def _wait(self, event):
        torch.cuda.current_stream().wait_event(event)

    # -- prefill policy ----------------------------------------------------------------------------------
    def _set_prediction(self, mask, provisional):
        """Install a (provisional) prediction: protect its resident experts; rank its experts as wanted."""
        self.predicted = np.asarray(mask, dtype=bool).reshape(self.layers, self.experts)
        self.provisional = provisional
        self.bank.set_priority(np.where(self.predicted, 0.0, 1.0))
        if not self.paged:   # paged: the prediction may not fit; rank alone keeps it
            self.bank.protect([x for x in np.flatnonzero(self.predicted.reshape(-1)) if self.bank.index[x] >= 0])

    def begin_prefill(self, budget, known=None, first_chunk=True):
        """known: a decode set fixed in advance (no prediction needed). Otherwise the previous decode set D
        stands in until early R* predicts (the final chunk); a later chunk keeps whatever is current."""
        if self.paged:
            self._settle_admissions(wait=True)
            if self.premask is not None:
                # Concurrent decode: other requests' logged steps may still be in the ring; keep their rows.
                for index, event in self.pending_routed:
                    event.synchronize()
                    self._log_premask(index)
            self.pending_routed.clear()
        self.bank.unprotect_all()
        if first_chunk or known is not None:
            if known is not None:
                self._set_prediction(known, provisional=False)
            else:
                self._set_prediction(self.decode_set, provisional=True)
                if self.pre_prediction_keep == 'layers':
                    # Layers < layer_cap will all be needed before the prediction: their leftovers rank as wanted,
                    # so making room for them takes leftovers of later layers first.
                    rank = np.where(self.predicted, 0.0, 1.0)
                    rank[:self.predictor.layer_cap] = 0.0
                    self.bank.set_priority(rank)
        elif not self.paged:
            self.bank.protect([x for x in np.flatnonzero(self.predicted.reshape(-1)) if self.bank.index[x] >= 0])
        self.budget = budget
        self.issued = {}
        self.stats = {'loaded': 0, 'evicted': 0, 'evicted_before_prediction': 0,
                      'evicted_before_prediction_from_D': 0, 'evicted_before_prediction_early_layers': 0, 'issue_seconds': 0.0,
                      'predict_seconds': 0.0, 'prediction_at_layer_entry': False,
                      'pre_prediction_keep': self.pre_prediction_keep,
                      'resident_at_start': int((self.bank.index >= 0).sum()), 'started': time.perf_counter()}
        self.host_before = dict(self.loader.host.stats)

    def _layer_experts(self, layer):
        return range(layer * self.experts, (layer + 1) * self.experts)

    def _issue(self, layer):
        if layer not in self.issued:
            # Layers below `layer` have computed in this pass: their unprotected experts are evicted first.
            self.bank.horizon = layer
            self.issued[layer] = self.ensure(layer)
            # The layer's experts must survive until its MoE has run.
            self.bank.protect([x for x in self._layer_experts(layer) if self.bank.index[x] >= 0])

    def _prediction_pending(self, layer):
        return layer == self.predictor.layer_cap and self.provisional and self.features is not None

    def at_layer_entry(self, layer):
        """Hook at a layer's input norm, after the tap capture. At layer_cap every early-R* tap exists
        (hl{cap} is the residual entering it), so predict here, then request the layer's experts."""
        if self._prediction_pending(layer):
            self._predict(self.features)
            self.stats['prediction_at_layer_entry'] = True
            self._issue(layer)

    def before_moe(self, layer, features):
        """features(): the current tap buffer (host), or None when this step carries no read rows."""
        if self._prediction_pending(layer):   # the entry hook did not run: predict here instead
            self._predict(self.features)
        self._issue(layer)
        if self.issued[layer] is not None:
            self._wait(self.issued[layer])

    def after_moe(self, layer):
        # Used now: recency for the LRU tie-break means use, not load.
        self.bank.touch([x for x in self._layer_experts(layer) if self.bank.index[x] >= 0])
        self._settle(layer)
        # Layer layer_cap's experts are requested only after the prediction (at its entry).
        if layer + 1 < self.layers and not self._prediction_pending(layer + 1):
            self._issue(layer + 1)

    def _settle(self, layer):
        """After a layer computed: its experts outside the (provisional) prediction become evictable. They
        stay resident until their slot is needed. Before the prediction, with pre_prediction_keep='layers',
        the first layer_cap layers stay protected so the prediction can still keep them."""
        if self.provisional and self.pre_prediction_keep == 'layers' and layer < self.predictor.layer_cap:
            return
        # Paged: the prediction may not fit, so all of the layer becomes evictable; its rank still keeps it.
        experts = [x for x in self._layer_experts(layer)
                   if self.bank.index[x] >= 0 and (self.paged or not self.predicted[layer, x % self.experts])]
        if experts:
            self.bank.protected[self.bank.index[experts]] = False

    def _predict(self, features):
        before = time.perf_counter()
        self._sync()   # the taps are written by kernels on the compute stream
        x = np.asarray(features(), dtype=np.float64)
        selector = self.predictor
        early = selector.accept_features(x)
        if not np.isfinite(early).all():
            # NaN here means those rows were never captured -- TapCapture.begin() fills the buffer
            # with NaN exactly so an unwritten row is detectable. Name the tap and the rows, because
            # 'not finite' alone cannot distinguish a missing tap from a missing spacer row from an
            # overflow, and each has a different cause.
            detail = []
            for name, (lo, hi) in zip(selector.taps, selector.slices):
                block = early[0, lo:hi].reshape(selector.rows, -1)
                bad = np.flatnonzero(~np.isfinite(block).all(axis=1))
                detail.append(f'{name}: {len(bad)}/{selector.rows} rows nonfinite'
                              + (f' at {bad.tolist()}' if len(bad) else ''))
            whole = int(np.count_nonzero(~np.isfinite(np.asarray(x))))
            raise ValueError('Early R* taps are not finite at the prediction layer; '
                             f'{whole}/{np.asarray(x).size} values nonfinite in the whole buffer; '
                             + '; '.join(detail))
        fused = selector.fuse(*selector.retrieve(selector.embed(early)))[0]
        # The prediction replaces D: protection restarts from it plus the layers issued and not yet computed.
        computed = selector.layer_cap
        self.bank.unprotect_all()
        self.predicted = selector.cut(fused[None], self.budget)[0]
        self.provisional = False
        self.bank.set_priority(fused)
        resident = [x for x in np.flatnonzero(self.predicted.reshape(-1)) if self.bank.index[x] >= 0]
        if not self.paged:
            self.bank.protect(resident)
        for layer in self.issued:
            if layer >= computed:
                self.bank.protect([x for x in self._layer_experts(layer) if self.bank.index[x] >= 0])
        self.stats['predict_seconds'] = time.perf_counter() - before
        self.stats['predicted_resident_at_prediction'] = len(resident)

    def end_prefill(self):
        if self.predicted is not None:
            for layer in range(self.layers):
                self._settle(layer)
        self.bank.horizon = None
        stats, self.stats = self.stats, None
        if stats is not None:
            stats['seconds'] = time.perf_counter() - stats.pop('started')
            stats['host_tier'] = {k: v - self.host_before[k] for k, v in self.loader.host.stats.items()}
        return stats

    # -- decode --------------------------------------------------------------------------------------------
    def prepare_decode(self, keep):
        """Load the decode set's missing experts and protect the set. Returns (loaded, events)."""
        self.bank.unprotect_all()
        resident = [x for x in np.flatnonzero(keep.reshape(-1)) if self.bank.index[x] >= 0]
        self.bank.protect(resident)
        events, loaded, early_layers = [], 0, 0
        for layer in range(self.layers):
            wanted = np.flatnonzero(keep[layer])
            # Experts this call loads: the wanted ones not resident now. (Counting the layer's missing
            # experts before and after undercounts when the load evicts other experts of the same layer.)
            count = int(np.count_nonzero(self.bank.index[layer * self.experts + wanted] < 0))
            event = self.ensure(layer, wanted)
            if event is not None:
                events.append(event)
                loaded += count
                if self.predictor is not None and layer < self.predictor.layer_cap:
                    early_layers += count
                self.bank.protect([layer * self.experts + int(e) for e in wanted])
        self.decode_set = np.asarray(keep, dtype=bool).copy()   # D for the next request
        self.bank.touch(np.flatnonzero(self.decode_set.reshape(-1)))
        self.decode_loaded_before_layer_cap = early_layers
        return loaded, events

    # -- paged decode ---------------------------------------------------------------------------------------------
    def prepare_paged(self, keep):
        """Decode set D larger than the bank. Holds all of D in the host tier at fixed addresses, writes their
        host table entries and alphas, and preloads the top-G of D by this request's R* rank, most
        wanted stamped most recent. Returns (loaded, events) like prepare_decode."""
        keep = np.asarray(keep, dtype=bool).reshape(self.layers, self.experts)
        self._settle_admissions(wait=True)
        if self.premask is not None:
            # The routing log keeps the steps in flight; their admissions are dropped with the old set as before.
            for index, event in self.pending_routed:
                event.synchronize()
                self._log_premask(index)
        self.pending_routed.clear()
        records = np.flatnonzero(keep.reshape(-1))
        from_nvme = self._publish_host_table(keep, records)
        return self._paged_preload(keep, records, self.bank.priority, from_nvme)

    def _publish_host_table(self, keep, records):
        """Hold D's records in the host tier; write their host-table entries and alphas. Returns NVMe reads."""
        host = self.loader.host
        addresses, from_nvme = host.hold_decode(records)
        address = np.array([addresses[int(r)] for r in records], dtype=np.int64)
        layer_of, local = records // self.experts, records % self.experts
        table = np.zeros((self.layers, 4, self.experts), dtype=np.int64)
        table[layer_of, :, local] = address[:, None] + self.host_offsets[None, :]
        globals_ = np.ones((2, self.layers, self.experts), dtype=np.float32)
        base = address - host.pointer
        for k, offset in enumerate(self.global_offsets):
            words = host.array.view(np.float32)
            globals_[k, layer_of, local] = words[(base + offset) // 4]
        # Written on the compute stream (a pageable copy is ordered after the queued steps): a step still queued
        # from the previous phase reads the old entries first. Same reciprocal-then-multiply as _transfer.
        device_keep = torch.from_numpy(keep).to(self.device)
        self.host_table.copy_(torch.from_numpy(table))
        g = torch.from_numpy(globals_).to(self.device)
        self.alpha13.copy_(torch.where(device_keep, (1.0 / g[0]) * self.layer_dequant[:, 0:1], self.alpha13))
        self.alpha2.copy_(torch.where(device_keep, (1.0 / g[1]) * self.layer_dequant[:, 1:2], self.alpha2))
        return from_nvme

    def _paged_preload(self, keep, records, rank, from_nvme):
        """The most wanted seats of D that fit, by this request's rank (the early prediction), else resident first."""
        order = (np.argsort(rank[records], kind='stable') if rank is not None
                 else np.argsort(self.bank.index[records] < 0, kind='stable'))
        target = records[order][:self.capacity]
        self.bank.unprotect_all()
        self.bank.horizon = None
        preference = np.full(self.layers * self.experts, 2.0)
        preference[records] = 1.0
        preference[target] = 0.0
        self.bank.set_priority(preference)   # victims for the preload: outside D first, then D beyond the top G
        events, loaded, early_layers = [], 0, 0
        for layer in range(self.layers):
            wanted = target[target // self.experts == layer] % self.experts
            if not len(wanted):
                continue
            count = int(np.count_nonzero(self.bank.index[layer * self.experts + wanted] < 0))
            event = self.ensure(layer, wanted)
            if event is not None:
                events.append(event)
                loaded += count
                if self.predictor is not None and layer < self.predictor.layer_cap:
                    early_layers += count
            self.bank.protect([layer * self.experts + int(e) for e in wanted])
        self.bank.unprotect_all()
        self.bank.priority = None   # decode evicts by recency alone
        slots = self.bank.index[target]
        self.bank.clock += len(target)
        self.bank.last_used[slots] = self.bank.clock - np.arange(len(target))
        self.decode_set = keep.copy()
        self.decode_loaded_before_layer_cap = early_layers
        if self.decode_stats is not None and self.decode_stats['steps']:
            # A swap within the request (think -> answer): keep the finished phase's counters as their own row.
            if self.cpu_experts is not None and self.cpu_stats_start is not None:
                now = self.cpu_experts.stats()
                self.decode_stats['cpu_experts'] = {k: now[k] - self.cpu_stats_start[k] for k in now}
            if getattr(self, 'decode_stats_history', None) is None:   # engines built without enable_paged
                self.decode_stats_history = []
            self.decode_stats_history.append(self.decode_stats)
        self.decode_stats = {'decode_set': int(len(records)), 'preloaded_resident': int(len(target)),
                             'host_served_at_start': int(len(records) - len(target)), 'held_from_nvme': int(from_nvme),
                             'steps': 0, 'routed_terms': 0, 'misses': 0, 'admitted': 0, 'evicted': 0,
                             'admission_short': 0, 'forced_ring_waits': 0, 'host_seconds': 0.0,
                             # miss_layers[k]: (step, layer) pairs with k misses: how the
                             # misses of a token spread over layers, which sets the cost of serving them per layer.
                             'miss_layers': [0] * 9}
        if self.cpu_experts is not None:
            self.cpu_stats_start = self.cpu_experts.stats()
        return loaded, events

    def enable_cpu_experts(self, spec, rescue=None):
        """Paged decode's absent experts computed by a CPU server. spec: cpu_experts_spec();
        rescue: cpu_rescue_spec() or None (the worker switches it per phase with cpu_experts.set_rescue)."""
        from .cpu_experts import CpuExperts
        if not self.paged:
            raise ValueError('CPU-served misses need paged decode')
        offsets = [self.host_layout[name]['offset'] for name in ('w13', 's13', 'w2', 's2', 'global13', 'global2')]
        index = torch.device(self.device).index
        self.cpu_experts = CpuExperts(self.layers, offsets, spec['cores'], spec['timeout_us'],
                                      torch.cuda.current_device() if index is None else index, spec['per_pair_us'],
                                      absent_cpu=spec['absent'] == 'cpu', rescue_x=0.0 if rescue is None else rescue['x'],
                                      rescue_cap=0 if rescue is None else rescue['cap'])
        self.rescue = rescue
        # The rescue score is weight x gain[layer, expert]; the worker writes each request's gain per phase
        # (BankWorker._set_rescue_gain). All ones until then.
        self.cpu_experts.set_rescue_gain(np.ones((self.layers, self.experts), np.float32), self.device)
        if rescue is not None:
            # A rescued expert is outside the decode set, so its record is not in host_table; the rule needs every
            # record at a fixed address, which only a host tier holding the whole store gives (it never evicts).
            host = self.loader.host
            if host.capacity != self.layers * self.experts or np.any(host.index < 0):
                raise ValueError('The rescue rule needs the whole layer store in the host tier (--host-tier-gib 17)')
            address = host.pointer + host.index.astype(np.int64) * host.stride + self.host_layout['w13']['offset']
            self.rescue_w13 = torch.from_numpy(address.reshape(self.layers, self.experts)).to(self.device)
        self.admit = spec['admit']

    def record_routed(self, layer, ids):
        """Inside the decode graph: this step's masked routed ids for the between-step admission, every row's (rows
        past `self.rows`, CUDA graph padding, are left out; padding rows inside it carry -1, their slot being -1)."""
        rows = min(ids.shape[0], self.rows)
        if rows < self.rows:
            self.routed[layer].fill_(-1)
        self.routed[layer][:rows * 8].copy_(ids[:rows].reshape(-1))

    def enable_premask(self):
        """Routing log (verification only): each decode step's PRE-mask router top-8 ids and weights,
        written inside the decode graph and copied out with the routed ids. Called at load, before graph capture."""
        if not self.paged:
            raise ValueError('The routing log rides the paged decode ring')
        R = self.rows
        self.register_buffer('premask_ids', torch.full((self.layers, R, 8), -1, dtype=torch.int32, device=self.device))
        self.register_buffer('premask_weights', torch.zeros((self.layers, R, 8), dtype=torch.float32, device=self.device))
        # per slot: 1 kept by the mask, 0 dropped, 2 dropped and rescued
        self.register_buffer('premask_state', torch.zeros((self.layers, R, 8), dtype=torch.int8, device=self.device))
        self.premask_ring = [(torch.empty((self.layers, R, 8), dtype=torch.int32).pin_memory(),
                              torch.empty((self.layers, R, 8), dtype=torch.float32).pin_memory(),
                              torch.empty((self.layers, R, 8), dtype=torch.int8).pin_memory(),
                              torch.full((self.layers, R, 8), float('nan'), dtype=torch.float32).pin_memory())
                             for _ in self.ring]
        # Logged steps not yet taken: [requests of the rows, ids [layers, rows, 8], weights, fate, requests not yet taken]
        self.premask = []

    def record_premask(self, layer, ids, weights, state):
        """Inside the decode graph: this step's router top-8 before the mask drops any slot, and each slot's fate,
        one row per request (rows past `self.rows` are CUDA graph padding)."""
        rows = min(ids.shape[0], self.rows)
        self.premask_ids[layer, :rows].copy_(ids[:rows])
        self.premask_weights[layer, :rows].copy_(weights[:rows])
        self.premask_state[layer, :rows].copy_(state[:rows])

    def _log_premask(self, index):
        rids = self.ring_rows[index] or [None]
        ids, weights, fate, norms = self.premask_ring[index]
        self.premask.append([rids, ids.numpy().astype(np.int16), weights.numpy().copy(), fate.numpy().copy(),
                             norms.numpy().copy(), {r for r in rids if r is not None}])

    def take_premask(self, request_id=None):
        """Every logged step of a request so far (waits for copies in flight), oldest first, as (ids, weights, fate)
        [layers, 8] each; the steps are dropped once every request in them has taken its rows. request_id None: the
        single-request engine, whose steps carry row 0."""
        while self.pending_routed:
            self.between_steps(wait=True)
        rows, keep = [], []
        for entry in self.premask:
            rids, ids, weights, fate, norms, untaken = entry
            row = 0 if request_id is None else (rids.index(request_id) if request_id in rids else None)
            if row is not None:
                rows.append((ids[:, row], weights[:, row], fate[:, row], norms[:, row]))
                untaken.discard(request_id)
            if untaken and request_id is not None:
                keep.append(entry)
        self.premask = keep
        return rows

    def after_decode_launch(self, row_requests=None):
        """After a decode step was enqueued: copy its routed ids out, stream-ordered after the step. row_requests: the
        request of each token row (the routing log's attribution), None for the single-request engine."""
        started = time.perf_counter() if _HOST_PHASES else None
        index = self.ring_next % len(self.ring)
        self.ring_next += 1
        if any(i == index for i, _ in self.pending_routed):   # ring full: read the oldest first
            self.decode_stats['forced_ring_waits'] += 1
            self.between_steps(wait=True)
        self.ring_rows[index] = None if row_requests is None else list(row_requests)
        self.ring[index].copy_(self.routed, non_blocking=True)
        if self.premask is not None:
            self.premask_ring[index][0].copy_(self.premask_ids, non_blocking=True)
            self.premask_ring[index][1].copy_(self.premask_weights, non_blocking=True)
            self.premask_ring[index][2].copy_(self.premask_state, non_blocking=True)
            if self.cpu_experts is not None:   # the CPU pairs' output norms, filed by (token row, k)
                R = self.premask_ring[index][3].shape[1]
                self.premask_ring[index][3].copy_(self.cpu_experts.norms()[:, :R * 8].view(self.layers, R, 8),
                                                  non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.pending_routed.append((index, event))
        if started is not None:
            self._phase('ring', started)

    def between_steps(self, wait=False):
        """Before a decode step is enqueued: publish finished admissions to the bookkeeping, then, for every
        earlier step whose routed ids have landed, stamp recency per routed expert and admit its misses."""
        started = time.perf_counter()
        self._settle_admissions(wait=False)
        mark = self._phase('settle', started) if _HOST_PHASES else None
        stats = self.decode_stats
        while self.pending_routed:
            index, event = self.pending_routed[0]
            if not event.query():
                if not wait:
                    break
                event.synchronize()
            self.pending_routed.popleft()
            if mark is not None:
                mark = self._phase('poll', mark)
            if self.premask is not None:
                self._log_premask(index)
            self._process_rows(self.ring[index].numpy())
            wait = False
            if mark is not None:
                mark = time.perf_counter()
        if mark is not None:
            self._phase('poll', mark)
        stats['host_seconds'] += time.perf_counter() - started

    def _process_rows(self, rows):
        """One decode step's masked routed ids [layers, rows * 8]: recency per routed expert, then admit its
        misses."""
        stats = self.decode_stats
        mark = time.perf_counter() if _HOST_PHASES else None
        misses = []
        for layer in range(self.layers):
            ids = rows[layer][rows[layer] >= 0].astype(np.int64)
            experts = layer * self.experts + ids
            resident = self.bank.index[experts] >= 0
            self.bank.touch(experts[resident])
            stats['routed_terms'] += len(ids)
            for expert in experts[~resident]:
                if int(expert) not in self.inflight and self.decode_set.reshape(-1)[expert]:
                    misses.append(int(expert))
            missed = int((~resident).sum())
            stats['misses'] += missed
            stats['miss_layers'][min(missed, 8)] += 1
        stats['steps'] += 1
        if mark is not None:
            self._phase('rows', mark)
        if misses and self.admit:
            self._admit(sorted(set(misses)))

    def _admit(self, experts):
        mark = time.perf_counter() if _HOST_PHASES else None
        bank = self.bank
        evictable = int(np.count_nonzero((bank.owner >= 0) & ~bank.in_flight & ~bank.protected))
        count = min(len(experts), len(bank.free) + evictable)
        if count < len(experts):
            self.decode_stats['admission_short'] += len(experts) - count
        if not count:
            return
        allocations = bank.allocate(experts[:count])
        victims = [victim for _, _, victim in allocations if victim >= 0]
        if mark is not None:
            self._phase('allocate', mark)
        self.copies.append((self._admit_transfer(allocations, victims), allocations))
        if mark is not None:
            mark = time.perf_counter()
        self.inflight.update(expert for expert, _, _ in allocations)
        self.decode_stats['admitted'] += count
        self.decode_stats['evicted'] += len(victims)
        if mark is not None:
            self._phase('admit_rest', mark)

    def _admit_transfer(self, allocations, victims):
        """GPU side of an admission, in A1/A2 order. Returns the completion event."""
        mark = time.perf_counter() if _HOST_PHASES else None
        if victims:   # A2: unpublished in stream order before the next step; the copy waits for it
            self.table.view(-1).index_fill_(0, _device_ids(victims, self.device), -1)
        self.loader_stream.wait_stream(torch.cuda.current_stream())
        if mark is not None:
            mark = self._phase('unpublish', mark)
        by_layer = collections.defaultdict(list)
        for expert, slot, _ in allocations:
            by_layer[expert // self.experts].append((expert % self.experts, slot))
        for layer, pairs in sorted(by_layer.items()):
            local, slots = [p[0] for p in pairs], [p[1] for p in pairs]
            self.loader.load(layer, local, slots)
            if mark is not None:
                mark = time.perf_counter()
            with torch.cuda.stream(self.loader_stream):   # A1: published after its copy, on the copy's stream
                self.table[layer].index_copy_(0, _device_ids(local, self.device),
                                              _device_ids(slots, self.device).to(torch.int32))
            if mark is not None:
                mark = self._phase('publish', mark)
        with torch.cuda.stream(self.loader_stream):
            done = torch.cuda.Event()
            done.record(self.loader_stream)
        return done

    def _settle_admissions(self, wait):
        """Admissions whose copies finished become resident in the bookkeeping (the GPU table already is)."""
        remaining = []
        for event, allocations in self.copies:
            if wait:
                event.synchronize()
            if wait or event.query():
                for expert, slot, _ in allocations:
                    self.bank.complete(expert, slot)
                    self.inflight.discard(expert)
            else:
                remaining.append((event, allocations))
        self.copies = remaining
