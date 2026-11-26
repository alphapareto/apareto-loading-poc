"""CPU-served decode misses: paged decode hands the experts that are absent from the bank to a
CPU server instead of reading them over PCIe.

Per MoE layer of a decode step, inside the decode CUDA graph:
- ring (replaces the paged branch's mask ops): the GPU's ids and scales (allowed and resident, or absent without a
  host record), the routed ids for the admission, and for the absent allowed experts a request in pinned mapped
  memory plus a doorbell;
- the bank's slot kernel on the GPU pairs (an id of -1 is skipped, so the kernel is unchanged);
- finish: wait for the server's done flag and add its rows; after the timeout compute the timed-out pairs zero-copy
  from their records on the GPU. Nothing is dropped.

The server is a C++ thread with a worker pool pinned to the given cores; it reads the records in place in the host
tier (their host_table addresses are host addresses: the tier is registered, UVA).

Built with torch's cpp_extension on first use (sm_120 machine code, AVX2); the build runs in the worker process,
so build it once beforehand in a capped scope (tests/test_cpu_experts.py does).
"""
import os
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent / 'cpu_experts_src'
COUNTERS = ('layers', 'pairs', 'timeouts', 'fallback_pairs', 'wait_ns', 'no_host_record',
            'rescued', 'rescued_uweight', 'rescue_no_record', 'rescued_gpu', 'served_requests', 'served_pairs', 'served_experts', 'stale', 'compute_ns')
_module = None


def module():
    global _module
    if _module is None:
        import shutil
        import sys
        from torch.utils.cpp_extension import load
        os.environ.setdefault('MAX_JOBS', '2')
        if shutil.which('ninja') is None:   # the venv's ninja, when the venv is not on PATH (the server's case)
            os.environ['PATH'] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
        _module = load(name='apareto_cpu_experts',
                       sources=[str(SRC / 'cpu_experts.cpp'), str(SRC / 'cpu_experts_kernels.cu')],
                       extra_cflags=['-O3', '-mavx2', '-mfma', '-mf16c'],
                       extra_cuda_cflags=['-O3', '-gencode=arch=compute_120,code=sm_120'],
                       verbose=False)
    return _module


def parse_cores(text):
    """'4-7' or '4,5,6,7' -> [4, 5, 6, 7]; 'auto:N' -> the last N cores this process may run on (a server started
    under taskset shares its own cores, so two servers on one machine never pin the same core)."""
    if str(text).startswith('auto'):
        mine = sorted(os.sched_getaffinity(0))
        return mine[-int(str(text).partition(':')[2] or len(mine)):]
    cores = []
    for part in str(text).split(','):
        a, _, b = part.partition('-')
        cores.extend(range(int(a), int(b or a) + 1))
    return cores


class CpuExperts:
    """One per worker process. offsets: the record field offsets (layer store layout) in w13, s13, w2, s2, global13,
    global2 order."""

    def __init__(self, layers, offsets, cores, timeout_us=1000.0, device=0, per_pair_us=150.0, absent_cpu=True,
                 rescue_x=0.0, rescue_cap=0):
        self.m = module()
        self.layers = layers
        self.cores = parse_cores(cores)
        # rescue_cap: one cap for every layer, or one per layer; 0 = no rescue there.
        caps = [int(rescue_cap)] * int(layers) if isinstance(rescue_cap, (int, np.integer)) else [int(c) for c in rescue_cap]
        if len(caps) != int(layers) or not all(0 <= c <= 8 for c in caps):
            raise ValueError(f'rescue cap: one value in 0..8, or {layers} of them')
        self.rescue_caps = caps
        self.m.start(int(layers), [int(x) for x in offsets], self.cores, float(timeout_us), float(per_pair_us),
                     int(device), bool(absent_cpu), float(rescue_x), caps)
        self.ids = {}       # per token count: GPU ids / scales and CPU ids / scales (graph-stable addresses)
        # The rescue score's gain per (layer, expert): all ones is the plain weight rule.
        # Filled in place (set_rescue_gain) before any graph capture, so its address stays fixed.
        self.rescue_gain = None
        if str(cores).startswith('auto'):
            self.isolate()

    def isolate(self, wait_s=5.0):
        """The CPU path's cores serve only its threads: every other thread of this process moves to the process's
        remaining cores (threads created later inherit their creator's affinity). Shared cores let the scheduler
        preempt the server for a time slice, which the GPU sees as a timeout."""
        import time
        mine = set(self.cores)
        deadline = time.monotonic() + wait_s
        while True:     # the server and pool threads pin themselves when they start
            ours = [t for t in os.listdir('/proc/self/task') if os.sched_getaffinity(int(t)) <= mine
                    and len(os.sched_getaffinity(int(t))) == 1]
            if len(ours) >= len(mine) or time.monotonic() > deadline:
                break
            time.sleep(0.01)
        if len(ours) < len(mine):
            raise RuntimeError(f'CPU path: {len(ours)} of {len(mine)} threads pinned to {sorted(mine)}')
        moved = 0
        for t in os.listdir('/proc/self/task'):
            if t in ours:
                continue
            try:
                current = os.sched_getaffinity(int(t))
                rest = current - mine
                if rest and rest != current:
                    os.sched_setaffinity(int(t), rest)
                    moved += 1
            except ProcessLookupError:
                pass
        self.isolated = {'cores': sorted(mine), 'cpu_path_threads': len(ours), 'threads_moved': moved}

    def _buffers(self, tokens, device):
        if tokens not in self.ids:
            self.ids[tokens] = tuple(torch.empty((tokens, 8), dtype=dtype, device=device)
                                     for dtype in (torch.int32, torch.float32, torch.int32, torch.float32))
        return self.ids[tokens]

    def ring(self, layer, logical, weights, keep, table, host_w13, hidden, routed, rescue_w13=None, row_slot=None):
        """Returns the GPU's (ids, scales) for this layer; the CPU's share is requested. rescue_w13: every expert's
        record address in the host tier (the rescue rule's lookup; host_w13 holds the decode set's only). keep: [256],
        or [slots, 256] with row_slot naming each token row's slot; routed:
        every row's allowed ids, -1 past this step's rows."""
        ids, scales, cpu_ids, cpu_scales = self._buffers(logical.shape[0], logical.device)
        if self.rescue_gain is None:
            self.rescue_gain = torch.ones((self.layers, keep.shape[-1]), dtype=torch.float32, device=logical.device)
        if row_slot is None:
            row_slot = self._zeros(logical.shape[0], logical.device)
        self.m.ring(layer, logical, weights.float().contiguous(), keep, table, host_w13, hidden.contiguous(), routed,
                    ids, scales, cpu_ids, cpu_scales, host_w13 if rescue_w13 is None else rescue_w13, row_slot,
                    self.rescue_gain[layer])
        return ids, scales

    def _zeros(self, tokens, device):
        key = ('slots', tokens)
        if key not in self.ids:
            self.ids[key] = torch.zeros(tokens, dtype=torch.int32, device=device)
        return self.ids[key]

    def served(self, tokens):
        """The last ring's CPU share per (token, k): ids (-1 where the GPU serves) and routing weights (0 there).
        Valid until the next layer's ring, in stream order: the verification audit reads it inside engine.run."""
        return self.ids[tokens][2], self.ids[tokens][3]

    def norms(self):
        """[layers, 64] float32 on the device: the last step's per-(token row, k) L2 norm of each CPU-computed expert
        output, before its routing weight; -1 not computed on the CPU, -2 timed out."""
        return self.m.norms()

    def finish(self, layer, fused):
        self.m.finish(layer, fused)
        return fused

    def set_enabled(self, on):
        self.m.set_enabled(bool(on))

    def set_rescue(self, on):
        """The rescue rule on or off for decode steps enqueued from now on (stream-ordered)."""
        self.m.set_rescue(bool(on))

    def set_rescue_gain(self, gain, device, slot=None):
        """gain: [layers, experts] positive finite floats, the rescue score's per-expert factor. Copied in place
        into the buffer the ring reads; the first call (or slot_gain) fixes its shape and must precede graph capture.
        slot: after slot_gain, write only that keep slot's row (a per-request gain under concurrent decode)."""
        gain = torch.as_tensor(gain, dtype=torch.float32)
        if gain.ndim != 2 or gain.shape[0] != self.layers or not torch.isfinite(gain).all() or (gain <= 0).any():
            raise ValueError(f'rescue gain: [{self.layers}, experts] positive finite values, got {tuple(gain.shape)}')
        if self.rescue_gain is None:
            self.rescue_gain = torch.ones(tuple(gain.shape), dtype=torch.float32, device=device)
        if slot is not None:
            if self.rescue_gain.ndim != 3 or tuple(self.rescue_gain[:, slot].shape) != tuple(gain.shape):
                raise ValueError('rescue gain: per-slot write needs slot_gain first')
            self.rescue_gain[:, slot].copy_(gain.to(self.rescue_gain.device))
            return
        if tuple(self.rescue_gain.shape) != tuple(gain.shape):
            raise ValueError('rescue gain shape changed')
        self.rescue_gain.copy_(gain.to(self.rescue_gain.device))

    def slot_gain(self, slots, device):
        """One gain row per keep slot ([layers, slots, experts], all ones), before any graph capture."""
        experts = self.rescue_gain.shape[-1] if self.rescue_gain is not None else None
        if experts is None:
            raise ValueError('slot_gain after the shared gain is set (enable_cpu_experts)')
        self.rescue_gain = torch.ones((self.layers, int(slots), experts), dtype=torch.float32, device=device)

    def stats(self):
        return dict(zip(COUNTERS, self.m.stats()))

    def stop(self):
        self.m.stop()


_E2M1 = None


def dequant_record(address, layout):
    """fp64 (w13 [2F, H] as [up; gate], w2 [H, F]) of the record at a host address, from the stored bytes: the
    unswizzle of layer_store.inverse_record, independent of the kernels' scale-offset arithmetic."""
    import ctypes
    import numpy as np
    global _E2M1
    if _E2M1 is None:
        _E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])

    def raw(name):
        f = layout[name]
        return np.frombuffer(ctypes.string_at(address + f['offset'], f['nbytes']), dtype=np.uint8)

    def e4m3(x):
        x = x.astype(np.int64)
        e, m = (x >> 3) & 0xF, x & 7
        return np.where((x & 0x7F) == 0x7F, 0.0, np.where(e == 0, m * 2.0 ** -9, (1 + m / 8) * 2.0 ** (e - 7)))

    def matrix(packed, scales, rows, cols):
        q = raw(packed).reshape(rows, cols // 2)
        w = np.empty((rows, cols))
        w[:, 0::2], w[:, 1::2] = _E2M1[q & 15], _E2M1[q >> 4]
        pr, pc = layout[scales]['shape']
        s = raw(scales).reshape(pr // 128, pc // 4, 32, 4, 4).transpose(0, 3, 2, 1, 4).reshape(pr, pc)[:rows, :cols // 16]
        return w * np.repeat(e4m3(s), 16, axis=1)

    rows13, cols13 = layout['w13']['shape'][0], layout['w13']['shape'][1] * 2
    rows2, cols2 = layout['w2']['shape'][0], layout['w2']['shape'][1] * 2
    g13 = raw('global13').view(np.float32)[0]
    g2 = raw('global2').view(np.float32)[0]
    return matrix('w13', 's13', rows13, cols13) / g13, matrix('w2', 's2', rows2, cols2) / g2


class NumericsProbe:
    """Measurement only (eager boots): for every absent allowed (token, expert) of the first
    `steps` paged decode steps, the expert's output three ways from the same MoE input row: the GPU path as served
    without the CPU (FP4 activations, zero-copy), the CPU kernel (int8 activations), and fp64 from the record.
    Writes one JSON row per pair: relative L2 error and cosine of the GPU and CPU outputs against fp64."""

    def __init__(self, spec):
        path, _, steps = spec.partition(':')
        self.path, self.steps = Path(path), int(steps or 20)
        if self.path.exists():
            raise FileExistsError(f'{self.path} exists; give a new file')
        self.seen = 0
        self.log = None

    def observe(self, engine, layer, hidden, logical, weights, keep):
        import json
        import numpy as np
        if torch.cuda.is_current_stream_capturing() or self.seen > self.steps:
            return
        if layer == 0:
            self.seen += 1
        if self.seen == 0 or self.seen > self.steps:
            return
        if self.log is None:
            self.log = self.path.open('x')
        ids = logical.long()
        allowed = keep[ids] & (ids >= 0)
        absent = allowed & (engine.table[layer][ids] < 0)
        pairs = absent.nonzero().tolist()
        if not pairs:
            return
        offsets = {n: engine.host_layout[n]['offset'] for n in ('w13', 's13', 'w2', 's2', 'global13', 'global2')}
        host_w13 = engine.host_table[layer, 0].cpu().numpy()
        for t, k in pairs:
            e = int(ids[t, k])
            base = int(host_w13[e]) - offsets['w13']
            if host_w13[e] == 0:
                continue
            x = hidden[t:t + 1].contiguous()
            gid = torch.full((1, 8), -1, dtype=torch.int32, device=hidden.device); gid[0, 0] = e
            gw = torch.zeros((1, 8), dtype=weights.dtype, device=hidden.device); gw[0, 0] = 1
            gpu = engine.run(layer, x, gid, gw, host=engine.host_table)[0].double().cpu().numpy()
            cpu = module().expert(base, x[0].float().cpu().contiguous(),
                                  [offsets[n] for n in ('w13', 's13', 'w2', 's2', 'global13', 'global2')], 1)
            cpu = cpu.double().numpy()
            w13, w2 = dequant_record(base, engine.host_layout)
            xf = x[0].double().cpu().numpy()
            middle = w13.shape[0] // 2
            up, gate = w13[:middle] @ xf, w13[middle:] @ xf
            ref = w2 @ (gate / (1 + np.exp(-gate)) * up)
            norm = float(np.linalg.norm(ref)) or 1e-30
            row = {'step': self.seen, 'layer': layer, 'token': t, 'expert': e, 'ref_norm': norm,
                   'rel_gpu': float(np.linalg.norm(gpu - ref) / norm), 'rel_cpu': float(np.linalg.norm(cpu - ref) / norm),
                   'cos_gpu': float(gpu @ ref / (np.linalg.norm(gpu) * norm + 1e-30)),
                   'cos_cpu': float(cpu @ ref / (np.linalg.norm(cpu) * norm + 1e-30))}
            self.log.write(json.dumps(row) + '\n')
        self.log.flush()
