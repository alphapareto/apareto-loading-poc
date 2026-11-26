"""Host RAM tier and bank loader.

HostTier: a bounded LRU cache of layer-store records in pinned host RAM; misses are read from the layer store
on NVMe into the least recently used slot.
- Why bounded: pinning the whole store (18.16 GB) left too little reclaimable RAM on the 31.7 GB test rig;
  boot drove the session to 84% memory pressure and systemd-oomd killed it. Pinned pages can be
  neither reclaimed nor swapped, so the tier must be sized, not assumed to fit.
- Why LRU suffices: the decode set stays resident in the GPU bank, so it is never requested from the host and
  ages out here; what prefill requests again and again (the experts decode does not keep) stays recent. A
  tier at least the size of that complement (10,240 - bank slots records) holds it after the first request.
- Why pinned: loads DMA straight from it (one copy over PCIe, 6.8 GB/s on the rig, vs ~3.3 GB/s from NVMe).
- A slot is never overwritten while a DMA queued from it may still be reading it: each slot keeps the CUDA
  event of its last copy, and eviction waits for it.
- Records are not checksummed on read (skipping the checksum halved the NVMe miss rate); the store's header and file identity are checked at open and on every read.

BankLoader: copies experts from the host tier into bank slots.
- One DMA per 1.77 MB record into a GPU staging buffer (measured 6.36 GB/s scattered, against 4.3 GB/s for
  a zero-copy gather kernel), then a GPU kernel splits each record into its fields in the bank's slot.
  GPU-to-GPU is ~100x faster than PCIe, so the split costs next to nothing.
"""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import glob
import mmap
import os
import threading
import time

import numpy as np
import torch
import triton
import triton.language as tl

from apareto_loader.layer_store import LayerStore

GIB = 1 << 30


def _cudart():
    import nvidia
    paths = sorted(p for root in nvidia.__path__ for p in glob.glob(os.path.join(root, 'cu*', 'lib', 'libcudart.so*')))
    library = ctypes.CDLL(paths[0])
    library.cudaHostRegister.restype = ctypes.c_int
    library.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    library.cudaHostUnregister.restype = ctypes.c_int
    library.cudaHostUnregister.argtypes = [ctypes.c_void_p]
    library.cudaMemcpyAsync.restype = ctypes.c_int
    library.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    return library


def mem_available():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable missing from /proc/meminfo')


def _device_ids(values, device):
    """Small index list to the device without blocking the CPU: a pageable copy would wait for everything
    queued on the stream (the loads), serialising the CPU with the DMA it just enqueued."""
    return torch.tensor(values, dtype=torch.int64).pin_memory().to(device, non_blocking=True)


class HostTier:
    """`capacity` records of the layer store in pinned RAM (None: the whole store, preloaded)."""

    def __init__(self, path, ranking=None, source_header_sha256=None, capacity=None, readers=4,
                 reserve_bytes=4 * GIB):
        self.store = LayerStore(path, ranking, source_header_sha256)
        self.stride = self.store.stride
        self.layers, self.experts = self.store.ranking.layers, self.store.ranking.experts
        total = self.layers * self.experts
        self.capacity = total if capacity is None else int(capacity)
        if not self.experts <= self.capacity <= total:
            raise ValueError(f'Host tier capacity {self.capacity} outside [{self.experts}, {total}] records')
        self.size = self.capacity * self.stride
        # Pinned memory cannot be reclaimed: refuse rather than push the machine into memory pressure.
        available = mem_available()
        if self.size + reserve_bytes > available:
            raise ValueError(f'Host tier of {self.size / GIB:.1f} GiB plus {reserve_bytes / GIB:.0f} GiB reserve exceeds '
                             f'MemAvailable {available / GIB:.1f} GiB; configure a smaller host tier')
        self.cudart = _cudart()
        started = time.perf_counter()
        self.map = mmap.mmap(-1, self.size, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        self.array = np.frombuffer(self.map, dtype=np.uint8)
        status = self.cudart.cudaHostRegister(self.array.ctypes.data, self.size, 0)
        if status:
            raise RuntimeError(f'cudaHostRegister of the host tier failed with status {status}')
        self.pointer = self.array.ctypes.data
        self.index = np.full(total, -1, dtype=np.int64)          # record -> host slot
        self.owner = np.full(self.capacity, -1, dtype=np.int64)   # host slot -> record
        self.last_used = np.zeros(self.capacity, dtype=np.int64)
        self.pending = [None] * self.capacity                     # CUDA event of the last copy from the slot
        self.free = list(range(self.capacity - 1, -1, -1))
        self.held = np.zeros(total, dtype=bool)                   # staged ahead of a known load: not evicted
        self.decode_held = np.zeros(total, dtype=bool)            # paged decode: the MoE kernel reads them here
        self.lock = threading.Lock()                              # stage() runs on a background thread
        self.clock = 0
        self.pool = ThreadPoolExecutor(readers)
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'nvme_bytes': 0, 'nvme_seconds': 0.0, 'eviction_waits': 0}
        if self.capacity == total:
            for layer in range(self.layers):
                self.acquire(layer, range(self.experts))
        self.boot_seconds = time.perf_counter() - started

    def _read(self, record, slot):
        layer, expert = divmod(int(record), self.experts)
        view = self.array[slot * self.stride:(slot + 1) * self.stride]
        self.store.read_into(layer, int(self.store.positions[layer, expert]), 1, view, drop_cache=True, verify=False)

    def acquire(self, layer, experts):
        """Host addresses of these records (ids within the layer); misses are read from NVMe first."""
        with self.lock:
            return self._acquire(layer, experts)

    def stage(self, records):
        """Read these records (global ids) into RAM ahead of a known load and hold them until release_held().
        Returns the number read from NVMe."""
        records = np.asarray(sorted(int(r) for r in records), dtype=np.int64)
        before = self.stats['misses']
        for layer in np.unique(records // self.experts):
            ids = records[records // self.experts == layer] % self.experts
            with self.lock:
                self._acquire(int(layer), ids)
                self.held[layer * self.experts + ids] = True
        return self.stats['misses'] - before

    def release_held(self):
        with self.lock:
            self.held[:] = False

    def hold_decode(self, records):
        """Paged decode: read these records (global ids) into RAM and keep them at fixed
        addresses until release_decode(), because the MoE kernel reads an absent expert from its record here.
        Returns ({record: address}, records read from NVMe)."""
        records = np.asarray(sorted(int(r) for r in records), dtype=np.int64)
        before, addresses = self.stats['misses'], {}
        for layer in np.unique(records // self.experts):
            ids = records[records // self.experts == layer] % self.experts
            with self.lock:
                found, _ = self._acquire(int(layer), ids)
                self.decode_held[layer * self.experts + ids] = True
            addresses.update(zip((layer * self.experts + ids).tolist(), found))
        return addresses, self.stats['misses'] - before

    def release_decode(self):
        """Only once no queued kernel can still read a held record (the request has finished)."""
        with self.lock:
            self.decode_held[:] = False

    def _acquire(self, layer, experts):
        records = [layer * self.experts + int(e) for e in experts]
        if len(records) > self.capacity:
            raise ValueError('More records requested at once than the host tier holds')
        misses = [r for r in records if self.index[r] < 0]
        if misses:
            need = len(misses) - len(self.free)
            if need > 0:
                candidates = np.flatnonzero(self.owner >= 0)
                kept = self.held[self.owner[candidates]] | self.decode_held[self.owner[candidates]]
                candidates = candidates[~np.isin(self.owner[candidates], records) & ~kept]
                victims = candidates[np.argsort(self.last_used[candidates], kind='stable')[:need]]
                if len(victims) < need:
                    raise RuntimeError('Host tier cannot hold the requested records')
                for slot in victims:
                    event = self.pending[slot]
                    if event is not None and not event.query():
                        self.stats['eviction_waits'] += 1
                        event.synchronize()
                    self.pending[slot] = None
                    self.index[self.owner[slot]] = -1
                    self.owner[slot] = -1
                    self.free.append(int(slot))
                self.stats['evictions'] += len(victims)
            slots = [self.free.pop() for _ in misses]
            before = time.perf_counter()
            list(self.pool.map(self._read, misses, slots))
            self.stats['nvme_seconds'] += time.perf_counter() - before
            self.stats['nvme_bytes'] += len(misses) * self.stride
            for record, slot in zip(misses, slots):
                self.index[record] = slot
                self.owner[slot] = record
        self.stats['hits'] += len(records) - len(misses)
        self.stats['misses'] += len(misses)
        self.clock += 1
        slots = self.index[records]
        self.last_used[slots] = self.clock
        return [self.pointer + int(slot) * self.stride for slot in slots], slots

    def retire(self, slots, event):
        """Copies from these slots were queued before `event`: they may not be overwritten until it completes."""
        for slot in slots:
            self.pending[int(slot)] = event

    def resident(self):
        return int((self.owner >= 0).sum())


@triton.jit
def _split(staging, bank_base, slots, offsets, sizes, destinations, strides, STRIDE: tl.constexpr,
           FIELDS: tl.constexpr, BLOCK: tl.constexpr):
    """Record i of the staging buffer -> its fields at slot slots[i] of each bank tensor (8-byte words)."""
    record = tl.program_id(0)
    field = tl.program_id(1)
    chunk = tl.program_id(2)
    slot = tl.load(slots + record).to(tl.int64)
    offset = tl.load(offsets + field)
    size = tl.load(sizes + field)
    destination = tl.load(destinations + field).to(tl.pointer_type(tl.int64))
    stride = tl.load(strides + field)
    words = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = words < size
    value = tl.load(staging + record * STRIDE + offset + words, mask=mask)
    tl.store(destination + slot * stride + words, value, mask=mask)


class BankLoader:
    """Host tier -> GPU staging (DMA per record) -> bank slots (split kernel), on one CUDA stream."""

    def __init__(self, host, bank_tensors, staging_records=64, device='cuda'):
        """bank_tensors: name -> device tensor [capacity, ...] for every record field the bank stores."""
        self.host = host
        layout = host.store.layout
        if set(bank_tensors) != set(layout):
            raise ValueError(f'Bank fields {sorted(bank_tensors)} differ from store fields {sorted(layout)}')
        for name, tensor in bank_tensors.items():
            per_slot = tensor[0].numel() * tensor.element_size()
            if per_slot != layout[name]['nbytes'] or not tensor.is_contiguous():
                raise ValueError(f'Bank {name} slot holds {per_slot} bytes, store {layout[name]["nbytes"]}')
        names = sorted(layout, key=lambda n: layout[n]['offset'])
        if any(layout[n]['offset'] % 8 for n in names) or host.stride % 8:
            raise ValueError('Store fields are not word-aligned')
        # Large fields (weights, block scales) move by the split kernel in 8-byte words; the float32
        # scalars (global scales) by a small gather, since a 4-byte field is not a whole word.
        self.large = [n for n in names if layout[n]['nbytes'] % 8 == 0]
        self.small = [n for n in names if layout[n]['nbytes'] % 8]
        self.capacity = next(iter(bank_tensors.values())).shape[0]
        self.staging_records = staging_records
        self.staging = torch.empty(staging_records * host.stride // 8, dtype=torch.int64, device=device)
        self.word_sizes = [layout[n]['nbytes'] // 8 for n in self.large]
        self.destinations = torch.tensor([bank_tensors[n].data_ptr() for n in self.large], dtype=torch.int64, device=device)
        self.offsets = torch.tensor([layout[n]['offset'] // 8 for n in self.large], dtype=torch.int64, device=device)
        self.sizes = torch.tensor(self.word_sizes, dtype=torch.int64, device=device)
        self.strides = torch.tensor(self.word_sizes, dtype=torch.int64, device=device)
        self.bank = bank_tensors
        self.stream = torch.cuda.Stream(device=device)
        self.bytes = 0
        self.phase = None   # measurement only: phase(name, since) -> now (bank_engine, APARETO_HOST_PHASES)

    def load(self, layer, experts, slots):
        """Enqueue copies of `experts` (ids within the layer) into bank `slots`; returns the completion event."""
        if len(experts) != len(slots):
            raise ValueError('Experts and slots differ in length')
        stream = self.stream
        handle = stream.cuda_stream
        stride = self.host.stride
        phase = self.phase
        mark = time.perf_counter() if phase is not None else None
        with torch.cuda.stream(stream):
            for start in range(0, len(experts), self.staging_records):
                batch_experts = experts[start:start + self.staging_records]
                batch_slots = slots[start:start + self.staging_records]
                base = self.staging.data_ptr()
                addresses, host_slots = self.host.acquire(layer, batch_experts)
                if mark is not None:
                    mark = phase('load_acquire', mark)
                for i, address in enumerate(addresses):
                    status = self.host.cudart.cudaMemcpyAsync(base + i * stride, address, stride, 1, handle)
                    if status:
                        raise RuntimeError(f'cudaMemcpyAsync failed with status {status}')
                if mark is not None:
                    mark = phase('load_memcpy', mark)
                copied = torch.cuda.Event()
                copied.record(stream)
                self.host.retire(host_slots, copied)
                if mark is not None:
                    mark = phase('load_retire', mark)
                count = len(batch_experts)
                slot_tensor = _device_ids(batch_slots, self.staging.device)
                block = 2048
                _split[(count, len(self.large), triton.cdiv(max(self.word_sizes), block))](
                    self.staging, 0, slot_tensor, self.offsets, self.sizes, self.destinations, self.strides,
                    STRIDE=stride // 8, FIELDS=len(self.large), BLOCK=block)
                if mark is not None:
                    mark = phase('load_split', mark)
                # Scalars (float32 fields) by gather: tiny, exact.
                staging_bytes = self.staging.view(torch.uint8)[:count * stride].view(count, stride)
                for name in self.small:
                    field = self.host.store.layout[name]
                    values = staging_bytes[:, field['offset']:field['offset'] + field['nbytes']].contiguous()
                    self.bank[name].view(torch.uint8).view(self.capacity, -1)[slot_tensor] = values
                if mark is not None:
                    mark = phase('load_gather', mark)
                self.bytes += count * stride
            done = torch.cuda.Event()
            done.record(stream)
        return done
