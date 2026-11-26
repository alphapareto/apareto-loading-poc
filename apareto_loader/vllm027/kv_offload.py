"""Host KV tier for paused requests: the worker's copies.

An entry is one paused request at one position: the raw bytes of its held attention blocks in every KV storage tensor, and
its GDN snapshot slot, in pages of one fixed pinned arena (host_bytes, allocated once). Both move as bytes, never
through a dtype view, so an onload writes back exactly what the offload read.
- KV storage is block-major (vllm/v1/worker/utils.py copy_kv_cache_blocks_inplace, the copy-on-write path): block i
  of a storage tensor owns bytes [i * page, (i + 1) * page), and the hybrid allocator's storages are shared across
  KV-cache groups, so the rows of the attention group's block ids, over every storage, are exactly that group's KV.
- No GPU staging buffer (one sized for a request costs GPU memory the KV pool needs): each block row is copied
  straight between the GPU storage and its pinned host row.
- The scheduler stamps the instructions (concurrent_pause.py) and the worker runs them in _update_states,
  in order, then synchronizes the stream: an offload is complete before the scheduler, acknowledging it at its next
  schedule, can free the blocks, and an onload is complete before the step's copy-on-write and forward read them.
- verify (verification runs): after an onload the written rows and slot are read back and compared
  bit for bit with the entry; a mismatch fails the run.
"""
import torch


class HostKVTier:
    def __init__(self, runner, snapshot_buffers, verify=False, host_bytes=0):
        self.runner, self.snapshot_buffers, self.verify = runner, snapshot_buffers, verify
        self.entries = {}           # key -> {'pos', 'blocks', 'pages': [one per block], 'snap_pages': [...]}
        self.counts = {'offloads': 0, 'onloads': 0, 'drops': 0, 'verified_onloads': 0,
                       'offload_bytes': 0, 'onload_bytes': 0}
        self._storages = None
        # The arena: one pinned buffer of exactly host_bytes, allocated here, cut into pages of one KV
        # block (every storage's row of it, side by side) when the storages are bound. An entry takes one page per
        # block and enough pages for its GDN snapshot; a drop returns them. Same-size pages cannot fragment, and the
        # RAM never grows: a buffer per copy would leave far more pinned than live in PyTorch's power-of-two
        # host cache. The registry sizes entries in the same
        # pages, so what it admits always fits.
        device = next((t.device for slot in snapshot_buffers for ts in slot.values() for t in ts), None)
        self.arena = torch.empty(int(host_bytes), dtype=torch.uint8,
                                 pin_memory=device is not None and device.type == 'cuda')
        self.pages, self.page_bytes, self.free = None, None, []

    def storages(self):
        """Each KV storage tensor once, as uint8 rows of one block each (bound after KV allocation)."""
        if self._storages is None:
            num_blocks = self.runner.kv_cache_config.num_blocks
            seen, rows = set(), []
            for entry in self.runner.kv_caches:
                for tensor in entry if isinstance(entry, (list, tuple)) else (entry,):
                    pointer = tensor.untyped_storage().data_ptr()
                    if pointer in seen:
                        continue
                    seen.add(pointer)
                    raw = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                    raw.set_(tensor.untyped_storage())
                    if raw.numel() % num_blocks:
                        raise RuntimeError('A KV storage tensor is not block-major; the host tier cannot copy it')
                    rows.append(raw.view(num_blocks, -1))
            self._storages = rows
            self.page_bytes = sum(r.shape[1] for r in rows)
            count = self.arena.numel() // self.page_bytes if self.page_bytes else 0
            self.pages = self.arena[:count * self.page_bytes].view(count, self.page_bytes)
            self.free = list(range(count - 1, -1, -1))
        return self._storages

    def entry_bytes(self, key):
        entry = self.entries.get(key)
        if entry is None:
            return 0
        return (len(entry['pages']) + len(entry['snap_pages'])) * self.page_bytes

    def run(self, ops):
        """Apply one output's host instructions in order, then wait for the copies."""
        if not ops:
            return
        self.storages()
        for op in ops:
            kind = op[0]
            if kind == 'drop':
                if self._release(op[1]):
                    self.counts['drops'] += 1
            elif kind == 'copy':
                self._offload(*op[1:])
            elif kind == 'onload':
                self._onload(*op[1:])
            else:
                raise RuntimeError(f'Unknown host KV instruction {kind!r}')
        storage = self.storages()[0] if self.storages() else None
        if storage is not None and storage.is_cuda:
            torch.cuda.current_stream(storage.device).synchronize()
        if self.verify:
            for op in ops:
                if op[0] == 'onload':
                    self._check(*op[1:])

    def _release(self, key):
        entry = self.entries.pop(key, None)
        if entry is None:
            return False
        self.free += entry['pages'] + entry['snap_pages']
        return True

    def _take(self, count, key):
        if count > len(self.free):
            raise RuntimeError(f'Host arena exhausted copying {key}: {count} pages wanted, {len(self.free)} '
                               f'free of {len(self.pages)}; the registry admitted more than the tier holds')
        taken, self.free = self.free[-count:] if count else [], self.free[:-count] if count else self.free
        return taken

    def _snapshot_chunks(self, slot, pages):
        """The slot's tensors as one byte stream laid over its pages: (host view, device bytes) pairs."""
        position = 0
        for name, tensors in sorted(self.snapshot_buffers[slot].items()):
            for tensor in tensors:
                flat, done = _bytes(tensor), 0
                while done < flat.numel():
                    page, offset = divmod(position, self.page_bytes)
                    take = min(flat.numel() - done, self.page_bytes - offset)
                    yield self.pages[pages[page], offset:offset + take], flat[done:done + take]
                    done += take
                    position += take

    def _snapshot_pages(self, slot):
        total = sum(t.nbytes for tensors in self.snapshot_buffers[slot].values() for t in tensors)
        return -(-total // self.page_bytes)

    def _offload(self, key, slot, pos, block_ids):
        self._release(key)
        pages = self._take(len(block_ids), key)
        snap_pages = self._take(self._snapshot_pages(slot), key)
        for page, block in zip(pages, block_ids):
            offset = 0
            for rows in self.storages():
                width = rows.shape[1]
                self.pages[page, offset:offset + width].copy_(rows[block], non_blocking=True)
                offset += width
        for host, device in self._snapshot_chunks(slot, snap_pages):
            host.copy_(device, non_blocking=True)
        self.entries[key] = {'pos': int(pos), 'blocks': len(block_ids), 'pages': pages,
                                    'snap_pages': snap_pages}
        self.counts['offloads'] += 1
        self.counts['offload_bytes'] += self.entry_bytes(key)

    def _onload(self, key, slot, pos, block_ids):
        entry = self.entries.get(key)
        if entry is None or entry['pos'] != int(pos):
            raise RuntimeError(f'Onload of {key} at {pos}: the host holds '
                               f'{None if entry is None else entry["pos"]}')
        if entry['blocks'] != len(block_ids):
            raise RuntimeError(f'Onload of {key}: {len(block_ids)} blocks for an entry of {entry["blocks"]}')
        for page, block in zip(entry['pages'], block_ids):
            offset = 0
            for rows in self.storages():
                width = rows.shape[1]
                rows[block].copy_(self.pages[page, offset:offset + width], non_blocking=True)
                offset += width
        for host, device in self._snapshot_chunks(slot, entry['snap_pages']):
            device.copy_(host, non_blocking=True)
        self.counts['onloads'] += 1
        self.counts['onload_bytes'] += self.entry_bytes(key)

    def _check(self, key, slot, pos, block_ids):
        entry = self.entries[key]
        # Row by row: a gather of the whole request would be the GPU staging buffer the tier avoids.
        for page, block in zip(entry['pages'], block_ids):
            offset = 0
            for rows in self.storages():
                width = rows.shape[1]
                if not torch.equal(rows[block].cpu(), self.pages[page, offset:offset + width]):
                    raise RuntimeError(f'Onload of {key} at {pos}: KV block {block} differs from the host copy')
                offset += width
        for host, device in self._snapshot_chunks(slot, entry['snap_pages']):
            if not torch.equal(device.cpu(), host):
                raise RuntimeError(f'Onload of {key} at {pos}: GDN snapshot differs from the host')
        self.counts['verified_onloads'] += 1


def _bytes(tensor):
    """A contiguous tensor's storage range as flat uint8, for byte-exact copies."""
    if not tensor.is_contiguous():
        raise RuntimeError('GDN snapshot buffers must be contiguous')
    return tensor.view(-1).view(torch.uint8)
