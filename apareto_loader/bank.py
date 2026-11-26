"""Expert bank bookkeeping: CPU master copy.

- index[expert] -> slot or -1, over all 10,240 experts (expert = layer * 256 + e).
- owner[slot]   -> expert or -1, over the bank's slots.
- free          -> stack of free slots.
- per slot: in_flight (copy not complete), protected (must not be evicted now), last_used (fallback order).

A load is two-phase: `allocate` reserves a slot (the expert is not yet visible: index stays -1), the caller
copies the expert in, then `complete` publishes the index entry. Why: compute must never see an index
entry pointing at a half-loaded expert.

Eviction happens only when no slot is free. It never takes an in-flight or protected slot. Among the rest
it evicts the expert this request is least likely to want back, by the early R* ranking: the highest fused
rank first. Before a ranking exists it falls back to least recently used.
Horizon (during a prefill pass): experts of layers the pass has not reached yet are taken only when no
expert of an already-computed layer is evictable, because the pass would load them again a few layers
later. It changes which expert is chosen, never when: nothing is evicted before a slot is needed.
"""
import numpy as np


def layer_quotas(budget, layers=40):
    """Per-layer keep counts of a budget, as the selectors cut it (equal split, remainder to the first layers)."""
    return [budget // layers + (1 if layer < budget % layers else 0) for layer in range(layers)]


def bank_minimum(budget, layer_cap, mode, layers=40, experts=256):
    """Slots the bank needs so that nothing protected is ever evicted.
    'layers': the decode set D plus the first layer_cap layers in full, all protected until early R* predicts
    (4,893 at budget 4,301, layer_cap 4). 'decode': D plus the two layers in flight (computing and loading)."""
    quotas = layer_quotas(budget, layers)
    if mode == 'layers':
        return layer_cap * experts + sum(quotas[layer_cap:])
    if mode == 'decode':
        return budget + 2 * max(experts - quota for quota in quotas)
    raise ValueError(f'Unknown pre-prediction mode {mode!r}')


class BankFull(RuntimeError):
    """No free slot and nothing evictable: the bank is too small for what is protected and in flight."""


class ExpertBank:
    def __init__(self, capacity, layers=40, experts=256):
        if capacity < 1:
            raise ValueError('Bank needs at least one slot')
        self.capacity, self.layers, self.experts = capacity, layers, experts
        self.index = np.full(layers * experts, -1, dtype=np.int32)
        self.owner = np.full(capacity, -1, dtype=np.int32)
        self.free = list(range(capacity - 1, -1, -1))   # pop() yields slot 0 first
        self.in_flight = np.zeros(capacity, dtype=bool)
        self.protected = np.zeros(capacity, dtype=bool)
        self.last_used = np.zeros(capacity, dtype=np.int64)
        self.clock = 0
        self.priority = None   # per expert: larger = less wanted by the current request (early R* rank)
        self.horizon = None    # first layer a prefill pass has not computed yet (None outside prefill)
        self.evictions = 0

    # -- queries ---------------------------------------------------------------------------------------
    def slot(self, expert):
        return int(self.index[expert])

    def missing(self, layer):
        """Experts of a layer that are neither resident nor already being loaded."""
        base = layer * self.experts
        row = self.index[base:base + self.experts]
        loading = set(int(e) for e in self.owner[self.in_flight])
        return [base + e for e in np.flatnonzero(row < 0) if base + e not in loading]

    def table(self):
        """[layers, experts] expert -> slot, the per-layer tables the decode kernel reads."""
        return self.index.reshape(self.layers, self.experts)

    # -- policy inputs -----------------------------------------------------------------------------------
    def set_priority(self, fused_ranks):
        """Early R* fused ranks [layers, experts] for the current request: lower = more wanted."""
        ranks = np.asarray(fused_ranks, dtype=np.float64).reshape(-1)
        if ranks.shape != self.index.shape or not np.isfinite(ranks).all():
            raise ValueError('Priority must be finite and cover every expert')
        self.priority = ranks

    def protect(self, experts, value=True):
        slots = self.index[np.asarray(list(experts), dtype=np.int64)]
        if np.any(slots < 0):
            raise ValueError('Only resident experts can be protected')
        self.protected[slots] = value

    def unprotect_all(self):
        self.protected[:] = False

    def touch(self, experts):
        self.clock += 1
        slots = self.index[np.asarray(list(experts), dtype=np.int64)]
        self.last_used[slots[slots >= 0]] = self.clock

    # -- load / free ---------------------------------------------------------------------------------------
    def allocate(self, experts):
        """Reserve one slot per expert, evicting if needed. Returns [(expert, slot, evicted_expert or -1)].

        All-or-nothing: if the bank cannot provide every slot, nothing changes and BankFull is raised.
        """
        experts = [int(e) for e in experts]
        if len(set(experts)) != len(experts):
            raise ValueError('Duplicate experts in one allocation')
        loading = set(int(e) for e in self.owner[self.in_flight])
        for expert in experts:
            if self.index[expert] >= 0 or expert in loading:
                raise ValueError(f'Expert {expert} is already resident or loading')
        need = max(0, len(experts) - len(self.free))
        victims = self._victims(need) if need else []
        if len(victims) < need:
            raise BankFull(f'Need {need} evictions, {len(victims)} evictable slots')
        victim_experts = {}
        for slot, expert in victims:
            self._release(expert)
            victim_experts[slot] = expert
            self.evictions += 1
        out = []
        for expert in experts:
            slot = self.free.pop()
            self.owner[slot] = expert
            self.in_flight[slot] = True
            out.append((expert, slot, victim_experts.get(slot, -1)))
        return out

    def complete(self, expert, slot):
        """The copy into `slot` finished: publish the expert."""
        if self.owner[slot] != expert or not self.in_flight[slot]:
            raise ValueError('Completing a slot that is not loading this expert')
        self.in_flight[slot] = False
        self.index[expert] = slot
        self.clock += 1
        self.last_used[slot] = self.clock

    def abort(self, expert, slot):
        """The copy failed or was cancelled: return the slot without publishing."""
        if self.owner[slot] != expert or not self.in_flight[slot]:
            raise ValueError('Aborting a slot that is not loading this expert')
        self.in_flight[slot] = False
        self.owner[slot] = -1
        self.free.append(slot)

    def release(self, experts):
        """Free resident experts that are not protected."""
        for expert in experts:
            slot = self.index[int(expert)]
            if slot < 0:
                continue
            if self.protected[slot]:
                raise ValueError(f'Expert {expert} is protected')
            self._release(int(expert))

    def _release(self, expert):
        slot = int(self.index[expert])
        self.index[expert] = -1
        self.owner[slot] = -1
        self.protected[slot] = False
        self.free.append(slot)

    def _victims(self, count):
        """[(slot, expert)] of the `count` least wanted evictable residents."""
        candidates = np.flatnonzero((self.owner >= 0) & ~self.in_flight & ~self.protected)
        if not len(candidates):
            return []
        experts = self.owner[candidates]
        # Already-computed layers first (horizon); then least wanted (highest early-R* rank); then oldest use.
        upcoming = (experts // self.experts >= self.horizon) if self.horizon is not None else np.zeros(len(experts), bool)
        wanted = -self.priority[experts] if self.priority is not None else np.zeros(len(experts))
        order = np.lexsort((self.last_used[candidates], wanted, upcoming))
        return [(int(slot), int(self.owner[slot])) for slot in candidates[order[:count]]]

    # -- invariants (used by tests and verification) -------------------------------------------------------
    def check(self):
        resident = np.flatnonzero(self.index >= 0)
        slots = self.index[resident]
        if len(set(slots.tolist())) != len(slots):
            raise AssertionError('Two experts share a slot')
        if not np.array_equal(self.owner[slots], resident):
            raise AssertionError('owner does not mirror index')
        if np.any(self.in_flight[slots]):
            raise AssertionError('A published expert is still in flight')
        used = set(np.flatnonzero(self.owner >= 0).tolist())
        if used & set(self.free) or len(self.free) != len(set(self.free)):
            raise AssertionError('free list overlaps used slots or has duplicates')
        if len(used) + len(self.free) != self.capacity:
            raise AssertionError('Slots leaked')
        if np.any(self.protected & (self.owner < 0)):
            raise AssertionError('A free slot is protected')
