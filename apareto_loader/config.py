"""The bank engine's settings, the CPU-expert and rescue specs, and the boot warm set."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .store import integer


def cpu_experts_spec(text):
    """'cores=4-7,timeout_us=1000,per_pair_us=150,admit=on,absent=cpu' -> dict. The GPU waits timeout_us + per_pair_us
    per (token, expert) pair of the layer's request before computing it itself. absent=gpu keeps absent allowed experts
    on the GPU's zero-copy path (the CPU then serves only rescues, cpu_rescue_spec). cores=auto:N takes the last N
    cores of the server's own affinity."""
    spec = {'cores': '4-7', 'timeout_us': 1000.0, 'per_pair_us': 150.0, 'admit': True, 'absent': 'cpu'}
    for part in filter(None, str(text).split(',')):
        key, _, value = part.partition('=')
        if key == 'cores':
            spec['cores'] = value.replace(';', ',')
        elif key in ('timeout_us', 'per_pair_us'):
            spec[key] = float(value)
        elif key == 'admit' and value in ('on', 'off'):
            spec['admit'] = value == 'on'
        elif key == 'absent' and value in ('cpu', 'gpu'):
            spec['absent'] = value
        else:
            raise ValueError(f"cpu_experts: unknown or malformed item {part!r}")
    return spec


RESCUE_PHASES = ('think', 'answer')


def cpu_rescue_spec(text):
    """'x=0.00014,ax=0.12,gout=0.001,cap=1,phases=think+answer' -> dict: the rescue rule; 'off' -> None.

    Per decode token row and layer, up to `cap` router picks that the keep-set drops are computed anyway, the
    highest-scoring first, if their score is above `x`. During think the score is the routing weight times 1 for an
    expert inside the request's static think cut and `gout` outside it; during answer it is the routing weight times
    x / ax, i.e. the plain weight rule at threshold `ax`. A rescued pick is computed on the GPU when the bank holds
    it, else on the CPU from its host record. Prefill runs every expert anyway; probes never rescue."""
    if str(text).strip() == 'off':
        return None
    spec = {'x': 0.00014, 'cap': 1, 'phases': RESCUE_PHASES, 'gain': 'static_think', 'gout': 0.001, 'ax': 0.12}
    seen = set()
    for part in filter(None, str(text).split(',')):
        key, _, value = part.partition('=')
        if key in seen:
            raise ValueError(f"cpu_rescue: {key} given twice")
        seen.add(key)
        if key in ('x', 'gout', 'ax'):
            spec[key] = float(value)
        elif key == 'cap' and value.isdigit() and 1 <= int(value) <= 8:
            spec['cap'] = int(value)
        elif key == 'phases' and value and set(value.split('+')) <= set(RESCUE_PHASES):
            spec['phases'] = tuple(p for p in RESCUE_PHASES if p in value.split('+'))
        else:
            raise ValueError(f"cpu_rescue: unknown or malformed item {part!r}")
    # gout >= x is allowed: a pick outside the cut is then rescued when w > x / gout and no in-cut pick qualifies.
    if not 0 < spec['x'] <= spec['ax'] < 1 or not 0 < spec['gout'] <= 1:
        raise ValueError("cpu_rescue: needs 0 < x <= ax < 1 and 0 < gout <= 1")
    return spec


@dataclass(frozen=True)
class ExpertConfig:
    """The bank engine's settings (scripts/serve_bank.py builds it; vLLM carries it in --additional-config)."""
    store: str                      # the expert store's source header (stores/expert_store_header.json)
    budget: int                     # experts in every keep-set (of 10,240), cut with the per-layer quota
    layer_store: str                # the kernel-ready layer store the bank and the host tier read
    # R* bundles: the think bundle selects the static think cut from the prefill read and, read on its shallow taps
    # alone, predicts during prefill (speed only); the answer bundle selects the answer mask from a spacer probe at
    # `</think>`, run under the think mask and rolled back.
    think_bundle: str
    answer_bundle: str
    # Midrun think masks: every midrun_every think tokens a spacer probe under the live mask is read by the midrun
    # bundle; the new think mask cuts (1 - w) * static think field + w * midrun field, w = midrun_weight, and only the
    # difference is loaded. None: off. midrun_at_prefill: the first think block is a midrun selection too, from the
    # prefill read.
    midrun_every: int | None = None
    midrun_bundle: str | None = None
    midrun_weight: float = 0.5
    midrun_at_prefill: bool = False
    # The bank's contents at boot (warm_set): this many experts, split evenly over the layers.
    warm_experts: int = 2000
    # Pinned host RAM for the layer-store records (None: the whole layer store).
    host_tier_bytes: int | None = None
    # GPU budget in bytes, no utilization fraction: free memory at startup minus this margin (allocator
    # fragmentation, allocations torch does not see). Activations + CUDA graphs are estimated for the load-time fit
    # check and then measured by vLLM's profiling run.
    gpu_margin_bytes: int = 256 << 20
    activation_reserve_bytes: int = 1536 << 20
    # Paged decode: GPU bank slots below the bank minimum. Experts of the decode set that are not in the bank are
    # read by the MoE kernel from pinned host memory and admitted between steps, least recently routed evicted.
    # None: the bank holds the whole decode set.
    gpu_experts: int | None = None
    # Paged decode, CPU-served experts: 'cores=4-7,timeout_us=1000,admit=on,absent=gpu' (cpu_experts_spec). None: off.
    cpu_experts: str | None = None
    # The rescue rule (cpu_rescue_spec), computed on the CPU path. Needs cpu_experts. None: off.
    cpu_rescue: str | None = None
    # Think-end ladder: a JSON spec (config/ladder.json) the server attaches to every thinking chat request; the
    # ladder logits processor applies it. None: off.
    ladder: str | None = None
    # Verification only: write each request's pre-mask router top-8 per decode step, each pick's fate (kept, dropped,
    # rescued) and its static think cut ($APARETO_HOME/results/<tag>_selection/routing/), for the offline rescue-rule check.
    routing_log: bool = False

    def __post_init__(self):
        integer(self.budget, "expert budget", 1)
        integer(self.warm_experts, "warm experts")
        if not Path(self.store).is_file():
            raise ValueError("Expert store header does not exist")
        if not Path(self.layer_store).is_file():
            raise ValueError("Layer store does not exist")
        for path in (self.think_bundle, self.answer_bundle):
            if not Path(path).is_dir():
                raise ValueError(f"R* bundle directory does not exist: {path}")
        integer(self.gpu_margin_bytes, "GPU margin bytes")
        integer(self.activation_reserve_bytes, "activation reserve bytes")
        if self.host_tier_bytes is not None:
            integer(self.host_tier_bytes, "host tier bytes", 1)
        if self.gpu_experts is not None:
            integer(self.gpu_experts, "GPU experts", 512)   # two full layers: a dense prefill pass
        if self.cpu_experts is not None:
            if self.gpu_experts is None:
                raise ValueError("CPU-served experts need paged decode (gpu_experts)")
            cpu_experts_spec(self.cpu_experts)
        if self.cpu_rescue is not None and cpu_rescue_spec(self.cpu_rescue) is not None:
            if self.cpu_experts is None:
                raise ValueError("The rescue rule runs on the CPU path (cpu_experts)")
        if self.midrun_every is not None:
            integer(self.midrun_every, "midrun interval", 1)
            if self.midrun_bundle is None or not Path(self.midrun_bundle).is_dir():
                raise ValueError("Midrun masks need a midrun bundle directory")
            if not 0 <= self.midrun_weight <= 1:
                raise ValueError("midrun_weight must be in [0, 1]")
        elif self.midrun_bundle is not None:
            raise ValueError("A midrun bundle needs midrun_every")
        if self.routing_log and self.gpu_experts is None:
            raise ValueError("The routing log is written by paged decode (gpu_experts)")
        if self.midrun_at_prefill and self.midrun_every is None:
            raise ValueError("midrun_at_prefill needs midrun masks (midrun_every and a midrun bundle)")
        if self.ladder is not None and not Path(self.ladder).is_file():
            raise ValueError(f"Ladder spec does not exist: {self.ladder}")

    @classmethod
    def from_vllm(cls, vllm_config):
        return cls(**vllm_config.additional_config["apareto"])


def warm_set(layers, experts, count):
    """The bank's contents at boot, before any request: the first experts of every layer, `count` in all, split
    evenly (the first count % layers layers take one more). Which experts sit there only affects the first request's
    speed: prefill is dense and every decode set comes from R*."""
    integer(count, "warm experts")
    if count > layers * experts:
        raise ValueError("Warm set exceeds the experts")
    keep = np.zeros((layers, experts), dtype=bool)
    for layer in range(layers):
        keep[layer, :count // layers + (layer < count % layers)] = True
    return keep
