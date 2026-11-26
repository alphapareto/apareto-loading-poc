"""Request-bound adapter to the NumPy reference of R* (scripts/rstar_emit.py)."""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import time

import numpy as np

from scripts.rstar_emit import Selector

ARTIFACTS = Path(os.environ.get("APARETO_ARTIFACTS", Path(__file__).resolve().parents[1] / "artifacts"))


class SelectionRejected(ValueError):
    """This request's own input cannot be selected; reject it without stopping the engine."""


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@dataclass(frozen=True)
class Selection:
    request_id: str
    prompt_sha256: str
    features_sha256: str
    keep_sha256: str
    budget: int
    keep: np.ndarray
    seconds: float

    def record(self):
        return {"request_id": self.request_id, "prompt_sha256": self.prompt_sha256,
            "features_sha256": self.features_sha256, "keep_sha256": self.keep_sha256,
            "budget": self.budget, "selection_seconds": self.seconds,
            "keep_by_layer": [np.flatnonzero(row).tolist() for row in self.keep]}

    def require_request(self, request_id, prompt_tokens, served_keep):
        if request_id != self.request_id or array_digest(np.asarray(prompt_tokens, dtype="<i8")) != self.prompt_sha256:
            raise ValueError("Selection belongs to another request or prompt")
        served_keep = np.asarray(served_keep)
        if served_keep.dtype != np.bool_ or not np.array_equal(served_keep, self.keep):
            raise ValueError("Served keep-set differs from this request's selector output")


_LOADED = {}


def load_reference(bundle):
    """The reference network of a bundle, loaded once per process. Two bundles with the same query tower (same
    taps, normalisation and weights, value for value) share one copy of it in memory."""
    key = os.path.realpath(str(bundle))
    selector = _LOADED.get(key)
    if selector is None:
        selector = Selector(str(bundle))
        for other in _LOADED.values():
            if other.taps == selector.taps and all(
                    np.array_equal(a, b) for mine, theirs in ((selector.P, other.P), (selector.mu, other.mu),
                                                              (selector.sd, other.sd)) for a, b in zip(mine, theirs)):
                selector.P, selector.mu, selector.sd = other.P, other.mu, other.sd
                break
        _LOADED[key] = selector
    return selector


class RuntimeSelector:
    def __init__(self, artifacts=ARTIFACTS):
        self.artifacts = Path(artifacts)
        self.reference = load_reference(self.artifacts)

    def select(self, request_id, prompt_tokens, features, budget):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Selection requires a nonempty request ID")
        tokens = np.asarray(prompt_tokens)
        if tokens.ndim != 1 or tokens.dtype.kind not in "iu" or len(tokens) < self.reference.rows or np.any(tokens < 0):
            raise SelectionRejected("Selected prompts require at least eight nonnegative token IDs")
        x = np.asarray(features)
        if x.shape != (self.reference.n_features,) or x.dtype.kind != "f" or not np.isfinite(x).all():
            raise SelectionRejected("Missing, nonfinite or malformed selector features")
        budget = self.check_budget(budget)
        started = time.perf_counter()
        keep = self.reference.keep_set(x, budget=int(budget))[0]
        if int(keep.sum()) != budget or not np.array_equal(keep.sum(axis=1), self.reference.quota(int(budget))):
            raise ValueError("Selector output violates its budget or layer quotas")
        keep = np.frombuffer(keep.tobytes(), dtype=np.bool_).reshape(keep.shape)
        return Selection(request_id, array_digest(np.asarray(tokens, dtype="<i8")), array_digest(x),
                         array_digest(keep), int(budget), keep, time.perf_counter() - started)

    def fused(self, features):
        """The aggregated rank field [L, E] before the cut (the midrun refresh blends two of them before cutting)."""
        x = np.asarray(features)
        if x.shape != (self.reference.n_features,) or x.dtype.kind != "f" or not np.isfinite(x).all():
            raise SelectionRejected("Missing, nonfinite or malformed selector features")
        reference = self.reference
        return reference.fuse(*reference.retrieve(reference.embed(x)))[0]

    def cut(self, field, budget):
        keep = self.reference.cut(np.asarray(field, np.float64)[None], int(budget))[0]
        if int(keep.sum()) != budget or not np.array_equal(keep.sum(axis=1), self.reference.quota(int(budget))):
            raise ValueError("Selector output violates its budget or layer quotas")
        return keep

    def check_budget(self, budget):
        if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or not 0 < budget <= self.reference.L * self.reference.E:
            raise SelectionRejected("Invalid selector budget")
        return int(budget)

    def manifest_check(self):
        """Startup check: every file in the bundle is listed in MANIFEST.sha256 with its full SHA-256, and nothing
        unlisted is present. Lines are `<digest>  <file>` with an optional byte size."""
        listed = {}
        for line in (self.artifacts / "MANIFEST.sha256").read_text().splitlines():
            if not line.strip():
                continue
            digest, name, *size = line.split()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or Path(name).name != name:
                raise ValueError(f"Bundle manifest line is malformed: {line!r}")
            listed[name] = (digest, int(size[0]) if size else None)
        present = {p.name for p in self.artifacts.iterdir() if p.is_file() and p.name != "MANIFEST.sha256"}
        if present != set(listed):
            raise ValueError(f"Bundle files differ from its manifest: unlisted {sorted(present - set(listed))}, "
                             f"missing {sorted(set(listed) - present)}")
        for name, (digest, size) in sorted(listed.items()):
            data = (self.artifacts / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest or (size is not None and size != len(data)):
                raise ValueError(f"Bundle file {name} does not match its manifest digest {digest}")
        return {"check": "manifest", "files": len(listed),
                "manifest_sha256": hashlib.sha256((self.artifacts / "MANIFEST.sha256").read_bytes()).hexdigest()}
