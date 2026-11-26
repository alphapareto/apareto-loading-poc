"""Reference implementation of Rankfield (R*). NumPy, no GPU, no vLLM.

Rankfield (R*) is a two-tower retrieval network (dual encoder) with an aggregation head that predicts a ranking
of the model's experts for a request.

Input : a feature vector, the 16 taps of one read concatenated
Output: a boolean keep-set [L, E] with exactly `budget` True entries and a fixed per-layer quota

Network (all weights come from the bundle):
  1 query tower  per tap: (x - mean)/scale -> @ weight -> L2-normalise; concatenate the 16 unit vectors; L2-normalise
  2 item tower   item embeddings (item_tower.npz); score = dot product with the query; keep the top k
  3 aggregation  weighted mean of the top-k items' expert rankings, weights per the bundle's `weighting` field --
                 clipped score ("sim") or equal ("uniform") -- plus the prior ranking at weight
                 prior_weight * mean(weights)
  4 output       per layer keep the quota-many lowest aggregated ranks, ties broken by the bundle's tie-break order

Bundle files (docs/SELECTION.md lists every array):
  query_tower.npz   taps, dims, rows; per tap mean_<tap>, scale_<tap>, weight_<tap>
  item_tower.npz    embedding [n, d], ranking [n, L, E], top_k, prior_weight, weighting
  prior.npz         ranking [L, E], tie_break [L, E], layers, experts
  layer_quota.json  optional: per-layer quotas at the budgets the bundle lists

This file is the contract the vLLM integration must match bit for bit; tests/test_quota.py checks its per-layer
quota, and every served bundle is checked against its MANIFEST.sha256 at boot.
"""
import json, os
import numpy as np

if __package__:
    from .artifact_io import open_artifact
else:
    from artifact_io import open_artifact

ART = os.environ.get("APARETO_ARTIFACTS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"))


def read_layer_quota(artifacts, layers, experts):
    """Optional layer_quota.json: {"quotas": {"<budget>": [L ints], ...}}, the bundle's per-layer cuts at the budgets it
    lists. Absent: every cut is the equal split.
    Present: a cut at a listed budget uses its quota exactly; any other budget uses interpolate_quota()."""
    path = os.path.join(artifacts, "layer_quota.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    quotas = {}
    for budget, quota in d["quotas"].items():
        budget, quota = int(budget), np.asarray(quota, np.int64)
        if quota.shape != (layers,) or int(quota.sum()) != budget or quota.min() < 0 or quota.max() > experts:
            raise ValueError(f"{path}: quota for {budget} must be {layers} counts in [0, {experts}] summing to {budget}")
        quotas[budget] = quota
    if not quotas:
        raise ValueError(f"{path}: no quotas")
    return quotas


def interpolate_quota(quotas, budget, layers, experts):
    """Per-layer quota at a budget the bundle does not list, from the listed ones.

    Each listed quota gives per-layer shares q / B. Uniform shares (1 / layers) are added as anchors at budget 0 and at
    layers * experts (every expert), so the cut becomes the equal split at the ends and the unmasked model at the top.
    The shares at `budget` are interpolated linearly in the budget between the two anchors around it, scaled to
    `budget`, and rounded by largest remainder: floor every layer, then add one to the layers with the largest
    fractional parts (lower layer first on ties) until the sum is `budget`. No layer exceeds `experts`; a layer at the
    cap passes its turn to the next."""
    total = layers * experts
    budget = int(budget)
    if not 0 < budget <= total:
        raise ValueError(f"budget {budget} outside 1..{total}")
    uniform = np.full(layers, 1.0 / layers)
    anchors = [(0, uniform)] + [(b, np.asarray(quotas[b], np.float64) / b) for b in sorted(quotas)] + [(total, uniform)]
    for (lo, s_lo), (hi, s_hi) in zip(anchors, anchors[1:]):
        if lo <= budget <= hi:
            t = (budget - lo) / (hi - lo)
            share = (1.0 - t) * s_lo + t * s_hi
            break
    target = np.minimum(share * budget, experts)
    quota = np.floor(target).astype(np.int64)
    fraction = target - quota
    order = sorted(range(layers), key=lambda l: (-fraction[l], l))
    while quota.sum() < budget:
        for l in order:
            if quota.sum() == budget:
                break
            if quota[l] < experts:
                quota[l] += 1
    return quota


class Selector:
    def __init__(self, artifacts=ART):
        with open_artifact(os.path.join(artifacts, "query_tower.npz")) as source:
            with np.load(source, allow_pickle=False) as b:
                self.taps = [str(t) for t in b["taps"]]
                self.dims = [int(d) for d in b["dims"]]
                self.rows = int(b["rows"])
                self.mu = [b[f"mean_{t}"].astype(np.float64) for t in self.taps]
                self.sd = [b[f"scale_{t}"].astype(np.float64) for t in self.taps]
                self.P = [b[f"weight_{t}"].astype(np.float64) for t in self.taps]
        self.slices, o = [], 0
        for d in self.dims:
            self.slices.append((o, o + self.rows * d)); o += self.rows * d
        self.n_features = o

        with open_artifact(os.path.join(artifacts, "item_tower.npz")) as source:
            with np.load(source, allow_pickle=False) as items:
                self.item_emb = items["embedding"].astype(np.float64)       # [n, d]
                self.item_rank = items["ranking"]                           # [n, L, E], integer ranks as stored
                self.k = int(items["top_k"]); self.alpha = float(items["prior_weight"])
                self.weighting = str(items["weighting"])
        if self.weighting not in ("sim", "uniform"):
            raise ValueError(f"unknown weighting {self.weighting!r} (expected sim or uniform)")

        with np.load(os.path.join(artifacts, "prior.npz"), allow_pickle=False) as pr:
            self.prior = pr["ranking"].astype(np.float64)                   # [L, E]
            self.tie = pr["tie_break"].astype(np.int64)                     # [L, E]
            self.L, self.E = int(pr["layers"]), int(pr["experts"])
        self.layer_quota = read_layer_quota(artifacts, self.L, self.E)

    @staticmethod
    def _nz(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

    def quota(self, budget):
        if self.layer_quota is not None:
            if budget in self.layer_quota:
                return self.layer_quota[budget].copy()
            return interpolate_quota(self.layer_quota, budget, self.L, self.E)
        return np.array([budget // self.L + (1 if l < budget % self.L else 0) for l in range(self.L)])

    def embed(self, x):
        """x: [n_features] or [m, n_features] raw tap concatenation -> unit embedding."""
        x = np.atleast_2d(np.asarray(x, np.float64))
        parts = [self._nz(((x[:, lo:hi] - mu) / sd) @ P) for (lo, hi), mu, sd, P in
                 zip(self.slices, self.mu, self.sd, self.P)]
        return self._nz(np.concatenate(parts, 1))

    def retrieve(self, emb):
        sim = emb @ self.item_emb.T
        idx = np.argsort(-sim, axis=1)[:, : self.k]
        return idx, np.take_along_axis(sim, idx, axis=1)

    def fuse(self, idx, sim):
        # "sim" weights each of the top-k items by its score; "uniform" weights them equally.
        # Each bundle declares its own, so it is read from the bundle rather than fixed here.
        w = np.ones_like(sim) if self.weighting == "uniform" else np.clip(sim, 1e-6, None)
        # Only the top-k rows are converted to float64: the same values as converting the whole table.
        F = np.concatenate([self.item_rank[idx].astype(np.float64), self.prior[None, None]. repeat(len(idx), 0)], 1)
        ww = np.concatenate([w, (self.alpha * w.mean(1))[:, None]], 1)
        return (F * ww[:, :, None, None]).sum(1) / ww.sum(1)[:, None, None]

    def cut(self, fused, budget):
        q = self.quota(budget)
        keep = np.zeros(fused.shape, bool)
        for i in range(fused.shape[0]):
            for l in range(self.L):
                keep[i, l, np.lexsort((self.tie[l], fused[i, l]))[: int(q[l])]] = True
        return keep

    def keep_set(self, features, budget):
        """features -> boolean keep-set [m, L, E]."""
        emb = self.embed(features)
        idx, sim = self.retrieve(emb)
        return self.cut(self.fuse(idx, sim), budget)


if __name__ == "__main__":
    import sys
    s = Selector()
    budget = int(os.environ.get("BUDGET", "5018"))
    quota = s.quota(budget)
    print(f"taps {len(s.taps)}  features {s.n_features}  items {len(s.item_emb)}  "
          f"top_k={s.k} prior_weight={s.alpha} weighting={s.weighting}  budget {budget}  "
          f"quota {quota.min()}-{quota.max()}")
    if len(sys.argv) > 1:
        f = np.load(sys.argv[1])
        k = s.keep_set(f["features"][: int(os.environ.get("N", "4"))].astype(np.float64), budget)
        print("keep per query:", k.reshape(k.shape[0], -1).sum(1))
