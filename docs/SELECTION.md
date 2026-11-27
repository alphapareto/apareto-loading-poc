# How the keep-sets are computed

This release is a proof of concept for one idea: **from a request's prefill, it is possible to predict which
experts will contribute to solving it.** This page follows that prediction through the code: what is read from
the model, how Rankfield (R\*) turns it into a ranking of the experts, how the ranking becomes a keep-set, and
where decode applies it. Every step names the file and function that does it, so you can read along.

For the overall design see [HOW-IT-WORKS.md](HOW-IT-WORKS.md). For the numbers see [RESULTS.md](RESULTS.md).

## 1. The path in one table

| step | what happens | file, function |
|---|---|---|
| 1 | Hooks copy 16 activations ("taps") on 8 rows into a feature vector | `apareto_loader/vllm027/tap_capture.py`, `TapCapture` |
| 2 | Those 8 rows are a short fixed spacer appended to the prompt and rolled back afterwards | `apareto_loader/vllm027/spacer.py`, `SpacerRollback` |
| 3 | R\* maps the feature vector to an aggregated ranking of all 10,240 seats | `scripts/rstar_emit.py`, `Selector.embed`, `.retrieve`, `.fuse` |
| 4 | A per-layer quota cuts the ranking into a keep-set of exactly `budget` seats | `scripts/rstar_emit.py`, `Selector.quota`, `.cut` |
| 5 | The worker runs steps 3 and 4 at the end of prefill, at every refresh and at `</think>` | `apareto_loader/vllm027/bank_worker.py`, `_think_select`, `_midrun_refresh`, `_post_think_select` |
| 6 | The keep-set is written into every MoE layer and the experts it adds are loaded | `bank_worker.py`, `_decode_set`, `_publish` |
| 7 | Each decode step drops router picks outside the keep-set, except the rescue | `apareto_loader/vllm027/bank_model.py`, `BankRunner._apply_quant_method` |

`apareto_loader/selector.py` (`RuntimeSelector`) is the thin adapter between the worker and the reference: it
validates the input, calls `scripts/rstar_emit.py`, checks the result against the budget and quota, and hashes
it for the run record. The reference itself is plain NumPy and imports neither torch nor vLLM.

## 2. The input: features

A **seat** is one (layer, expert) pair; the model has 40 × 256 = 10,240 of them.

The features are 16 taps, each read on 8 rows (`apareto_loader/taps.py`: `TAPS`, `DIMS`, `ROWS`):

| taps | where they are read | width |
|---|---|---|
| `hl2`, `hl4`, `hl8`, `hl14`, `hl20`, `hl27` | the residual stream entering decoder layer N (the input of its first layer norm) | 2,048 |
| `kvl3`, `kvl7`, … , `kvl39` (every fourth layer) | the key and value projections of full-attention layer N | 1,024 |

Concatenated in that order, row-major, they give 6 × 8 × 2,048 + 10 × 8 × 1,024 = **180,224** fp16 values.
`TapCapture.attach` registers the forward hooks; a small Triton kernel copies the 8 rows inside the compiled
graph, and `TapCapture.collect` returns the vector. A read with a missing row or a non-finite value is rejected.

**Which 8 rows.** The rows are a fixed spacer of 8 tokens (`spacer_token_ids` in each bundle's
`read_contract.json`). There are three read points:

| read | when | spacer placed | model state during the read |
|---|---|---|---|
| prefill read | end of prefill | after the last prompt token, in the same forward as the last prompt chunk | dense, every expert |
| refresh probe | every 1,056 thinking tokens | after the last generated token | the live think keep-set |
| post-think probe | when `</think>` is sampled | after `</think>` | the live think keep-set |

After each read the spacer is rolled back exactly (`spacer.py`; for the probes `bank_worker._probe_forward`): its
attention KV is never read again and the gated-DeltaNet state is restored. Generation continues as if the
spacer had never been there.

## 3. The network: features to ranking

Rankfield (R\*) is a two-tower retrieval network (dual encoder) with an aggregation head that predicts a ranking
of the model's experts for a request. It is a small feed-forward network with three stages, run once per read:

```
 features x ──▶ QUERY TOWER ──▶ q ──▶ SCORING vs ITEM TOWER ──▶ top-k ──▶ AGGREGATION HEAD ──▶ ranking [40, 256]
 180,224        16 branches     3,200   dot product with n item              weighted sum of the
                norm → linear           embeddings (keys)                    selected output rows
                → L2 norm                                                    (values) + prior
```

| stage | layer | operation | parameters (per bundle) |
|---|---|---|---|
| query tower | input normalisation, one per tap | `(x_t − mean_t) / scale_t` | `mean_t`, `scale_t`: one value per input |
| | linear projection, one per tap | `· weight_t`, no bias | `weight_t`: 16,384 × 200 (`hl` taps), 8,192 × 200 (`kvl` taps) |
| | L2 normalisation, per tap | unit length | none |
| | concatenate the 16 branches, L2 normalisation | query embedding `q`, 3,200 values | none |
| item tower | embedding table | `n` item embeddings `K`, `[n, 3200]`, unit length | `embedding` |
| scoring | dot product | `s = K · q`, one score per item | none |
| | top-k | keep the `top_k` highest scores | `top_k` |
| aggregation head | output table | one output row `V_i` `[40, 256]` per item | `ranking` |
| | weighted sum | `Σ a_i · V_i + a_prior · prior`, weights normalised to sum to 1 | `prior`, `prior_weight`, `weighting` |

The query tower has 36 million weights (16 projection matrices). The item tower and the head together hold
`n × (3,200 + 10,240)` values: 5.4 million in the think network (`n` = 401), 100.8 million in the midrun
network (7,498) and 5.0 million in the answer network (375).

Stage by stage, with `x` the 180,224 features (`scripts/rstar_emit.py`, class `Selector`; the server calls this
file directly, so what you read there is what runs):

1. **Query tower** (`embed`). The input is split into its 16 taps. Each tap goes through its own branch:
   normalisation, a linear layer to 200 values, and L2 normalisation:

   `e_t = unit( ((x_t − mean_t) / scale_t) · weight_t )`

   The 16 branch outputs are concatenated and normalised again: `q = unit([e_1, …, e_16])`.

2. **Scoring against the item tower** (`retrieve`). The item tower is an embedding table in the same 3,200-value
   space as `q`. Every item is scored by its dot product with the query, `s = K · q`, and the `top_k` items
   with the highest scores are selected. This is the dual-encoder step: query and items meet only through this
   dot product.

3. **Aggregation head** (`fuse`). Each item has an output row `V_i`, a ranking of all 40 × 256 seats. The head
   mixes the selected rows, as a sparse attention layer would, and adds a prior:

   `a_i = max(s_i, 10⁻⁶)`  (`weighting = "sim"`; 1 for `"uniform"`),  `a_prior = prior_weight · mean(a)`

   `output = ( Σ_i a_i · V_i + a_prior · prior ) / ( Σ_i a_i + a_prior )`

The output is `[40, 256]` float64: for every seat a predicted rank, lower meaning the expert is more likely to
be needed. `RuntimeSelector.fused` returns exactly this array. The cut in section 4 is not part of the network;
it turns the prediction into a keep-set of a chosen size.

## 4. Ranking to keep-set: the cut

`Selector.cut(aggregated, budget)` keeps, in each layer, the `quota[layer]` seats with the lowest aggregated
rank. Ties are broken by the bundle's fixed `tie_break` order (`np.lexsort`), so the cut is deterministic.

`Selector.quota(budget)` gives the per-layer counts, which always sum to `budget`:

- a bundle with `layer_quota.json` uses the listed quota at the budgets it lists (3,482, 4,301, 4,608, 5,018,
  5,734 and 6,451 in the think and midrun bundles) and `interpolate_quota` at any other budget;
- a bundle without it (the answer bundle) splits the budget equally over the 40 layers.

`RuntimeSelector.select` and `.cut` raise if a keep-set does not have exactly the budget's size and the quota's
per-layer counts. `tests/test_quota.py` checks the quota at every budget from 1 to 10,240.

## 5. The bundles

A bundle holds the parameters of one network. The `code` profile uses three (`config/profiles.json`):

| bundle | read point | what it produces |
|---|---|---|
| `code/think` | prefill read | the **static** ranking, kept for the whole request |
| `code/midrun` | prefill read and every refresh probe | the **midrun** ranking |
| `code/answer` | post-think probe | the answer ranking |

Files in a bundle, and the arrays `Selector.__init__` loads from them:

| file | arrays | used in |
|---|---|---|
| `query_tower.npz` | `taps`, `dims`, `rows`; per tap the normalisation `mean_<tap>`, `scale_<tap>` and the linear layer `weight_<tap>` | `embed` |
| `item_tower.npz` | the item embedding table `embedding` `[n, 3200]`, the head's output table `ranking` `[n, 40, 256]`, `top_k`, `prior_weight`, `weighting` | `retrieve`, `fuse` |
| `prior.npz` | the head's prior `ranking` `[40, 256]`, the cut's `tie_break` `[40, 256]`, `layers`, `experts` | `fuse`, `cut` |
| `layer_quota.json` | `quotas`: budget → 40 per-layer counts (think and midrun only) | `quota` |
| `read_contract.json` | taps, spacer token ids, phase, read point | checked at boot (`apareto_loader/think.py`, `think_read_contract`) |
| `MANIFEST.sha256` | SHA-256 and size of every file | checked at boot (`RuntimeSelector.manifest_check`) |

Large files are stored as numbered parts (`query_tower.npz.part000`, …); `scripts/artifact_io.py` reads them as
one file. `tests/check_bundles.py` checks every bundle against its manifest and loads it.

The bundles are fixed data. This repository contains the code that applies them and no tooling to produce them.

## 6. Where the worker uses the networks

All of this is in `apareto_loader/vllm027/bank_worker.py`. `B` is the budget (5,018) and `w` the midrun weight
(0.5).

**End of prefill** (`_think_select`), from the prefill read `x`:

```
static     = think.fused(x)                               # kept in the request's state
static_cut = think.cut(static, B)                         # the rescue rule's reference set
think_keep = midrun.cut((1 − w) · static + w · midrun.fused(x), B)
```

`think_keep` is the first keep-set. With `--no-midrun-at-prefill` it is `static_cut`.

**Every 1,056 thinking tokens** (`_midrun_refresh`), from a probe read `p`:

```
think_keep = midrun.cut((1 − w) · static + w · midrun.fused(p), B)
```

`static` is the same array as at prefill; only the midrun half is new. The blend is taken on the aggregated
rankings, before the cut. Both bundles must have the same quota at `B`, which is checked at boot
(`_load_midrun`).

**At `</think>`** (`_answer_swap`, `_post_think_select`), from the post-think probe `q`:

```
answer_keep = answer.select(q, B)
```

If a probe cannot run (it would pass the context limit) or its read is rejected, the current keep-set stays, and
the record says so.

**Early prediction** (`bank_engine.py`, `LateShallowSelector`). Three of the taps (`hl2`, `hl4`, `kvl3`) already
exist when prefill enters layer 4. The think network, run on those three alone, predicts the keep-set there, and
the bank keeps the predicted experts while prefill evicts. This makes the load after prefill smaller and never
changes a keep-set.

## 7. Where decode applies the keep-set

`_decode_set` loads the experts the new keep-set adds, and `_publish` copies the keep-set into the `apareto_keep`
buffer of every MoE layer. With several requests decoding, each has its own row of that buffer.

In `bank_model.py`, `BankRunner._apply_quant_method` runs once per layer and step:

1. the router picks its top-8 experts per token, as in the stock model;
2. `allowed_rows` looks each pick up in the request's keep row;
3. picks outside the keep-set get weight 0, except the at most one per token and layer that the rescue rule
   restores ([HOW-IT-WORKS.md §4](HOW-IT-WORKS.md#4-the-rescue-rule));
4. the MoE kernel computes the remaining picks.

The think-phase rescue scores a dropped pick against `static_cut` from section 6 (`_set_rescue_gain`).

One more component acts on a thinking request, and it is not part of selection: nudging writes
text into long reasoning, and `</think>` itself at 60,000 tokens. It works on the emitted tokens only and never reads or
changes a keep-set; the answer keep-set is selected at `</think>` whether the model or a nudge wrote it
([HOW-IT-WORKS.md §5](HOW-IT-WORKS.md#5-nudging)).

## 8. Looking at a run

Every selection is written to `<tag>_selection/selections.jsonl`, one JSON row each:

| row | written by | useful fields |
|---|---|---|
| `"mode": "think"` | `_think_select` | `features_sha256`, `think_keep_sha256`, `midrun_prefill_moved_from_static`, `early_recall` |
| `"kind": "midrun_swap"` | `_midrun_refresh` | `think_pos`, `probe_features_sha256`, `keep_sha256`, `added`, `dropped`, `moved_from_static` |
| `"kind": "think_swap"` | `_answer_swap` | `answer_keep_sha256`, `answer_moved_from_think`, `answer_loaded` |

`moved_from_static` is the number of seats in the keep-set that are not in the static cut, so it shows how far
the refreshes move the prediction during a long think.

To run a network yourself, outside the server:

```python
import sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from apareto_loader.selector import RuntimeSelector

think = RuntimeSelector("artifacts/code/think")
x = ...                                   # 180,224 float features of one read
ranking = think.fused(x)                  # [40, 256] aggregated ranks
keep = think.cut(ranking, 5018)           # [40, 256] bool, 5,018 True
print(keep.sum(axis=1))                   # the per-layer quota
```

## 9. Trained and held-out problems

R\* is a trained network. 70 of LiveCodeBench v6's 175 problems were part of its training data (*trained*); the
other 105 were kept out of the training data, on purpose (*held out*). The benchmark therefore shows both cases: prompts the network has seen and
prompts it has not. `runners/lcb/trained_problems.json` lists the 70 (66 of them for the answer network), and
[RESULTS.md](RESULTS.md) reports the two groups separately.
