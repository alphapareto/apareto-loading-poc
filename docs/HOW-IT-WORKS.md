# How it works

## 1. The problem

Qwen3.6-35B-A3B has 40 MoE layers, each with 256 routed experts: 10,240 (layer, expert) **seats**. In the NVFP4
checkpoint each expert is 1,769,496 bytes, so the routed experts alone are **16.875 GiB**. The rest of the
model (attention, gated-DeltaNet, shared experts, embeddings) adds about 4 GiB. A conventional server keeps
every expert resident, because any token can route to any of them. On a 16 GB card the model does not load.

## 2. Phase-dependent residency

Prefill and decode use the experts differently:

| | prefill | decode |
|---|---|---|
| tokens per step | a whole chunk of the prompt | one per request |
| experts touched | effectively all of them | the ones this request keeps routing to |
| each expert is used | once per chunk | on every step, for thousands of steps |

So each phase gets its own residency:

```
 request:  PREFILL ──▶ SELECT ──▶ DECODE (think) ── every 1,056 tokens: REFRESH ──▶ </think> ─▶ SWAP ─▶ DECODE (answer)
 experts:  dense,      static +   the think          probe, re-read, blend,                  R* on    the answer
           all of      midrun     keep-set           cut; load only what                     the      keep-set
           them        R*                            the new cut adds                        answer
```

1. **Prefill is dense.** Every layer runs with all 256 experts. The ones that are not in the GPU's expert bank
   are copied in from a pinned host-RAM tier as the layer runs. Quality during prefill is the stock model's.
2. **Select.** At the end of prefill, activation taps at 16 sites are read over 8 spacer rows (§3). Rankfield
   (R\*) turns them into two rankings of all 10,240 seats: the **static** think ranking, kept for the whole
   request, and a **midrun** ranking. The think keep-set is the budget's cut of their blend,
   `(1 − w) · static + w · midrun` (w = 0.5). A budget of **5,018 seats (49%)**, cut per layer, gives the keep-set.
3. **Decode on the keep-set.** Each step the router picks its top-8 experts as usual. Picks outside the keep-set
   are dropped, except for the **rescue** (§4): per token and layer at most one dropped pick, chosen by a fixed
   rule, is computed anyway.
4. **Refresh every 1,056 thinking tokens.** The spacer is run once more as a probe under the live keep-set and
   rolled back. The midrun bundle reads it, its ranking is blended with the same static ranking, and the cut
   becomes the new keep-set. Only the experts it adds are loaded (typically a few dozen).
5. **Think, then answer.** When `</think>` is sampled, the spacer runs once more as a probe after `</think>`; the
   answer bundle reads it and selects the *answer* keep-set, loaded in one swap.

Prefill, the probes and the swaps are the only places experts are loaded; decode itself never loads an expert.

## 3. Rankfield (R\*)

Rankfield (R\*) is a two-tower retrieval network (dual encoder) with an aggregation head that predicts a ranking
of the model's experts for a request.

- **Input:** the features captured at a read point: 16 activation taps × 8 rows, fp16 (180,224 values).
- **Output:** a ranking of the 10,240 seats. A per-layer quota cut gives the keep-set for a budget.
- **Bundles:** one per read point, in `artifacts/code/`: `think` (the static ranking, read after prefill),
  `midrun` (read after prefill and at every refresh), `answer` (read after `</think>`). Each is checked against
  its `MANIFEST.sha256` at boot.
- **Contract:** `scripts/rstar_emit.py` is the NumPy reference implementation. The in-engine selector calls it,
  and the blend `(1 − w) · static + w · midrun` is taken on its aggregated rankings before the cut.
  [SELECTION.md](SELECTION.md) walks through the computation and the code, file by file.
- **Per-layer quota:** a bundle's `layer_quota.json` lists the quota at a few budgets. At any other budget the
  reference interpolates (`interpolate_quota`): per-layer shares are interpolated linearly in the budget between
  the listed ones, with the equal split as anchor at 0 and at 10,240 seats, and rounded so the quota sums to the
  budget exactly. `tests/test_quota.py` checks every budget from 1 to 10,240.
- **Early prediction (speed only).** During prefill, layer 4's entry already has the taps hl2, hl4 and kvl3.
  The think bundle, read on those three taps alone, predicts the keep-set there, and the bank keeps the predicted
  experts while prefill evicts. That makes the load after prefill smaller. It never changes the keep-set.

**The spacer read.** The last prompt chunk is scheduled with a short fixed spacer appended (8 tokens). The taps
are read on the spacer rows. Afterwards the spacer is rolled back exactly: its attention KV is never read again,
and the gated-DeltaNet recurrent state is restored. The first token is sampled from the last prompt row, as
without a spacer. The refresh and post-think probes work the same way in the middle of decode: one extra
forward of the spacer under the live keep-set, read, rolled back.

## 4. The rescue rule

At every layer the router picks 8 experts for a token. The picks that are in the request's keep-set are computed;
the others are dropped. The rescue rule is the one exception: among the dropped picks of a token at a layer, the
most important one is computed anyway, if it is important enough. So a token never loses more than it has to of
the expert its router wanted most, and never gains more than one expert per layer from outside the keep-set.

**The rule.** For each decode token and layer, every dropped pick gets a score, the routing weight `w` the router
gave it times a factor. The pick with the highest score is rescued if that score is above the threshold `x`; up
to `cap` picks, highest first. A rescued pick is computed with its routing weight, like a kept one.

| phase | factor | with the default settings, a dropped pick is rescued when |
|---|---|---|
| think, expert inside the request's static think cut | 1 | `w` > 0.00014: practically always |
| think, expert outside the static think cut | `gout` (0.001) | `w` > 0.14 (`x / gout`), and no pick inside the cut is there to take the place |
| answer | `x / ax` | `w` > 0.12 (`ax`) |

The **static think cut** is the keep-set the think network alone gives at the end of prefill, before the midrun
network is blended in and before any refresh ([SELECTION.md §6](SELECTION.md#6-where-the-worker-uses-the-networks)).
During think the keep-set moves away from it at every refresh. A dropped pick inside the static cut is therefore
an expert the first prediction wanted and a later one removed; the rule trusts those, and trusts an expert that
no prediction wanted only when the router gives it a large weight. During the answer there is no such reference
set, and the rule is a plain weight threshold.

Example, during think: the router gives a token the dropped picks A (weight 0.02, inside the static cut) and B
(weight 0.30, outside it). A scores 0.02 and B scores 0.30 × 0.001 = 0.0003. Both are above 0.00014, A is higher,
and with `cap` 1 only A is computed. Had A been in the keep-set, B would have been the rescue.

**The parameters** (`--cpu-rescue x=0.00014,ax=0.12,gout=0.001,cap=1,phases=think+answer`; `off` turns the rule
off, and every pick outside the keep-set is then dropped):

| item | default | meaning |
|---|---|---|
| `x` | 0.00014 | the score a dropped pick must exceed. During think it is also the smallest routing weight at which an expert inside the static cut is rescued |
| `gout` | 0.001 | think only: the factor for experts outside the static cut. Smaller makes them harder to rescue (weight above `x / gout`); 1 treats them like the others |
| `ax` | 0.12 | answer only: the routing weight a dropped pick must exceed |
| `cap` | 1 | how many dropped picks may be rescued per token and layer, 1 to 8 |
| `phases` | `think+answer` | the phases in which the rule applies; with more than one request at a time it must be both |

The server requires 0 < `x` ≤ `ax` < 1 and 0 < `gout` ≤ 1. Only the defaults have been measured.

**Where a rescued expert is computed.** On the GPU when the expert bank happens to hold it. Otherwise two reserved
CPU cores compute it from the expert's copy in host RAM, inside the decode step (`vllm027/cpu_experts.py`,
`--cpu-experts`, [RUNNING.md](RUNNING.md#the---cpu-experts-setting)); if the CPU has not answered within its time
limit, the GPU computes it from the host copy itself. Prefill and the probes never rescue. The rule needs every
expert's host copy at a fixed address, so the host tier must hold the whole store (17 GiB); the server refuses to
boot otherwise.

**How often it acts.** In the three published runs and the clean-clone run (measured, 6.5 to 6.7 million generated
tokens each): about 16.5% of the router's picks fall outside the keep-set, about 2% of all picks are rescued, and the rest,
about 14.5%, are dropped. A rescue happens at 15.4 to 15.8% of the token-and-layer positions; 27% of the rescues are computed on
the GPU and 73% on the CPU ([RESULTS.md](RESULTS.md#what-the-rescue-did)).

**What is and is not measured.** The rescue is part of the configuration whose LiveCodeBench result is reported.
The score with the rule off has not been measured in this release. `tests/check_rescue_log.py` checks every logged
rescue of a run against the rule (a run with `--routing-log`).

## 5. Nudging

A request may generate 64,000 tokens, thinking and answer together. A request that is still thinking when it
reaches that cap returns no answer and is scored as failed. Nudging keeps that from happening: text is written
into the reasoning to make the model finish it. A soft nudge is a short phrase after which the model continues
freely. The hard nudge, at 60,000 tokens, writes `</think>` itself, so the reasoning phase closes and there is
room left for the answer. It is still a nudge: it cannot make the model stop, only move it on to the answer.

When to nudge, and how hard, is set by the **ladder**: a schedule whose steps get stricter the longer the
reasoning runs. Its first steps nudge softly, and only when the reasoning repeats itself; its last step is the
hard nudge, which is unconditional.

The `code` profile nudges every thinking chat request. Nudging acts on the tokens the model emits and on nothing
else: it does not read or change a keep-set.

**What it measures.** For every request one number is kept, *n\**: the length, in tokens, of the longest
sequence that has occurred at least 4 times (`repeats`) in what the request has generated so far, counted up to
320 tokens (`maxn`). Reasoning that goes round in a loop has a large n\*; reasoning that moves on has a small
one. n\* never decreases during a request.

**The ladder.**

| generated tokens | nudges when | writes |
|---|---|---|
| from 40,000 | n\* is at least 100 | soft nudge: `I will output` |
| from 45,000 | n\* is at least 75 | soft nudge: `All set.` |
| from 50,000 | n\* is at least 60; the bar falls by 2 per 1,000 tokens and stops at 50 (at 55,000) | soft nudge: `Ready.` |
| at 60,000 | always | hard nudge: `Done.` and `</think>`; the reasoning phase closes and the answer phase starts |

A soft nudge is a few tokens written into the reasoning as if the model had produced them. Each of the first two rows acts at most once per request,
and the third only in a request that has had no nudge yet, so a request gets at most two soft nudges. Nudging stops
for a request as soon as the model emits `</think>` itself.

**Where the text is written.** Not in the middle of a line or of a code block. A soft nudge waits for the next line
end outside a code block; after 1,500 tokens without one (`nudge_wait`) it takes any line end, and after 200 more
(`wait_cap`) it starts a new line itself. The hard nudge at 60,000 waits for a line end outside a code block for
at most 1,200 tokens (`final_wait`), then closes the open code block and ends the reasoning. The reasoning
therefore ends by about token 61,200, which leaves at least 2,000 tokens for the answer (`min_answer_reserve`).
A request whose own token limit is too small for that reserve is not nudged.

**How often it acts.** In the three published runs the hard nudge at 60,000 tokens was reached in 43, 39 and 51
of the 175 requests, and 24, 21 and 29 soft nudges were written (measured, 1 boot per seed;
[RESULTS.md](RESULTS.md#what-the-nudging-did)). So about a quarter of the requests think until the hard nudge ends
their reasoning.

**What the hard nudge does not fix.** Ending the reasoning does not always end the rambling. After a hard nudge
the model often carries on in the answer, now inside code comments: the answers after a hard nudge are 2,600 to
4,000 tokens long at the median, against about 650 for the other requests, and in about half of them (20 to 25
per run) the answer runs into the 64,000-token cap and is cut off. In the clean-clone run the 41 longest answers
consist of 62% comment lines, against 35% for the rest, and 14 of the 41 were solved, against 115 of the other 134
([RESULTS.md](RESULTS.md#what-the-nudging-did)). A hard nudge gives the request a chance to answer; it does not
make a request that was going round in circles solve its problem.

**What is and is not measured.** Nudging is part of the masked configuration whose LiveCodeBench result is
reported; the dense baseline ran without it ([RESULTS.md](RESULTS.md)). The masked configuration without
nudging has not been measured in this release, and it is a deviation from Red Hat's recipe
([RUNNING.md](RUNNING.md#livecodebench-v6)).

**Settings and records.** `config/ladder.json` is the ladder, and the whole configuration: the rows above are its `gates`, `ramp`
and `final` entries, and the waits are the fields named in the text. `--ladder <file>` serves another one and
`--ladder off` turns nudging off. Every action is written to `<tag>_selection/ladder.jsonl` with the request it belongs to
([RUNNING.md](RUNNING.md#the-nudging-record)).

## 6. Memory tiers and concurrency

| tier | holds | `code` profile |
|---|---|---|
| GPU expert bank | expert slots in VRAM | 5,256 slots, 8.66 GiB |
| GPU KV cache | attention KV + gated-DeltaNet state of all live requests | 1.40 GB, 65,737-token context |
| host expert tier | pinned host-RAM copy of every expert | 17 GiB (the whole store) |
| disk | the layer store (`$APARETO_HOME/stores/layer_store/experts.apkl`) | read at startup |

[MEMORY.md](MEMORY.md) gives the measured GPU and host figures item by item.

Three requests decode at once (`max_num_seqs` 3). Every token row is dispatched under its own request's
keep-set; the bank holds the union of the live keep-sets, and what does not fit (the union is larger than 5,256
slots) is read by the MoE kernel from the pinned host copy. A request's output depends only on its own keep-sets;
concurrency changes only the speed.

### Pause and resume

**Why.** The requests share one KV cache. It holds 66,267 tokens, and a single request may grow to 65,737. Three
long requests do not fit in it together, so when a decoding request needs a new KV block and none is free, one
request has to give way until another has finished.

**What stock vLLM does, and why that is not used.** vLLM preempts a request in that situation: it frees the
request's KV and, when the request is scheduled again, computes its whole context again from the first token.
Here that would be wrong as well as slow. A request's state was built during decode with its keep-sets, which
changed at every refresh; a recomputation would run all of it under one set of experts and give a different
state. So the state is kept, not recomputed.

**What happens instead.**

1. vLLM chooses the request to give way, by its own rule: the last one in its running list, which is the most
   recently admitted.
2. Before its blocks are freed, the request's state is copied to pinned host RAM: the attention KV of every token
   it has processed and its gated-DeltaNet state (`vllm027/concurrent_pause.py`, `vllm027/kv_offload.py`).
3. The request waits in vLLM's queue. The other requests keep decoding.
4. When enough blocks are free again, the request gets new blocks, the bytes are copied back, and it decodes on
   from the token where it stopped. The copy is byte for byte, so the request continues exactly as if it had
   never been paused.

A request is not paused during prefill: a prefill runs alone and is admitted only when its whole prompt fits. At
most two of the three requests can be paused at the same time, which is why the host arena has room for two
requests' KV ([MEMORY.md](MEMORY.md#host-ram)). With `--max-num-seqs 1` nothing is paused and there is no arena.

**How often, and what it costs.** In the three published runs and the clean-clone run (measured,
[RESULTS.md](RESULTS.md#what-pausing-did)): 186 to 197 pauses per run, each one resumed; 123 to 129 of the 175
requests are paused at least once; a paused request holds 16,000 tokens on average, 32,000 at most; and 65 to 68
GiB are copied to host RAM and back over a run. The cost is waiting, not copying: a paused request waits for
about 10,600 scheduling steps, between two and a half and three minutes at the runs' step rate, and a decode step carries
1.93 to 1.97 requests on average, not three (both computed from the server's counters). With the benchmark's
long reasoning the server therefore decodes about two requests at a time. Short requests are not affected: they
fit together.

**How it is checked.** In the audited runs a pause is forced every 1,024 tokens and every copy back is read
again and compared bit for bit with what was copied out ([RESULTS.md](RESULTS.md#correctness-gates-on-this-release)).

## 7. The vLLM integration

vLLM 0.27.1 is installed as a stock wheel and never modified. Every component is in `apareto_loader/vllm027/`,
registered through the `vllm.general_plugins` entry point and selected by launcher flags:

| component | flag | role |
|---|---|---|
| loader | `--load-format apareto_bank` | loads dense weights from the checkpoint and experts from the stores; sizes the bank |
| model | `--hf-overrides '{"architectures": ["AparetoBankQwen3_5Moe"]}'` | slot-indexed MoE layers: a `[expert] → slot` remap per layer, taps, per-request keep-set masks, the rescue ring |
| worker | `--worker-cls …bank_worker.BankWorker` | per-phase residency: dense prefill, selection, refreshes, the `</think>` swap, decode checks |
| scheduler | `--scheduler-cls …scheduler.AsyncBankScheduler` | async scheduling with the spacer and probe lookahead; pause and resume |
| middleware | `--middleware …middleware.BankControlMiddleware` | per-request control (`vllm_xargs`): refuses thinking-off, `select=false` and other-budget requests; attaches the nudging ladder |
| logits processor | `--logits-processors …ladder:LadderLogitsProcessor` | nudging ([§5](#5-nudging)) |
| MoE kernel | `vllm027/moe_slots.py` | a separately built FlashInfer module that indexes resident slots and host copies |
| CPU experts | `vllm027/cpu_experts.py` | the rescue's CPU path (C++/CUDA sources in `cpu_experts_src/`, built on first use) |

`scripts/serve_bank.py` is the launcher (it builds the vLLM options), and `scripts/serve.py` runs it with a
profile. At boot it checks the stock-vLLM manifest, the stores against the
checkpoint and the bundles against their manifests, then serves one warm-up request before writing `ready.json`.

## 8. What is checked, and when

- **Every boot:** vLLM tree = stock manifest; store header = checkpoint; bundles = their manifests; bank + KV +
  activation reserve + margin fit the free GPU memory, to the byte; the rescue's whole-store host tier; one
  warm-up request.
- **Every selection:** the keep-set has exactly the budget's size and the quota's per-layer counts.
- **Offline gates** (`tests/`): the quota at every budget (`test_quota.py`), the bundles (`check_bundles.py`),
  the CPU-expert and rescue kernels against their references (`test_cpu_experts.py`), and a
  served run's rescues against the rule (`check_rescue_log.py`, on a run booted with `--routing-log`).
