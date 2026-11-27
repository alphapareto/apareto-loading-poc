# AGENTS.md — how to work in this repository

## What this project is

apareto-loading-poc serves a large Mixture-of-Experts model (Qwen3.6-35B-A3B, 40 MoE layers × 256 experts =
10,240 *seats*) on a single 16 GB GPU, smaller than the model's expert weights, by making expert residency
**phase-dependent**:

- **prefill** runs dense, with every expert available (resident in VRAM, or streamed from host RAM / disk);
- at the end of prefill, **Rankfield (R\*)** turns features captured during that prefill into a per-layer
  expert ranking, and a budget cuts it into the request's **keep-set**;
- **decode** runs on that keep-set; a router pick outside it is dropped, except the at most one per token and
  layer that the **rescue rule** restores. While a request thinks, R\* re-reads the state every 1,056 tokens
  (*midrun*) and the keep-set is updated; at `</think>` it selects the answer phase's keep-set.

Rankfield (R\*) is a two-tower retrieval network (dual encoder) with an aggregation head that predicts a ranking
of the model's experts for a request. It ships as **data** (per-phase bundles) plus the NumPy reference that
applies it. This repository is the serving system around it: expert packing, residency tiers, streaming,
selection at runtime, concurrent decode, and the vLLM integration. The release serves and measures one benchmark,
LiveCodeBench v6.

## Layout

```
apareto_loader/            core library: config, store and layer store, bank, selector adapter, taps, think mode
apareto_loader/vllm027/    the vLLM 0.27 plugin: loader, model, worker, scheduler, middleware, MoE kernels, nudging (ladder)
  cpu_experts_src/         C++/CUDA sources for the CPU-served experts of the rescue (JIT-built on first use)
config/                    profiles.json: server options and R* bundles of the code profile and its 16/12/8 GB variants; ladder.json: the nudging
                           ladder; vllm-0.27.1-stock.sha256: the reference stock-vLLM manifest
scripts/                   entry points: server launcher, store builders, environment setup, run statistics
  serve.py                 start the server for one profile (wraps serve_bank.py)
  rstar_emit.py            the R* reference implementation, with the per-layer quota — the contract (see Rules)
setup/                     one-time setup: install.sh (everything), build_stores.sh (expert stores)
docs/                      INSTALL, HOW-IT-WORKS, SELECTION, MEMORY, RUNNING, RESULTS, LIMITS
runners/lcb/               LiveCodeBench v6: Red Hat's lighteval recipe against the server
tests/                     gates; each prints PASS/FAIL per check and exits nonzero on any failure
artifacts/                 R* bundles (code/think, code/midrun, code/answer) — read-only inputs
apareto_loader/paths.py    where everything that is not code or R* lives: $APARETO_HOME (default ~/.apareto) holds
                           the venvs, caches, results/, and by default the checkpoint and stores
```

`vllm027/` is this project's own code, named after the vLLM version it targets. vLLM itself is a pinned pip
dependency and is never vendored or patched.

## How it plugs into vLLM

- `scripts/bootstrap_venv.sh` builds `$APARETO_HOME/venv` with **stock vLLM 0.27.1** and the exact pins in
  `requirements.lock`, then records a SHA-256 manifest of the installed vLLM tree.
- `scripts/install_project.sh` installs this package editable (`--no-deps`). `pyproject.toml` declares a
  `vllm.general_plugins` entry point, so vLLM registers the apareto loader, model, worker, scheduler and
  middleware at startup. They take effect only when the launcher passes the matching flags
  (`--load-format apareto_bank`, `--worker-cls …`, `--scheduler-cls …`, `--middleware …`,
  `--logits-processors …` for nudging).
- Servers check the stock manifest at boot and refuse a modified vLLM.
- Verification code (an audited worker and scheduler) is not in this tree. `scripts/serve_bank.py` takes such
  classes by name (`--audit --worker-cls … --scheduler-cls …`); keep audit code out of the package.
- Only thinking requests are served: the middleware and the worker refuse thinking-off, `select=false`,
  other-budget and unselectable requests with an error, and the server keeps serving.
- The one kernel-level change, `vllm027/moe_slots.py`, JIT-builds its own FlashInfer module from patched
  copies of three SM120 sources; stock FlashInfer is untouched.

## Rules

- **Never modify the installed vLLM or anything in site-packages.** Every change lives in this package. If a
  change seems to need a vLLM edit, find the extension point instead, or stop and raise it.
- **`scripts/rstar_emit.py` is the contract.** The in-engine selector must reproduce it bit for bit, including
  the per-layer quota and its interpolation between listed budgets (`tests/test_quota.py`). If you believe the
  reference is wrong, stop and raise it; do not "fix" it.
- **R\* bundles are opaque inputs.** Use them exactly as shipped: do not regenerate, modify,
  prune or re-encode them, and do not change their dtype. Check them against `MANIFEST.sha256` before use.
- **Describe R\* only by its contract.** In code, comments and docs, describe R\* with the sentence above and
  its interface (input features, output ranking, shapes, bit-exact requirements). Do not speculate about, or
  document, how bundles were produced.
- **Decode runs the keep-set plus the rescue rule, nothing else.** A router pick outside the keep-set is
  dropped unless the rescue rule chooses it (at most `cap` per token and layer; the rule is in
  `config.cpu_rescue_spec` and checked offline by `tests/check_rescue_log.py`). Decode loads experts only at the
  refreshes and the `</think>` swap. Enforcement and residency checks stay on in production; the routing log is
  verification-only.
- **The only checkpoint requirement is geometry: 40 layers × 256 experts.** Do not gate on checkpoint name,
  quantization method or producer.
- **Never write into an existing output path.** A new run gets a new tag; results are append-only.
- **Flush per item.** Long generations write each row as it arrives; never buffer to the end.
- **Separate correctness gates from benchmarks.** Gates are exact (dispatched experts ⊆ keep-set, peak memory,
  selector parity) and run on short smokes. Published quality and speed numbers use at least 3 fresh boots
  and report their spread.
- **Report failures verbatim.** If a gate fails, say which, with the numbers. Do not average a regression away.
- **Label claims:** **measured** (you ran it, on how many boots), **predicted** (a formula says so),
  **assumed** (nothing has checked it yet).

## Hardware and environment

- One SM120 GPU per server, default profile sized for 16 GB (developed on RTX 5070 Ti, measured on an i7-9700K / Z390 desktop, PCIe 3.0 x8); TP=1, PP=1, up to three requests decoding at
  once per server.
- Host RAM for the whole expert store (17 GiB pinned) and paused requests' KV (27 GiB free per server, docs/MEMORY.md); NVMe for
  the ~18 GB expert layer store built from the checkpoint by `scripts/build_layer_store.py`.
- The repository holds code and R\* bundles only. `scripts/local_env.sh` keeps every cache (HF, torch, triton,
  vLLM, flashinfer, uv, tmp) in `$APARETO_HOME/cache`; venvs and run outputs live in `$APARETO_HOME` too, never in
  the tree. Servers expect it to be sourced.

## Code and comment style

- Comments explain the code in front of the reader and stand on their own. No references to tickets, run
  ids, private documents, dates, people or decision logs.
- Keep the plugin surface small: a new vLLM hook needs a line in the docs saying why.
- Entry points live in `scripts/` and `runners/`; library code goes in `apareto_loader/` where tests can
  import it.
