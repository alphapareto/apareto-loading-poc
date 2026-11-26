# Installation

`bash setup/install.sh` runs every step below in order. It skips any step whose output already exists, so you
can re-run it after a failure. This page lists the steps so you can run them yourself, or move the large
files elsewhere.

## Prerequisites

| | requirement | why |
|---|---|---|
| GPU | one NVIDIA SM120 card (compute capability 12.0). The default profile is sized for 16 GB, e.g. RTX 5070 Ti / 5080; a smaller card needs a lower `--gpu-experts` ([MEMORY.md](MEMORY.md#a-12-gb-card-simulated)) | the MoE kernels are built for SM120 only |
| driver / toolkit | a CUDA 13 driver; the CUDA 13 toolkit with `nvcc` at `$CUDA_HOME` (default `/usr/local/cuda`) | the CPU-expert server and the MoE slot kernel are compiled on first use |
| CPU | x86-64 with AVX2/FMA/F16C, at least 3 cores per server | the rescue rule's CPU-served experts take 2 cores |
| host RAM | 25 GiB free per server; a 32 GB machine for one server | the whole expert store pinned (17 GiB), plus paused requests' KV; measured peak 25.9 GiB ([MEMORY.md](MEMORY.md)) |
| disk | ~24 GB checkpoint + ~18 GB expert store, ~40 GB free while building, NVMe recommended | prefill reads experts from the store |
| Python | [uv](https://docs.astral.sh/uv/) (recommended) or Python 3.12 | |

The repository holds the code and the R\* bundles (`artifacts/`) only. Everything else goes under
`$APARETO_HOME`, default `~/.apareto` ([apareto_loader/paths.py](../apareto_loader/paths.py)):

| | where |
|---|---|
| the server's Python environment | `$APARETO_HOME/venv` |
| the LiveCodeBench harness's environment | `$APARETO_HOME/venv-lcb` |
| every cache and temporary file (Hugging Face, torch, Triton, FlashInfer, vLLM, uv, tmp) | `$APARETO_HOME/cache` |
| run outputs | `$APARETO_HOME/results` |
| checkpoint, expert stores (unless `APARETO_MODEL` / `APARETO_STORES` point elsewhere) | `$APARETO_HOME/models`, `$APARETO_HOME/stores` |

`scripts/local_env.sh` exports these locations; the scripts source it, and `scripts/serve.py` re-executes itself
under it.

## Steps

### 1. Python environment: stock vLLM

```bash
bash scripts/bootstrap_venv.sh
```

This creates `$APARETO_HOME/venv` with **stock vLLM 0.27.1** and the exact pins in `requirements.lock`. It then
writes `$APARETO_HOME/vllm_stock_manifest.sha256`, the SHA-256 of every file in the installed vLLM tree. `setup/install.sh`
compares that manifest with `config/vllm-0.27.1-stock.sha256` and stops if they differ. The server checks it
again at every boot and refuses to start if the installed vLLM has changed.

### 2. The plugin

```bash
bash scripts/install_project.sh install_<tag>
```

This installs this package editable, with `--no-deps`, so the pinned environment stays as it is.
`pyproject.toml` declares a `vllm.general_plugins` entry point (`apareto`). vLLM loads it at startup and
registers the bank loader, model, worker, scheduler and middleware. They only act when the launcher selects
them, so a plain `vllm serve` in the same venv is unaffected.

### 3. JIT kernels (optional, else on first boot)

Two components are compiled on first use:

- `apareto_loader/vllm027/moe_slots.py` builds its own FlashInfer CUTLASS MoE module from patched copies of
  three SM120 sources. It indexes resident slots instead of all 256 experts. The stock FlashInfer stays as
  installed.
- `apareto_loader/vllm027/cpu_experts.py` builds the CPU-expert server (C++ and CUDA, in
  `cpu_experts_src/`).

`setup/install.sh` compiles both ahead of time (`MAX_JOBS`, default 4). Compiling does not use the GPU. With
`--no-prebuild` they are compiled on the first boot instead, which then takes that much longer. The MoE module is
the slow one: about 20 minutes with `MAX_JOBS=4` (measured once; the CPU-expert server takes about 20 s).

### 4. Checkpoint

```bash
~/.apareto/venv/bin/hf download RedHatAI/Qwen3.6-35B-A3B-NVFP4 --revision a1ff8c8967426709e5528307c14c52c17616b008 \
    --local-dir ~/.apareto/models/Qwen3.6-35B-A3B-NVFP4
```

The installer pins this revision. Any checkpoint with the same geometry, **40 MoE layers × 256 experts**, can be
served; the release is only tested on this one. The published LiveCodeBench comparison refers
to this checkpoint.

### 5. Expert stores

```bash
bash setup/build_stores.sh            # reads $APARETO_MODEL, writes $APARETO_STORES (default ~/.apareto/stores)
```

1. `scripts/build_expert_store.py` packs the bytes of all 10,240 experts into one file.
2. `scripts/build_layer_store.py` repacks them layer-contiguous (`stores/layer_store/experts.apkl`, ~18 GB),
   verifying every record's payload against the packed copy.
3. `scripts/write_store_header.py` keeps the packed store's header (`stores/expert_store_header.json`). At every
   boot the server checks the served checkpoint against it.
4. The intermediate packed store is deleted. Pass `--keep-packed` to keep it.

The stores hold the checkpoint's own bytes, unchanged; they are a different layout, not a requantization.

### R\* bundles

The server reads the R\* data from `artifacts/`, or from `$APARETO_ARTIFACTS` (`scripts/serve.py --artifacts`):
one bundle per read point, as named in `config/profiles.json`:

```
artifacts/
  code/think/              static think ranking (and the early prediction), read after prefill
  code/midrun/             midrun ranking, read after prefill and at every refresh
  code/answer/             answer ranking, read after </think>
```

Each bundle carries its own `MANIFEST.sha256` and its read contract (`read_contract.json`: tap sites, rows, the
spacer). `tests/check_bundles.py` checks every file against the manifest, and the server does at every boot;
`setup/install.sh` runs it too. The bundles are part of this repository (410 MB). Files larger than 48 MiB are
stored as numbered parts (`query_tower.npz.part000`, …) and read as one file, so no single file exceeds
GitHub's 100 MB limit. [SELECTION.md](SELECTION.md#5-the-bundles) lists what each file holds.

## Paths

Each path has a default, and an environment variable and a `scripts/serve.py` flag that override it:

| what | default | env | flag |
|---|---|---|---|
| environments, caches, run outputs | `~/.apareto` | `APARETO_HOME` | — |
| checkpoint | `$APARETO_HOME/models/Qwen3.6-35B-A3B-NVFP4` | `APARETO_MODEL` | `--model` |
| expert stores | `$APARETO_HOME/stores` | `APARETO_STORES` | `--stores` |
| R\* bundles | `artifacts/` | `APARETO_ARTIFACTS` | `--artifacts` |
| GPU | 0 | `APARETO_DEVICE` | `--device` |
| CPU cores for the rescue's CPU experts | the last 2 of the server's cores | `APARETO_CPU_EXPERT_CORES` (e.g. `6;7`) | — |

`python3 scripts/serve.py --profile code --dry-run` prints the resolved server command and lists any input
that is missing.

## Troubleshooting

- **`Stock vLLM changed: <file>`** at boot: something modified the installed vLLM. Rebuild the environment
  (`rm -rf ~/.apareto/venv ~/.apareto/vllm_stock_manifest.sha256 ~/.apareto/vllm_path && bash setup/install.sh`).
- **The first boot sits at the kernel build for minutes:** that is the JIT build from step 3. It runs once per
  machine; the result is cached in `$APARETO_HOME/cache`.
- **The server refuses to start with a fit-check error:** the bank, KV cache and activation reserve do not fit
  the free GPU memory. Close other GPU processes (desktop compositors count), or lower `--max-model-len` (pass it
  after `--`; see [RUNNING.md](RUNNING.md#the-server)).
- **The first boot of a new configuration reports less KV room than later boots:** a cold `torch.compile`
  cache reports about 0.9 GB less at startup. Boot once to warm the cache before a measured run.
