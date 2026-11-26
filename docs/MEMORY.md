# Memory

What one server uses with the `code` profile: GPU memory, host RAM, and disk. Every figure is labeled
**measured** (with its number of boots), **computed** (from sizes in the code or the files) or **by subtraction**.

| | in use | you need |
|---|---|---|
| GPU memory | 15.2 GiB peak | a 16 GB card with nothing else on it |
| host RAM | 25.9 GiB peak | 25 GiB free at boot; a 32 GB machine for one server |
| disk | 43 GB (checkpoint, expert store, bundles) | about 85 GB including the build and the environments |

## GPU memory

A 16 GB card reports 15.92 GiB in total; the driver keeps 0.45 GiB, which leaves 15.47 GiB addressable.

| item | size | label |
|---|---|---|
| expert bank: 5,256 expert slots of 1.77 MB | 8.66 GiB | measured (startup record) |
| the rest of the model: attention, gated-DeltaNet, shared experts, routers, embeddings | 4.42 GiB | measured (vLLM's load report minus the bank) |
| KV cache for all live requests, 66,267 tokens | 1.30 GiB | set by the profile (1,400 MB) |
| CUDA graphs | 0.05 GiB | measured (vLLM's report) |
| CUDA context, compiled kernels, activations | 0.78 GiB | by subtraction |
| **peak over a whole run** | **15.17 to 15.21 GiB** | measured, 3 full LiveCodeBench runs, sampled every 0.1 s |
| left free | 0.26 GiB | |

The routed experts alone are 16.9 GiB, more than the card. The bank holds 5,256 of the 10,240; the rest are
read from host RAM when a keep-set needs them ([HOW-IT-WORKS.md §6](HOW-IT-WORKS.md#6-memory-tiers-and-concurrency)).
R\* itself uses no GPU memory beyond a 360 KB buffer for the captured features.

At boot the server checks, to the byte, that the bank, the KV cache, the activation reserve (250 MB) and the
margin (200 MiB) fit the free memory, and refuses to start if they do not.

## Host RAM

Peak of the server's whole process group, from boot to shutdown, read from the server's memory cgroup:
**25.9 GiB** with three requests decoding at once and 80 pauses and resumes (measured, 1 boot, 6 problems), and
25.7 GiB with one short request (measured, 1 boot). The large items are allocated at boot, so load adds little.

| item | size | label |
|---|---|---|
| pinned copy of every expert (prefill streams from it, decode reads experts outside the bank from it, the rescue computes from it) | 16.91 GiB | measured (startup record) |
| pinned arena for paused requests' KV: room for two requests | 2.74 GiB | computed: (3 − 1) × 1.05 × the KV cache |
| the three R\* networks | 0.83 GiB | measured in a separate process, see below |
| Python, vLLM's two processes, PyTorch and CUDA on the host side, the tokenizer | 5.2 GiB | by subtraction |
| **peak** | **25.7 to 25.9 GiB** | measured, 2 boots |

The launcher refuses to start unless 25 GiB are available (the pinned expert copy plus an 8 GiB reserve). It runs
the server in its own systemd scope with a memory limit of what is available minus 3 GiB and no swap, so a server
that runs out of memory is stopped alone and does not take the desktop session with it.

The published runs used two servers at once on a machine with 91 GiB.

### R\* in memory

| network | in RAM | on disk |
|---|---|---|
| think (also used for the early prediction during prefill) | 295 MB | 82 MB |
| midrun (shares the think network's query tower in RAM) | 263 MB | 247 MB |
| answer | 291 MB | 82 MB |
| **total** | **849 MB** (measured, loading them as the server does) | 411 MB |

On disk the query-tower weights are 16-bit floats and the rank tables 8-bit integers. In RAM the weights are
64-bit floats, which is why a 73 MB tower takes 288 MB; the rank tables stay 8-bit and only the selected rows
are converted for each prediction. One prediction needs a few megabytes of working memory.

## Disk

| item | size |
|---|---|
| checkpoint (`RedHatAI/Qwen3.6-35B-A3B-NVFP4`) | 24 GB |
| expert store, built from the checkpoint | 18 GB (about 40 GB free while it is built) |
| this repository, including the R\* bundles | 0.4 GB |
| Python environments and caches | about 20 GB |

## What changes these numbers

| option | effect |
|---|---|
| `--gpu-experts` | GPU: 1.77 MB per slot. Fewer slots leave more room for KV and make decode read more experts from host RAM, which is slower. |
| `--kv-cache-memory-bytes`, `--max-model-len` | GPU: the KV cache. Host: the paused-request arena scales with it. |
| `--max-num-seqs` | Host: the paused-request arena is (n − 1) × 1.05 × the KV cache. With 1 there is none. |
| `--budget` | Neither. The keep-set size changes which experts are used, not how many slots or bytes are allocated. |
| `--host-tier-gib` | Host: the pinned expert copy. The rescue needs the whole store (17), and the server refuses less while the rescue is on. |

## Measuring your own run

Every run records both peaks. `python3 scripts/server_stats.py ~/.apareto/results/<tag>` prints them under
`memory_gib`: `gpu_peak_used`, `expert_bank`, `kv_cache`, `host_peak` and `host_pinned_expert_tier`. The raw GPU
samples are in `<tag>_memory.jsonl`, and the host figures in `<tag>_results.json` under `host_memory`.
