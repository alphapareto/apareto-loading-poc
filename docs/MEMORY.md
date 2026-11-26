# Memory

What one server uses with the `code` profile: GPU memory, host RAM, and disk. Every figure is labeled
**measured** (with its number of boots), **computed** (from sizes in the code or the files) or **by subtraction**.

| | in use | you need |
|---|---|---|
| GPU memory | 15.2 GiB peak | at the default profile, a 16 GB card with nothing else on it; less with a smaller expert bank |
| host RAM | 25.9 GiB peak with three requests, 21.6 GiB with one | 26 GiB free for three requests (the launcher checks for 25); a 32 GB machine for one server |
| disk | 43 GB (checkpoint, expert store, bundles) | about 90 GB while installing, 67 GB afterwards, including the environments |

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

This is the `code` profile, sized for a 16 GB card. The bank is the adjustable part: `--gpu-experts` sets its
size, and everything else in the table (about 6.6 GiB) is needed whatever the bank holds. A card with less
memory takes a smaller bank, and the experts that do not fit are computed on the CPU or read from host RAM
([a 12 GB budget, measured](#a-12-gb-card-simulated)).

The routed experts alone are 16.9 GiB, more than the card. The bank holds 5,256 of the 10,240; the rest are
read from host RAM when a keep-set needs them ([HOW-IT-WORKS.md §6](HOW-IT-WORKS.md#6-memory-tiers-and-concurrency)).
R\* itself uses no GPU memory beyond a 360 KB buffer for the captured features.

At boot the server checks, to the byte, that the bank, the KV cache, the activation reserve (250 MB) and the
margin (200 MiB) fit the free memory, and refuses to start if they do not.

## A 12 GB card (simulated)

The same server with a smaller expert bank, measured on a 16 GB card on which another process held 3.98 GiB for
the whole run (1 boot, 5 problems, one request at a time). A 12 GB card then has 11.49 GiB addressable.

| item | 16 GB | 12 GB (simulated) |
|---|---|---|
| expert bank | 5,256 slots, 8.66 GiB | 2,880 slots, 4.75 GiB |
| the rest of the model | 4.42 GiB | 4.42 GiB |
| KV cache (64k context) | 1.30 GiB | 1.30 GiB |
| CUDA graphs, context, kernels, activations | 0.83 GiB | 0.75 GiB (by subtraction) |
| **peak** | **15.17 to 15.21 GiB** | **11.22 GiB** |
| left free | 0.26 GiB | 0.27 GiB |
| experts of a 5,018 keep-set in GPU memory | all | 2,880; the rest are computed on the CPU |
| speed | up to 215 tok/s over three requests, up to 104 tok/s with one | up to 75 tok/s with one request |
| host RAM peak, one request | 21.6 GiB | 21.7 GiB |

How to run it, and what an 8 GB card would take: [RUNNING.md](RUNNING.md#cards-with-16-12-and-8-gb). Options
used: `--max-num-seqs 1 --gpu-experts 2880 --gpu-margin-bytes 150000000`, and for the speed
`--cpu-experts cores=auto:6,absent=cpu,admit=off`. With the default margin
(200 MiB) the server's fit check allows 2,850 slots. The right number for a card has to be found by booting: the
fit check refuses a bank that is too large, and the recorded peak shows how much is left.

An 8 GB budget with a 32k context, measured the same way (7.96 GiB held by another process, 7.51 GiB
addressable): bank 830 slots, 1.37 GiB; KV cache 0.69 GiB; the rest as above; peak 7.24 GiB, 0.27 GiB left free;
up to 49 tok/s.

The speeds are the highest measured, without MTP; the 12 GB, 8 GB and three-request figures use the CPU for the
experts that do not fit ([RESULTS.md](RESULTS.md#throughput)). The 12 GB figures are with a 64k context and one request. A shorter context frees KV memory for more expert
slots (1.77 MB each). Three requests at once on 12 GB have not been measured.

## Host RAM

Peak of the server's whole process group, from boot to shutdown, read from the server's memory cgroup:
**25.6 to 25.9 GiB** at the profile's three requests (measured, 3 boots: 25.9 with three requests decoding at
once and 80 pauses and resumes, 25.7 and 25.6 on short requests), and **21.6 GiB** with `--max-num-seqs 1`
(measured, 1 boot). The large items are allocated at boot, so load adds little.

| item | three requests (profile) | one request (`--max-num-seqs 1`) | label |
|---|---|---|---|
| pinned copy of every expert (prefill streams from it, decode reads experts outside the bank from it, the rescue computes from it) | 16.91 GiB | 16.91 GiB | measured (size of the expert store) |
| pinned arena for paused requests' KV | 2.74 GiB, room for two requests | none | computed: (n − 1) × 1.05 × the KV cache |
| the three R\* networks | 0.83 GiB | 0.83 GiB | measured in a separate process, see below |
| Python, vLLM's two processes, PyTorch and CUDA on the host side, the tokenizer | 5.1 to 5.4 GiB | 3.9 GiB | by subtraction |
| **peak** | **25.6 to 25.9 GiB** | **21.6 GiB** | measured, 3 boots and 1 boot |

The last item is what is left after the others, and it is 1.2 to 1.5 GiB larger with three requests than with
one. What that difference consists of has not been measured.

The launcher refuses to start unless 25 GiB are available (a 17 GiB allowance for the pinned expert copy plus an
8 GiB reserve). That check is 0.9 GiB below the measured peak with three requests, so have 26 GiB free. It runs
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
| Python environments (15 GB) and caches (10 GB) | 25 GB (measured on the test machine) |

## What changes these numbers

| option | effect |
|---|---|
| `--gpu-experts` | GPU: 1.77 MB per slot. Fewer slots leave more room for KV; the experts that do not fit are computed on the CPU or read from host RAM. |
| `--kv-cache-memory-bytes`, `--max-model-len` | GPU: the KV cache. Host: the paused-request arena scales with it. |
| `--max-num-seqs` | Host: the paused-request arena is (n − 1) × 1.05 × the KV cache. With 1 there is none. |
| `--budget` | Neither. The keep-set size changes which experts are used, not how many slots or bytes are allocated. |
| `--host-tier-gib` | Host: the pinned expert copy. The rescue needs the whole store (17), and the server refuses less while the rescue is on. |

## Measuring your own run

Every run records both peaks. `python3 scripts/server_stats.py ~/.apareto/results/<tag>` prints them under
`memory_gib`: `gpu_peak_used`, `expert_bank`, `kv_cache`, `host_peak` and `host_pinned_expert_tier`. The raw GPU
samples are in `<tag>_memory.jsonl`, and the host figures in `<tag>_results.json` under `host_memory`.
