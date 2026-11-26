"""CPU-served decode misses: the CPU library and the in-graph protocol (P3), on GPU1-class
hardware with the real layer store. Each check prints PASS/FAIL; nonzero exit on any failure.

P1 parity      the server's expert kernel against an fp64 reference built from the checkpoint-layout fields
               (layer_store.inverse_record: an independent unswizzle), 2 experts in every one of the 40 layers:
               cosine >= 0.9999 and relative L2 error <= 2%.
P1 multi       a 3-token request in which two tokens share an expert, through ring -> server -> finish: each row
               equals the sum of routing weight x single-token expert over its CPU pairs (bf16 add tolerance).
P1 timing      single-token single-expert requests on cold records (256 experts of a layer, random): the server's
               compute time per request p50 <= 90 us at 4 workers.
P3 split       ring: resident allowed ids go to the GPU, absent allowed ids with a host record to the CPU (GPU id
               -1, scale 0), disallowed ids to neither; routed[] keeps every allowed id of token 0.
P3 partition   the audit's contract over 300 random routings, keeps and tables (1-8 tokens): every allowed
               (token, expert) is served exactly once with its routing weight, by the GPU or by the CPU; a
               disallowed one by neither.
P3 disabled    with the CPU path disabled the GPU ids equal the paged branch's (absent experts zero-copy) and
               nothing is requested.
P3 fallback    with a timeout of 0 the finish kernel computes the CPU pairs on the GPU zero-copy: rows agree with
               the CPU's within int8-activation error (cosine >= 0.999), every request counted as a timeout.
P3 stale       a late server answer for an older sequence is never added: after forced timeouts, a later request
               with the server running returns the CPU's result for that request (not the earlier one's).
R1 rescue off  absent=gpu with the rescue rule configured but switched off: over 200 random routings the GPU ids and
               scales equal the paged branch's torch ops (allowed ? id : -1) and nothing is requested; switched on,
               the GPU's share is that plus the rescues the bank holds (resident), each at its routing weight.
R2 rescue rule over 400 random routings (tokens 1-8, x in {0.13, 0.05}, cap in {1, 2}): the oracle's rescues -- per
               token row the up to `cap` heaviest picks outside the keep-set with weight > x, first k on a tie --
               are served each with its routing weight: the absent ones as CPU pairs (the audit view, served, names
               them), the resident ones in the GPU's ids (rescues the bank holds go to the GPU); a mutated
               oracle (cap + 1, or x / 2) disagrees somewhere (the check can fail).
R2c cap table  a per-layer cap list; in the tested layer caps 0-3 (0 = no rescue at all, the
               GPU share equals the paged branch's) match the oracle at that cap, with different caps in the other
               layers; the oracle at the neighbour layer's cap disagrees somewhere (the table is read per layer).
R2g gain rule  with a per-(layer, expert) gain table the rule ranks dropped picks by the fp32
               score weight x gain[expert] (threshold x on the score) and serves the chosen pick at its routing weight;
               400 random routings, log-normal gains (sd 0.7 in log), x in {0.13, 0.25}, cap in {1, 2}, against
               dispatch_check.rescue_terms with the same gain row; the plain-weight oracle disagrees somewhere (the
               gain is live), and an all-ones table reproduces R2's weight rule exactly.
R3 rescue      a rescued pair's output: the CPU's row equals weight x the single-token CPU expert, and the same
               pair forced through the GPU fallback (timeout 0) agrees with it within int8-activation error.
R3c multi      rows with 2-3 rescues (all CPU; cap 2 of 3; one resident -> GPU id at its
               weight; two rows sharing an expert; the same pairs via the GPU fallback): each row equals the sum of
               weight x CPU expert over its CPU rescues (cosine >= 0.9999, fallback >= 0.999).
mutations      deleting the fallback (finish without it) must fail P3 fallback; sending no hidden rows must fail
               P1 multi (both checked by running the check on a broken input).

  systemd-run --user --scope -p MemoryMax=16G env MAX_JOBS=2 CUDA_VISIBLE_DEVICES=1 \
      ~/.apareto/venv/bin/python tests/test_cpu_experts.py --tag <fresh>
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apareto_loader import paths
from apareto_loader.layer_store import LayerStore, inverse_record
from apareto_loader.vllm027 import cpu_experts

import os
STORES = paths.stores()
STORE = STORES / 'layer_store' / 'experts.apkl'
H, F, E = 2048, 512, 256
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def e4m3(raw):
    raw = raw.astype(np.int64)
    e, m = (raw >> 3) & 0xF, raw & 7
    v = np.where(e == 0, m * 2.0 ** -9, (1 + m / 8) * 2.0 ** (e - 7))
    return np.where((raw & 0x7F) == 0x7F, 0, v)


def dequant(packed, scales, rows, cols):
    q = np.frombuffer(packed, dtype=np.uint8).reshape(rows, cols // 2)
    w = np.empty((rows, cols))
    w[:, 0::2], w[:, 1::2] = E2M1[q & 15], E2M1[q >> 4]
    s = e4m3(np.frombuffer(scales, dtype=np.uint8).reshape(rows, cols // 16))
    return w * np.repeat(s, 16, axis=1)


def reference(record, layout, schema, x):
    """fp64 expert output from the checkpoint-layout fields (independent of the kernels' swizzle arithmetic)."""
    f = inverse_record(record, layout, schema)
    g13 = np.frombuffer(f['gate_proj.weight_global_scale'], dtype=np.float32)[0]
    g2 = np.frombuffer(f['down_proj.weight_global_scale'], dtype=np.float32)[0]
    gate = dequant(f['gate_proj.weight_packed'], f['gate_proj.weight_scale'], F, H) / g13
    up = dequant(f['up_proj.weight_packed'], f['up_proj.weight_scale'], F, H) / g13
    down = dequant(f['down_proj.weight_packed'], f['down_proj.weight_scale'], H, F) / g2
    g, u = gate @ x, up @ x
    return down @ (g / (1 + np.exp(-g)) * u)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--cores', default='4-7')
    args = parser.parse_args()
    out = paths.results() / f'{args.tag}_results.json'
    if Path(args.tag).name != args.tag or out.exists():
        parser.error('use a fresh filename-safe tag')
    checks = {}

    def check(name, ok, detail):
        checks[name] = {'pass': bool(ok), 'detail': detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    record = {'status': 'failed', 'claim_type': 'measured (one process, GPU + idle CPU)',
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'cores': args.cores}
    rng = np.random.default_rng(0)
    import faulthandler
    faulthandler.dump_traceback_later(240, repeat=True)   # a hang shows where it is
    try:
        t0 = time.time()
        m = cpu_experts.module()
        record['build_seconds'] = round(time.time() - t0, 1)
        store = LayerStore(STORE)
        layout, stride = store.layout, store.stride
        schema = json.loads((STORES / 'expert_store_header.json').read_text())['tensor_schema']
        offsets = [layout[n]['offset'] for n in ('w13', 's13', 'w2', 's2', 'global13', 'global2')]

        # records of two layers, pinned (cudaHostAlloc: host and device see the same address)
        layers_loaded = (3, 27)
        pinned = {}
        for layer in layers_loaded:
            buf = torch.empty(E * stride, dtype=torch.uint8).pin_memory()
            store.read_into(layer, 0, E, buf.numpy(), verify=False)
            pinned[layer] = buf
        addr = lambda layer, e: pinned[layer].data_ptr() + e * stride

        # P1 parity: 2 experts per layer, every layer
        worst_cos, worst_rel = 1.0, 0.0
        buf = np.empty(stride * E, dtype=np.uint8)
        for layer in range(40):
            store.read_into(layer, 0, E, buf, verify=False)
            for e in rng.choice(E, 2, replace=False):
                rec = buf[e * stride:(e + 1) * stride]
                x = rng.standard_normal(H).astype(np.float32)
                ref = reference(rec.tobytes(), layout, schema, x.astype(np.float64))
                got = m.expert(rec.ctypes.data, torch.from_numpy(x), offsets, 1).numpy().astype(np.float64)
                cos = float(ref @ got / np.linalg.norm(ref) / np.linalg.norm(got))
                rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
                worst_cos, worst_rel = min(worst_cos, cos), max(worst_rel, rel)
        check('P1 parity', worst_cos >= 0.9999 and worst_rel <= 0.02,
              f'80 experts (2 per layer x 40): min cosine {worst_cos:.6f}, max relative L2 error {worst_rel:.4f}')

        # in-graph pieces: one CpuExperts, on the given cores
        dev = torch.device('cuda')
        cx = cpu_experts.CpuExperts(40, offsets, args.cores, timeout_us=1000.0, device=0)
        keep = torch.ones(E, dtype=torch.bool, device=dev)
        table = torch.zeros(E, dtype=torch.int32, device=dev)
        host = torch.zeros(E, dtype=torch.int64, device=dev)
        routed = torch.full((8,), -1, dtype=torch.int32, device=dev)

        def setup(layer, absent):
            table.fill_(0); host.zero_()
            for e in absent:
                table[e] = -1
                host[e] = addr(layer, e) + offsets[0]

        def run(layer, logical, weights, hidden):
            ids, scales = cx.ring(layer, logical, weights, keep, table, host, hidden, routed)
            ids, scales = ids.clone(), scales.clone()
            fused = torch.zeros_like(hidden)
            cx.finish(layer, fused)
            torch.cuda.synchronize()
            return ids, scales, fused

        def expected(layer, logical, weights, hidden, absent):
            outs = torch.zeros(hidden.shape, dtype=torch.float64)
            for t in range(logical.shape[0]):
                for k in range(8):
                    e = int(logical[t, k])
                    if e in absent:
                        x = hidden[t].float().cpu().contiguous()
                        outs[t] += float(weights[t, k]) * m.expert(addr(layer, e), x, offsets, 1).double()
            return outs

        def close(a, b):
            a, b = a.double().cpu().flatten(), b.double().flatten()
            return float(a @ b / (a.norm() * b.norm() + 1e-30)), float((a - b).abs().max() / (b.abs().max() + 1e-30))

        # P3 split
        layer = 3
        logical = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32, device=dev)
        weights = torch.full((1, 8), 0.125, dtype=torch.float32, device=dev)
        hidden = (torch.randn(1, H, device=dev) * 0.5).to(torch.bfloat16)
        keep.fill_(True); keep[8] = False
        setup(layer, absent=[2, 5])
        ids, scales, fused = run(layer, logical, weights, hidden)
        want_ids = [1, -1, 3, 4, -1, 6, 7, -1]
        check('P3 split', ids[0].tolist() == want_ids and scales[0].tolist() == [0.125 if i >= 0 else 0 for i in want_ids]
              and routed.tolist() == [1, 2, 3, 4, 5, 6, 7, -1],
              f'gpu ids {ids[0].tolist()}, routed {routed.tolist()}')
        keep.fill_(True)

        # P1 multi: three tokens; tokens 0 and 2 share expert 11; token 1 has two absent experts
        logical = torch.tensor([[11, 20, 21, 22, 23, 24, 25, 26],
                                [30, 31, 12, 13, 32, 33, 34, 35],
                                [40, 41, 42, 11, 43, 44, 45, 46]], dtype=torch.int32, device=dev)
        weights = torch.rand(3, 8, device=dev)
        weights = weights / weights.sum(1, keepdim=True)
        hidden = (torch.randn(3, H, device=dev) * 0.5).to(torch.bfloat16)
        absent = [11, 12, 13]
        setup(layer, absent)
        before = cx.stats()
        ids, scales, fused = run(layer, logical, weights, hidden)
        want = expected(layer, logical.cpu(), weights.cpu(), hidden, set(absent))
        cos, err = close(fused.float(), want)
        after = cx.stats()
        served = after['served_experts'] - before['served_experts']
        multi_ok = cos >= 0.9999 and err <= 0.02 and served == 3
        check('P1 multi', multi_ok, f'3 tokens, 4 CPU pairs over 3 distinct experts (served {served}): cosine {cos:.6f}, '
              f'max abs error / max {err:.4f}')
        # P6 norms: each CPU pair's own output norm, filed at its (token, k); every other slot -1
        got_n = cx.norms()[layer].view(8, 8)[:3].cpu()
        worst_n, wrong_slots = 0.0, 0
        for t in range(3):
            for k in range(8):
                e = int(logical[t, k])
                if e in absent:
                    ref_n = float(m.expert(addr(layer, e), hidden[t].float().cpu().contiguous(), offsets, 1).double().norm())
                    worst_n = max(worst_n, abs(float(got_n[t, k]) - ref_n) / ref_n)
                elif float(got_n[t, k]) != -1.0:
                    wrong_slots += 1
        check('P6 norms', worst_n <= 2e-3 and wrong_slots == 0,
              f'4 CPU pairs: max relative error vs the single-expert kernel {worst_n:.2e}; non-CPU slots not -1: {wrong_slots}')
        # mutation: the same check on a broken request (hidden rows zeroed) must fail
        ids, scales, broken = run(layer, logical, weights, torch.zeros_like(hidden))
        cos_b, _ = close(broken.float() + 1e-12, want)
        check('P1 multi mutation caught', not (cos_b >= 0.9999), f'zeroed hidden rows give cosine {cos_b:.4f} (must fail)')

        # P1 timing: cold single-expert requests
        times = []
        for i in range(600):
            layer = layers_loaded[i % 2]
            e = int(rng.integers(E))
            setup(layer, [e])
            lg = torch.tensor([[e] + [int(x) for x in rng.choice([y for y in range(E) if y != e], 7, replace=False)]],
                              dtype=torch.int32, device=dev)
            s0 = cx.stats()
            run(layer, lg, torch.full((1, 8), 0.125, device=dev), (torch.randn(1, H, device=dev) * 0.5).to(torch.bfloat16))
            s1 = cx.stats()
            if i >= 50 and s1['served_requests'] - s0['served_requests'] == 1:
                times.append((s1['compute_ns'] - s0['compute_ns']) / 1000)
        p50, p90 = float(np.percentile(times, 50)), float(np.percentile(times, 90))
        check('P1 timing', p50 <= 90 and len(times) >= 500,
              f'{len(times)} cold single-expert requests: server compute p50 {p50:.1f} us, p90 {p90:.1f} us (cores {args.cores})')
        record['timing_us'] = {'p50': p50, 'p90': p90, 'n': len(times)}

        # P3 partition (the audit's contract): over random routings, keeps and bank tables, every allowed
        # (token, expert) is served exactly once, by the GPU (its id and weight in the GPU's ids/scales) or by the
        # CPU (a request pair with that weight); a disallowed one by neither; the routed row is token 0's allowed ids.
        bad = []
        layer = 27
        for case in range(300):
            tokens = int(rng.choice([1, 2, 3, 8]))
            lg = np.stack([rng.choice(E, 8, replace=False) for _ in range(tokens)]).astype(np.int32)
            lg[rng.random(lg.shape) < 0.05] = -1
            kp = rng.random(E) < 0.7
            ab = rng.random(E) < 0.4
            with_host = rng.random(E) < 0.9
            keep.copy_(torch.from_numpy(kp))
            table.copy_(torch.from_numpy(np.where(ab, -1, 1).astype(np.int32)))
            host.copy_(torch.from_numpy(np.where(ab & with_host, [addr(layer, e) + offsets[0] for e in range(E)], 0)))
            wts = rng.random(lg.shape).astype(np.float32)
            seq_before = m.request(layer)[0]
            ids, scales, _ = run(layer, torch.from_numpy(lg).to(dev), torch.from_numpy(wts).to(dev),
                                 (torch.randn(tokens, H, device=dev) * 0.5).to(torch.bfloat16))
            got_gpu = {(t, int(i)): float(s) for t in range(tokens) for i, s in zip(ids[t].tolist(), scales[t].tolist()) if i >= 0}
            req = m.request(layer)
            got_cpu = {}
            if req[0] != seq_before:   # a request was made this case
                got_cpu = {(int(req[3 + 3 * p]), int(req[4 + 3 * p])): req[5 + 3 * p] for p in range(int(req[2]))}
            want_gpu, want_cpu = {}, {}
            for t in range(tokens):
                for k in range(8):
                    e = int(lg[t, k])
                    if e < 0 or not kp[e]:
                        continue
                    (want_cpu if (ab[e] and with_host[e] and tokens <= m.TMAX) else want_gpu)[(t, e)] = float(wts[t, k])
            routed_ok = routed.tolist() == [int(e) if e >= 0 and kp[e] else -1 for e in lg[0]]
            close_map = lambda a, b: a.keys() == b.keys() and all(abs(a[x] - b[x]) < 1e-6 for x in a)
            if not (close_map(got_gpu, want_gpu) and close_map(got_cpu, want_cpu) and routed_ok):
                bad.append(case)
        check('P3 partition', not bad, f'300 random cases (tokens 1-8, 30% disallowed, 40% absent, 10% of those without '
              f'a host record, 5% dropped ids): {len(bad)} wrong (first {bad[:3]})')
        keep.fill_(True)

        # P3 disabled: everything allowed goes to the GPU (absent zero-copy), nothing requested
        layer = 3
        setup(layer, [2, 5])
        cx.set_enabled(False)
        s0 = cx.stats()
        logical = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32, device=dev)
        ids, scales, fused = run(layer, logical, torch.full((1, 8), 0.125, device=dev),
                                 (torch.randn(1, H, device=dev) * 0.5).to(torch.bfloat16))
        s1 = cx.stats()
        check('P3 disabled', ids[0].tolist() == list(range(1, 9)) and s1['layers'] == s0['layers'] and not fused.any(),
              f'gpu ids {ids[0].tolist()}, requests {s1["layers"] - s0["layers"]}')
        cx.set_enabled(True)
        cx.stop()

        # Rescue: absent experts stay on the GPU; dropped picks above x go to the CPU.
        rlayer = 27
        rescue_tab = torch.tensor([addr(rlayer, e) + offsets[0] for e in range(E)], dtype=torch.int64, device=dev)

        def rescue_case(cx, lg, kp, ab, wts, live):
            keep.copy_(torch.from_numpy(kp))
            table.copy_(torch.from_numpy(np.where(ab, -1, 1).astype(np.int32)))
            host.copy_(torch.from_numpy(np.where(ab, [addr(rlayer, e) + offsets[0] for e in range(E)], 0)))
            cx.set_rescue(live)
            seq_before = m.request(rlayer)[0]
            ids, scales = cx.ring(rlayer, torch.from_numpy(lg).to(dev), torch.from_numpy(wts).to(dev), keep, table, host,
                                  (torch.randn(lg.shape[0], H, device=dev) * 0.5).to(torch.bfloat16), routed, rescue_tab)
            cpu_ids, cpu_scales = (t.clone() for t in cx.served(lg.shape[0]))
            ids, scales = ids.clone(), scales.clone()
            cx.finish(rlayer, torch.zeros(lg.shape[0], H, dtype=torch.bfloat16, device=dev))
            torch.cuda.synchronize()
            req = m.request(rlayer)
            pairs = {}
            if req[0] != seq_before:
                pairs = {(int(req[3 + 3 * p]), int(req[4 + 3 * p])): req[5 + 3 * p] for p in range(int(req[2]))}
            return ids.cpu().numpy(), scales.cpu().numpy(), cpu_ids.cpu().numpy(), pairs

        def oracle(lg, kp, wts, x, cap, ab=None):
            """The rule's rescues {(t, expert): weight}; with ab, only the absent ones (the CPU's share)."""
            want = {}
            for t in range(lg.shape[0]):
                dropped = [k for k in range(8) if lg[t, k] >= 0 and not kp[lg[t, k]] and np.float32(wts[t, k]) > np.float32(x)]
                for k in sorted(dropped, key=lambda k: (-wts[t, k], k))[:cap]:
                    if ab is None or ab[lg[t, k]]:
                        want[(t, int(lg[t, k]))] = float(wts[t, k])
            return want

        def gpu_rescues(lg, ab, wts, x, cap, kp):
            """The GPU's share with the rule live: the allowed picks plus the rescues the bank holds."""
            ids, scales = gpu_want(lg, kp, wts)
            for (t, e), wt in oracle(lg, kp, wts, x, cap).items():
                if not ab[e]:
                    k = int(np.where(lg[t] == e)[0][0])
                    ids[t, k], scales[t, k] = e, wt
            return ids, scales

        def random_case():
            tokens = int(rng.choice([1, 1, 2, 3, 8]))
            lg = np.stack([rng.choice(E, 8, replace=False) for _ in range(tokens)]).astype(np.int32)
            w = rng.dirichlet(np.full(8, 0.6), size=tokens).astype(np.float32)
            w = -np.sort(-w, axis=1)                    # router order: heaviest first, as top-k returns them
            if rng.random() < 0.2:
                w[:, 1] = w[:, 2]                       # an exact tie between two picks
            return lg, rng.random(E) < 0.5, rng.random(E) < 0.4, w

        def gpu_want(lg, kp, wts):
            allowed = (lg >= 0) & kp[np.clip(lg, 0, None)]
            return np.where(allowed, lg, -1), np.where(allowed, wts, 0)

        cx = cpu_experts.CpuExperts(40, offsets, args.cores, timeout_us=1000.0, device=0, absent_cpu=False,
                                    rescue_x=0.13, rescue_cap=1)
        bad_off, bad_on = [], []
        for case in range(200):
            lg, kp, ab, w = random_case()
            gi, gs = gpu_want(lg, kp, w)
            ids, scales, _, pairs = rescue_case(cx, lg, kp, ab, w, live=False)
            if not (np.array_equal(ids, gi) and np.array_equal(scales, gs) and not pairs):
                bad_off.append(case)
            ids, scales, _, _ = rescue_case(cx, lg, kp, ab, w, live=True)
            gi, gs = gpu_rescues(lg, ab, w, 0.13, 1, kp)
            if not (np.array_equal(ids, gi) and np.array_equal(scales, gs)):
                bad_on.append(case)
        check('R1 rescue off', not bad_off and not bad_on,
              f'200 random routings: GPU share differs from the paged branch in {len(bad_off)} cases with rescue off '
              f'(or requested a pair), {len(bad_on)} with it on')
        cx.stop()

        bad, mutated_seen, total, gpu_total = [], 0, 0, 0
        for x, cap in ((0.13, 1), (0.13, 2), (0.05, 1), (0.05, 2)):
            cx = cpu_experts.CpuExperts(40, offsets, args.cores, timeout_us=1000.0, device=0, absent_cpu=False,
                                        rescue_x=x, rescue_cap=cap)
            for case in range(100):
                lg, kp, ab, w = random_case()
                ids, scales, cpu_ids, pairs = rescue_case(cx, lg, kp, ab, w, live=True)
                expect = oracle(lg, kp, w, x, cap, ab)
                gi, gs = gpu_rescues(lg, ab, w, x, cap, kp)
                view = {(t, int(cpu_ids[t, k])) for t in range(lg.shape[0]) for k in range(8) if cpu_ids[t, k] >= 0}
                total += len(oracle(lg, kp, w, x, cap))
                gpu_total += len(oracle(lg, kp, w, x, cap)) - len(expect)
                if (pairs.keys() != expect.keys() or any(abs(pairs[k] - expect[k]) > 1e-7 for k in expect) or
                        view != set(expect) or not np.array_equal(ids, gi) or not np.array_equal(scales, gs)):
                    bad.append((x, cap, case))
                if (oracle(lg, kp, w, x, cap + 1, ab).keys() != pairs.keys() or
                        oracle(lg, kp, w, x / 2, cap, ab).keys() != pairs.keys()):
                    mutated_seen += 1
            cx.stop()
        check('R2 rescue rule', not bad and total > 0 and gpu_total > 0 and mutated_seen > 0,
              f'400 random routings, {total} rescues expected ({gpu_total} resident, on the GPU): {len(bad)} cases wrong (first {bad[:3]}); a mutated '
              f'oracle disagreed in {mutated_seen} cases')

        # R2c: per-layer cap table: layer rlayer's own entry rules, other layers' entries do not leak in
        bad_c, leak_seen, total_c, zero_bad = [], 0, 0, 0
        for own_cap in (0, 1, 2, 3):
            caps = [int(c) for c in rng.integers(0, 9, 40)]
            caps[rlayer] = own_cap
            caps[rlayer - 1] = caps[rlayer + 1] = (own_cap + 2) % 9
            cx = cpu_experts.CpuExperts(40, offsets, args.cores, timeout_us=1000.0, device=0, absent_cpu=False,
                                        rescue_x=0.05, rescue_cap=caps)
            for case in range(100):
                lg, kp, ab, w = random_case()
                ids, scales, cpu_ids, pairs = rescue_case(cx, lg, kp, ab, w, live=True)
                expect = oracle(lg, kp, w, 0.05, own_cap, ab)
                gi, gs = gpu_rescues(lg, ab, w, 0.05, own_cap, kp)
                total_c += len(oracle(lg, kp, w, 0.05, own_cap))
                if (pairs.keys() != expect.keys() or any(abs(pairs[k] - expect[k]) > 1e-7 for k in expect) or
                        not np.array_equal(ids, gi) or not np.array_equal(scales, gs)):
                    bad_c.append((own_cap, case))
                if own_cap == 0 and (pairs or not np.array_equal(ids, gpu_want(lg, kp, w)[0])):
                    zero_bad += 1
                if oracle(lg, kp, w, 0.05, caps[rlayer + 1], ab).keys() != pairs.keys():
                    leak_seen += 1
            cx.stop()
        check('R2c cap table', not bad_c and not zero_bad and total_c > 0 and leak_seen > 0,
              f'400 random routings over caps 0-3 in layer {rlayer} (neighbours +2, the rest random): {total_c} rescues expected, '
              f'{len(bad_c)} cases wrong (first {bad_c[:3]}), cap 0 rescued in {zero_bad}; the neighbour layer\'s cap disagreed in {leak_seen}')

        # R2g: the gain-scored rule against the audit's own oracle
        from apareto_loader.dispatch_check import rescue_terms

        def gain_oracle(lg, kp, wts, x, cap, grow, ab=None):
            want = {}
            for t in range(lg.shape[0]):
                for k in rescue_terms(lg[t], wts[t], kp, x, cap, grow):
                    if ab is None or ab[lg[t, k]]:
                        want[(t, int(lg[t, k]))] = float(wts[t, k])
            return want

        bad_g, plain_differs, ones_bad, total_g = [], 0, [], 0
        for x, cap in ((0.13, 1), (0.13, 2), (0.25, 1), (0.25, 2)):
            cx = cpu_experts.CpuExperts(40, offsets, args.cores, timeout_us=1000.0, device=0, absent_cpu=False,
                                        rescue_x=x, rescue_cap=cap)
            gtab = np.exp(rng.normal(0, 0.7, (40, E))).astype(np.float32)
            cx.set_rescue_gain(gtab, dev)
            for case in range(100):
                lg, kp, ab, w = random_case()
                _, _, cpu_ids, pairs = rescue_case(cx, lg, kp, ab, w, live=True)
                expect = gain_oracle(lg, kp, w, x, cap, gtab[rlayer], ab)
                total_g += len(gain_oracle(lg, kp, w, x, cap, gtab[rlayer]))
                view = {(t, int(cpu_ids[t, k])) for t in range(lg.shape[0]) for k in range(8) if cpu_ids[t, k] >= 0}
                if pairs.keys() != expect.keys() or any(abs(pairs[k] - expect[k]) > 1e-7 for k in expect) or view != set(expect):
                    bad_g.append((x, cap, case))
                if oracle(lg, kp, w, x, cap, ab).keys() != pairs.keys():
                    plain_differs += 1
            cx.set_rescue_gain(np.ones((40, E), np.float32), dev)
            for case in range(50):
                lg, kp, ab, w = random_case()
                _, _, _, pairs = rescue_case(cx, lg, kp, ab, w, live=True)
                if pairs.keys() != oracle(lg, kp, w, x, cap, ab).keys():
                    ones_bad.append((x, cap, case))
            cx.stop()
        check('R2g gain rule', not bad_g and not ones_bad and total_g > 0 and plain_differs > 0,
              f'400 random routings with random gains, {total_g} rescues expected: {len(bad_g)} cases wrong (first {bad_g[:3]}); '
              f'the plain-weight oracle disagreed in {plain_differs}; all-ones table vs weight rule: {len(ones_bad)} of 200 wrong')

        # R3: one token, expert 9 dropped with weight 0.3: the CPU row, and the GPU fallback's row, vs the CPU expert
        lg = np.array([[9, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        w = np.array([[0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]], dtype=np.float32)
        kp = np.ones(E, dtype=bool); kp[9] = False
        x3 = (torch.randn(1, H, device=dev) * 0.5).to(torch.bfloat16)
        want3 = 0.3 * m.expert(addr(rlayer, 9), x3[0].float().cpu().contiguous(), offsets, 1).double()
        rows = {}
        for name, timeout in (('cpu', 1000.0), ('gpu_fallback', 0.0)):
            cx = cpu_experts.CpuExperts(40, offsets, args.cores, timeout_us=timeout, device=0, per_pair_us=0.0 if timeout == 0 else 150.0,
                                        absent_cpu=False, rescue_x=0.13, rescue_cap=1)
            keep.copy_(torch.from_numpy(kp)); table.fill_(1); table[9] = -1; host.zero_()   # 9 absent: a CPU rescue
            cx.set_rescue(True)
            cx.ring(rlayer, torch.from_numpy(lg).to(dev), torch.from_numpy(w).to(dev), keep, table, host, x3, routed, rescue_tab)
            row3 = torch.zeros(1, H, dtype=torch.bfloat16, device=dev)
            cx.finish(rlayer, row3)
            torch.cuda.synchronize()
            rows[name] = (close(row3[0].float(), want3), cx.stats()['timeouts'])
            cx.stop()
        check('R3 rescue', rows['cpu'][0][0] >= 0.9999 and rows['gpu_fallback'][0][0] >= 0.999 and rows['gpu_fallback'][1] == 1,
              f"rescued pair (expert 9, weight 0.3): CPU row cosine {rows['cpu'][0][0]:.6f} to weight x CPU expert; "
              f"GPU fallback row cosine {rows['gpu_fallback'][0][0]:.5f} (timeouts {rows['gpu_fallback'][1]})")
        keep.fill_(True); table.fill_(0); host.zero_()

        # R3c: several rescues in one token row (cap 2 / 3), a resident one going to the GPU, two rows sharing an
        # expert, and the GPU fallback of the same pairs: each row = sum of weight x expert over its CPU rescues
        def r3c(lg, w, dropped, resident, cap, timeout):
            cx = cpu_experts.CpuExperts(40, offsets, args.cores, timeout_us=timeout, device=0,
                                        per_pair_us=0.0 if timeout == 0 else 150.0, absent_cpu=False, rescue_x=0.13, rescue_cap=cap)
            kp = np.ones(E, dtype=bool); kp[list(dropped)] = False
            keep.copy_(torch.from_numpy(kp)); table.fill_(1); host.zero_()
            for e in dropped:
                if e not in resident:
                    table[e] = -1
            cx.set_rescue(True)
            xs = (torch.randn(lg.shape[0], H, device=dev) * 0.5).to(torch.bfloat16)
            ids, scales = cx.ring(rlayer, torch.from_numpy(lg).to(dev), torch.from_numpy(w).to(dev), keep, table, host, xs, routed, rescue_tab)
            ids, scales = ids.clone().cpu().numpy(), scales.clone().cpu().numpy()
            rows = torch.zeros(lg.shape[0], H, dtype=torch.bfloat16, device=dev)
            cx.finish(rlayer, rows)
            torch.cuda.synchronize()
            st = cx.stats(); cx.stop()
            cos, gpu_ok, n_cpu = [], True, 0
            for t in range(lg.shape[0]):
                chosen = [k for k in sorted(range(8), key=lambda k: (-w[t, k], k)) if lg[t, k] in dropped and w[t, k] > 0.13][:cap]
                want = torch.zeros(H, dtype=torch.float64)
                for k in chosen:
                    e = int(lg[t, k])
                    if e in resident:
                        gpu_ok &= bool(ids[t, k] == e and abs(scales[t, k] - w[t, k]) < 1e-7)
                    else:
                        n_cpu += 1
                        want += float(w[t, k]) * m.expert(addr(rlayer, e), xs[t].float().cpu().contiguous(), offsets, 1).double()
                for k in range(8):
                    if lg[t, k] in dropped and k not in chosen:
                        gpu_ok &= bool(ids[t, k] == -1)
                cos.append(close(rows[t].float(), want)[0] if want.abs().sum() > 0 else float(rows[t].abs().sum() == 0))
            keep.fill_(True); table.fill_(0); host.zero_()
            return min(cos), gpu_ok, n_cpu, st['timeouts']
        w1 = np.array([[0.3, 0.25, 0.2, 0.1, 0.06, 0.04, 0.03, 0.02]], dtype=np.float32)
        lg1 = np.array([[9, 10, 11, 1, 2, 3, 4, 5]], dtype=np.int32)
        lg2 = np.array([[9, 10, 11, 1, 2, 3, 4, 5], [12, 9, 13, 6, 7, 8, 14, 15]], dtype=np.int32)
        w2 = np.array([[0.3, 0.25, 0.2, 0.1, 0.06, 0.04, 0.03, 0.02], [0.28, 0.22, 0.18, 0.12, 0.1, 0.05, 0.03, 0.02]], dtype=np.float32)
        cases = {'cap3 all CPU': r3c(lg1, w1, {9, 10, 11}, set(), 3, 1000.0),
                 'cap2 of 3 (11 not rescued)': r3c(lg1, w1, {9, 10, 11}, set(), 2, 1000.0),
                 'cap3, 10 resident (GPU)': r3c(lg1, w1, {9, 10, 11}, {10}, 3, 1000.0),
                 '2 rows sharing 9, cap2': r3c(lg2, w2, {9, 10, 11, 12, 13}, set(), 2, 1000.0),
                 '2 rows cap2 GPU fallback': r3c(lg2, w2, {9, 10, 11, 12, 13}, set(), 2, 0.0)}
        ok = all(c[0] >= (0.999 if 'fallback' in n else 0.9999) and c[1] for n, c in cases.items()) and cases['2 rows cap2 GPU fallback'][3] == 1
        check('R3c multi-rescue rows', ok, '; '.join(f'{n}: min row cosine {c[0]:.6f}, GPU share ok {c[1]}, CPU pairs {c[2]}, timeouts {c[3]}' for n, c in cases.items()))
        # mutation: dropping one of a row's two CPU rescues from the reference must fail the cosine bar
        xs_m = (torch.randn(1, H, device=dev) * 0.5).to(torch.bfloat16)
        e9 = m.expert(addr(rlayer, 9), xs_m[0].float().cpu().contiguous(), offsets, 1).double()
        e10 = m.expert(addr(rlayer, 10), xs_m[0].float().cpu().contiguous(), offsets, 1).double()
        check('R3c mutation caught', close((0.3 * e9 + 0.25 * e10).float(), 0.3 * e9)[0] < 0.9999,
              'a row missing one of its two rescues fails the 0.9999 bar')

        # P3 fallback: timeout 0 -> the GPU computes the CPU pairs zero-copy
        cx = cpu_experts.CpuExperts(40, offsets, args.cores, timeout_us=0.0, device=0, per_pair_us=0.0)
        logical = torch.tensor([[11, 20, 21, 22, 23, 24, 25, 26],
                                [30, 31, 12, 13, 32, 33, 34, 35],
                                [40, 41, 42, 11, 43, 44, 45, 46]], dtype=torch.int32, device=dev)
        setup(layer, [11, 12, 13])
        s0 = cx.stats()
        ids, scales, fused = run(layer, logical, weights, hidden)
        s1 = cx.stats()
        cos, err = close(fused.float(), want)
        check('P3 fallback', cos >= 0.999 and s1['timeouts'] - s0['timeouts'] == 1 and s1['fallback_pairs'] - s0['fallback_pairs'] == 4,
              f'GPU zero-copy vs CPU rows: cosine {cos:.5f}, max error / max {err:.4f}; timeouts {s1["timeouts"] - s0["timeouts"]}, '
              f'fallback pairs {s1["fallback_pairs"] - s0["fallback_pairs"]}')
        fb = cx.norms()[layer].view(8, 8)[:3].cpu()
        check('P6 norms fallback', int((fb == -2).sum()) == 4 and int((fb == -1).sum()) == 20,
              f'timed-out pairs filed as -2: {int((fb == -2).sum())} of 4; other slots -1: {int((fb == -1).sum())} of 20')
        # mutation: without the fallback the rows stay zero, which this check rejects
        check('P3 fallback mutation caught', not (close(torch.zeros_like(fused).float() + 1e-12, want)[0] >= 0.999),
              'rows without the fallback (zeros) fail the cosine bar')

        # P3 stale: let the server answer late for one request; the next request must get its own answer
        time.sleep(0.05)
        hidden2 = (torch.randn(3, H, device=dev) * 0.5).to(torch.bfloat16)
        cx.stop()
        cx = cpu_experts.CpuExperts(40, offsets, args.cores, timeout_us=1000.0, device=0)
        ids, scales, fused2 = run(layer, logical, weights, hidden2)
        want2 = expected(layer, logical.cpu(), weights.cpu(), hidden2, {11, 12, 13})
        cos2, _ = close(fused2.float(), want2)
        cos_old, _ = close(fused2.float(), want)
        check('P3 stale', cos2 >= 0.9999 and cos_old < 0.9, f'new request: cosine {cos2:.6f} to its own result, '
              f'{cos_old:.4f} to the previous request\'s')
        record['stats'] = cx.stats()
        cx.stop()
        record['checks'] = checks
        record['status'] = 'pass' if all(c['pass'] for c in checks.values()) else 'failed'
    except Exception as error:
        traceback.print_exc()
        record['error'] = f'{type(error).__name__}: {error}'
        record['checks'] = checks
        print(f'[FAIL] {record["error"]}', flush=True)
    finally:
        record['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        out.open('x').write(json.dumps(record, indent=2) + '\n')
    print(f"{sum(c['pass'] for c in checks.values())}/{len(checks)} passed", flush=True)
    return record['status'] != 'pass'


if __name__ == '__main__':
    sys.exit(main())
