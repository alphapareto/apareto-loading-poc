#!/usr/bin/env python3
"""Pack the checkpoint's expert bytes in expert-id order (layer 0 expert 0, layer 0 expert 1, ...)."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apareto_loader.store import Checkpoint, Ranking, align_up, build_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, help="new .apex file; its parent must exist")
    parser.add_argument("--align", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true", help="read headers only, write nothing")
    parser.add_argument("--report", type=Path, help="new build result JSON (a build record, not a verification result)")
    args = parser.parse_args()
    if not args.dry_run and args.out is None:
        parser.error("--out is required unless --dry-run")
    if args.dry_run and args.report:
        parser.error("--dry-run writes nothing; omit --report")
    started = time.monotonic()
    report_file = args.report.open("x") if args.report else None
    result = {"status": "failed", "claim_type": "measured", "server_boots": 0,
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "versions": {"python": platform.python_version(), "numpy": version("numpy"),
                           "vllm": version("vllm")}}
    try:
        ranking = Ranking.expert_order()
        with Checkpoint(args.model, ranking) as checkpoint:
            stride = align_up(checkpoint.seat_bytes, args.align)
            result.update({"geometry": [ranking.layers, ranking.experts],
                           "seat_payload_bytes": checkpoint.seat_bytes, "record_stride": stride,
                           "store_record_bytes": len(ranking.order) * stride,
                           "tensor_schema": checkpoint.schema,
                           "provenance": checkpoint.provenance,
                           "gpu_resident_bytes_per_seat": None})
            print(f"Geometry: {ranking.layers} layers x {ranking.experts} experts; {len(ranking.order)} seats")
            for suffix, field in checkpoint.schema.items():
                print(f"  {suffix}: {field['dtype']} {field['shape']} ({field['nbytes']} bytes)")
            print(f"Seat payload: {checkpoint.seat_bytes} bytes; aligned file stride: {stride} bytes")
            print(f"Store records: {len(ranking.order) * stride} bytes "
                  f"({len(ranking.order) * stride / 2**30:.6f} GiB), plus aligned header")
            print("GPU representation: measured separately; file padding is not GPU weight storage", flush=True)
            if args.dry_run:
                return 0
            header = build_store(checkpoint, args.out, args.align, progress=lambda message: print(message, flush=True))
            result.update({"status": "pass", "store": str(args.out.resolve()),
                           "store_bytes": args.out.stat().st_size, "data_offset": header["data_offset"],
                           "source_fingerprint": header["source_fingerprint"],
                           "payload_sha256": header["payload_sha256"],
                           "header_sha256": header["header_sha256"]})
            print(f"[PASS] Built {args.out}: {result['store_bytes']} bytes", flush=True)
            print("The store still needs a separate verification before serving", flush=True)
        return 0
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        print(f"[FAIL] {result['error']}", file=sys.stderr, flush=True)
        return 1
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        if report_file:
            with report_file:
                json.dump(result, report_file, indent=2, allow_nan=False)
                report_file.write("\n")


if __name__ == "__main__":
    sys.exit(main())
