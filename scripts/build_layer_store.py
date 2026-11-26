#!/usr/bin/env python3
"""Repack a verified PackedStore into the kernel-ready, layer-contiguous store (one read per layer)."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apareto_loader.layer_store import build_layer_store
from apareto_loader.store import PackedStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', required=True, type=Path, help='verified checkpoint-layout .apex store')
    parser.add_argument('--out', required=True, type=Path, help='new .apkl file; its parent must exist')
    parser.add_argument('--report', required=True, type=Path, help='new build result JSON')
    parser.add_argument('--order', choices=('expert_id', 'global_rank'), default='expert_id')
    args = parser.parse_args()
    started = time.monotonic()
    result = {'status': 'failed', 'claim_type': 'measured', 'server_boots': 0,
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'source_store': str(args.store.resolve()),
              'versions': {name: version(name) for name in ('numpy', 'torch', 'vllm')}}
    report = args.report.open('x')
    try:
        with PackedStore(args.store, verify_payload=True) as packed:
            header = build_layer_store(packed, args.out,
                                       progress=lambda layer: print(f'layer {layer} written', flush=True),
                                       order=args.order)
        result.update(status='pass', store=str(args.out.resolve()), store_bytes=args.out.stat().st_size,
                      record_stride=header['record_stride'], layer_bytes=header['layer_bytes'], record_order=args.order,
                      header_sha256=header['header_sha256'], transform=header['transform'],
                      source_store_header_sha256=header['source_store_header_sha256'])
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'; print('[FAIL] ' + result['error'], flush=True)
    finally:
        result['seconds'] = time.monotonic() - started
        json.dump(result, report, indent=2); report.write('\n'); report.close()
    print(json.dumps({k: result.get(k) for k in ('status', 'store_bytes', 'seconds', 'error')}))
    return result['status'] != 'pass'


if __name__ == '__main__': sys.exit(main())
