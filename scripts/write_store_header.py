#!/usr/bin/env python3
"""Save a packed expert store's header as JSON.

The server binds the layer store to the checkpoint through the header of the packed store it was built from
(geometry, ranking, tensor schema, weight-byte fingerprint). The header is all it needs, so the packed store
itself can be deleted once the layer store is built.

    python3 scripts/write_store_header.py --store stores/experts.apex --out stores/expert_store_header.json
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apareto_loader.store import PackedStore, canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--store', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    with PackedStore(args.store, verify_payload=False) as store:
        header = store.header
    with args.out.open('xb') as out:
        out.write(canonical(header))
    print(f"wrote {args.out} (header_sha256 {header['header_sha256'][:16]})")


if __name__ == '__main__':
    main()
