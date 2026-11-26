"""Gate: the code profile's R* bundles are intact.

    python3 tests/check_bundles.py          (bundles under $APARETO_ARTIFACTS, default artifacts/)

For each bundle the profile names (config/profiles.json, profile `code`): every file is listed in its
MANIFEST.sha256 with the right SHA-256 and nothing unlisted is present, and the reference loads it.
Exit code 0 only if every bundle passes.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'scripts'))
from apareto_loader.selector import RuntimeSelector

artifacts = Path(os.environ.get('APARETO_ARTIFACTS', HERE / 'artifacts'))
bundles = json.loads((HERE / 'config' / 'profiles.json').read_text())['code']['bundles']
fails = []
for slot, relative in sorted(bundles.items(), key=lambda x: x[1]):
    path = artifacts / relative
    try:
        selector = RuntimeSelector(path)
        selector.manifest_check()
        result = {'manifest': 'ok', 'taps': len(selector.reference.taps), 'items': len(selector.reference.item_emb)}
        print(f'  [PASS] {relative} ({slot}): {result}')
    except Exception as error:
        print(f'  [FAIL] {relative} ({slot}): {type(error).__name__}: {error}')
        fails.append(relative)
print('PASS' if not fails else f'FAIL: {fails}')
sys.exit(1 if fails else 0)
