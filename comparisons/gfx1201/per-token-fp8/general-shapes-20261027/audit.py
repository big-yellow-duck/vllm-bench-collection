"""Audit the archived evidence without importing Torch or running a GPU."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
checksums = json.loads((root / "sha256.json").read_text())
for name, expected in checksums.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name

rows = [
    json.loads(line)
    for line in (root / "full-correctness.jsonl").read_text().splitlines()
]
shapes = {(r["M"], r["N"], r["K"]) for r in rows}
assert len(rows) == len(shapes) == 2513
assert all(r["status"] == "pass" for r in rows)
assert json.loads((root / "full-correctness-status.json").read_text())["cases"] == len(
    rows
)

expected_shapes = set()
for suite in ("broad", "blind", "boundaries", "prospective2"):
    records = [
        json.loads(line)
        for line in (root / f"validation-{suite}.jsonl").read_text().splitlines()
    ]
    expected_shapes.update((r["M"], r["N"], r["K"]) for r in records)
assert shapes == expected_shapes

controls = json.loads((root / "validation-controls.json").read_text())
assert len(controls) == 186
assert all(r["status"] == "pass" for r in controls)

frozen = json.loads((root / "frozen-candidate.json").read_text())
for name, expected in frozen["sha256"].items():
    assert (
        hashlib.sha256((root / "tested-source" / name).read_bytes()).hexdigest()
        == expected
    ), name

manifest = json.loads((root / "validation-controls-manifest.json").read_text())
for name, expected in manifest["source_sha256"].items():
    source = (
        root / "validate.py"
        if name.endswith("/validate.py")
        else root / "tested-source" / name.split("/vllm-fp8-heuristic-refine/")[1]
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, name

print(
    "PASS: artifact checksums, historical source hashes, "
    "2513 unique passing shapes, 186 passing controls"
)
for axis in ("M", "N", "K"):
    values = {r[axis] for r in rows}
    print(f"{axis}: {min(values)}..{max(values)}, {len(values)} distinct values")
