"""Audit the archived evidence without importing Torch or running a GPU."""

import csv
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
timings = {}
for suite in ("broad", "blind", "boundaries", "prospective2"):
    records = [
        json.loads(line)
        for line in (root / f"validation-{suite}.jsonl").read_text().splitlines()
    ]
    expected_shapes.update((r["M"], r["N"], r["K"]) for r in records)
    for record in records:
        timings.setdefault((record["M"], record["N"], record["K"]), record)
assert shapes == expected_shapes

controls = json.loads((root / "validation-controls.json").read_text())
assert len(controls) == 186
assert all(r["status"] == "pass" for r in controls)

with (root / "triton-vs-rowwise.csv").open() as handle:
    exported = list(csv.DictReader(handle))
assert len(exported) == len(rows)
assert {(int(r["M"]), int(r["N"]), int(r["K"])) for r in exported} == shapes
assert all(r["correctness"] == "pass" for r in exported)
for row in exported:
    record = timings[int(row["M"]), int(row["N"]), int(row["K"])]
    assert row["timing_source"] == record["timing_source"]
    for mode, metric in (("graph", "graph_gpu_us"), ("eager", "batch_wall_us")):
        torch = record["median_us"]["torch"][metric]
        triton = record["median_us"]["after"][metric]
        assert float(row[f"torch_{mode}_us"]) == torch
        assert float(row[f"triton_{mode}_us"]) == triton
        assert float(row[f"{mode}_speedup"]) == torch / triton

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
