"""Build plain Triton/RowWise tables from archived paired measurements."""

import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUITES = ("broad", "blind", "boundaries", "prospective2")
selected = {}
corpora = {}
for suite in SUITES:
    records = [
        json.loads(line)
        for line in (ROOT / f"validation-{suite}.jsonl").read_text().splitlines()
    ]
    corpora[suite] = len(records)
    for record in records:
        selected.setdefault((record["M"], record["N"], record["K"]), (suite, record))

correctness = [
    json.loads(line)
    for line in (ROOT / "full-correctness.jsonl").read_text().splitlines()
]
assert len(selected) == len(correctness) == 2513
assert set(selected) == {(r["M"], r["N"], r["K"]) for r in correctness}
assert all(r["status"] == "pass" for r in correctness)


def geomean(values):
    values = list(values)
    return math.exp(sum(math.log(value) for value in values) / len(values))


rows = []
for (m, n, k), (suite, record) in sorted(
    selected.items(), key=lambda entry: (entry[0][1], entry[0][2], entry[0][0])
):
    torch = record["median_us"]["torch"]
    triton = record["median_us"]["after"]
    assert not record["has_tuned_configs"]
    row = dict(M=m, N=n, K=k, corpus=suite, timing_source=record["timing_source"])
    for mode, metric in (("graph", "graph_gpu_us"), ("eager", "batch_wall_us")):
        assert torch[metric] > 0 and triton[metric] > 0
        row[f"torch_{mode}_us"] = torch[metric]
        row[f"triton_{mode}_us"] = triton[metric]
        row[f"{mode}_speedup"] = torch[metric] / triton[metric]
    row["paired_rounds"] = len(record["samples_us"]["torch"])
    expected_rounds = 3 if record["timing_source"] == "v3_equivalent_choice" else 5
    assert row["paired_rounds"] == len(record["samples_us"]["after"]) == expected_rounds
    row["correctness"] = "pass"
    rows.append(row)

with (ROOT / "triton-vs-rowwise.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

summary = []
for mode, label in (("graph", "Graph GPU time"), ("eager", "Batched eager completion")):
    ratios = [row[f"{mode}_speedup"] for row in rows]
    summary.append(
        f"| {label} | {geomean(row[f'torch_{mode}_us'] for row in rows):.2f} "
        f"| {geomean(row[f'triton_{mode}_us'] for row in rows):.2f} "
        f"| {geomean(ratios):.3f}× | {sum(ratio > 1 for ratio in ratios):,} "
        f"| {sum(ratio < 1 for ratio in ratios):,} |"
    )

table = [
    "| M | N | K | Torch graph µs | Triton graph µs | Graph speedup | Torch eager µs | Triton eager µs | Eager speedup |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    graph = f"{row['graph_speedup']:.3f}×"
    eager = f"{row['eager_speedup']:.3f}×"
    if row["graph_speedup"] < 1:
        graph = f"**{graph}**"
    if row["eager_speedup"] < 1:
        eager = f"**{eager}**"
    table.append(
        f"| {row['M']:,} | {row['N']:,} | {row['K']:,} "
        f"| {row['torch_graph_us']:.2f} | {row['triton_graph_us']:.2f} | {graph} "
        f"| {row['torch_eager_us']:.2f} | {row['triton_eager_us']:.2f} | {eager} |"
    )

sources = Counter(row["timing_source"] for row in rows)
assert sources == {
    "v3_equivalent_choice": 2450,
    "final_retimed": 33,
    "final_new_boundary": 30,
}
readme = (
    """# Triton vs Torch RowWise: full per-token FP8 comparison

On the archived R9700/gfx1201 workload, native Triton achieved **1.874× graph**
and **1.744× batched eager** geometric-mean speedup over Torch RowWise across
**2,513 distinct M/N/K shapes**. Triton was slower on **3 graph cases** and
**2 eager cases**; every distinct shape is retained below. All 2,513 shapes
passed the separate full-output correctness sweep; 186 additional controls passed.

These are **historical measurements against
[`6a1dbbe971`](https://github.com/big-yellow-duck/vllm/commit/6a1dbbe97195eccd06fbcb45d47b485ea6b6ecd1)**,
not a new run of the current [vLLM PR #61091](https://github.com/vllm-project/vllm/pull/61091)
tip or a model-throughput result. Both backends used the same prequantized FP8
operands, BF16 output, per-token/per-channel FP32 scales, and no bias.

| Measurement | Torch geomean µs | Triton geomean µs | Geomean speedup | Triton faster | Triton slower |
| --- | ---: | ---: | ---: | ---: | ---: |
"""
    + "\n".join(summary)
    + """

The summary gives every distinct shape equal weight. It is not weighted by a
model's layer frequency or token distribution. Speedup is **Torch time / Triton
time**: above 1× favors Triton, below 1× favors Torch. Latency columns are
geometric means of per-case medians; speedups and faster/slower counts use
unrounded data. The 2,450 reused cases have three paired rounds each; the
63 newly measured cases have five paired rounds each.

## Full comparison: all 2,513 shapes

Sorted by N, K, then M. **Bold speedups mark cases where Triton was slower**;
there are no exact ties in the unrounded records. Times are in microseconds.
Graph columns measure GPU graph replay; eager columns include CPU submission
and synchronized completion and must be compared separately.

[Download the complete CSV with full-precision timings, corpus, and timing provenance](triton-vs-rowwise.csv).

"""
    + "\n".join(table)
    + """

## Measurement scope and provenance

- **Hardware/software:** one Radeon AI PRO R9700 / gfx1201, 32 reported
  processors, about 32 GiB VRAM; Torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`,
  Triton `3.8.0`. Measurements were collected on 2026-10-27.
- **Operation:** the prequantized scaled-MM backend call. Operand creation and
  input quantization are outside timing. Normal backend work, including output
  allocation, remains inside the call. Compilation and warmup precede timing.
  Triton uses the untuned selector, without a shape-specific tuning map.
- **Graph timing:** HIP events around graph replay, with the same operands
  reused. Each event sample runs one replay; three event samples are reduced
  to their median per paired round. Captured graphs contain eight small calls
  or one large call, and times are divided by the call count. This is not a
  rotating-buffer cold-HBM benchmark.
- **Eager timing:** enqueue 16 backend calls, synchronize, and divide total wall
  time by 16. This includes host submission and completed GPU work. The arms'
  order is shuffled within each paired round. The table uses the stored median
  across the recorded paired rounds for each backend (three for reused cases,
  five for newly measured cases).
- **Deduplication:** the four corpora contain 2,590 records with 77 overlaps.
  For repeated shapes, take the first record in the fixed order `broad`,
  `blind`, `boundaries`, `prospective2`, independent of its timing. The table
  and summary therefore contain 2,513 distinct shapes rather than double-counting
  overlapping suites. All original repetitions and corpus records remain linked
  below; no slower case is excluded.
- **Timing reuse:** 2,450 selected records have `v3_equivalent_choice`
  provenance: prior paired timings were retained only for identical launch
  choices with byte-identical backend/GPU-launcher code. Another 33 records are
  `final_retimed` and 30 are `final_new_boundary`. The CSV preserves the source
  of each row. The [delta driver](delta.py) records the reuse rule and remeasurement
  procedure. This is not a wholly fresh measurement of every final-source
  case, and these small repetition counts do not establish statistical confidence.
- **Limits:** some corpora informed refinement, so these are regression checks
  rather than untouched final holdouts. Results are specific to the measured
  hardware, operand reuse, and shape distribution. They do not establish
  uniformly better performance or generalize to other architectures.

The table preserves the original screening measurements. Historical repeated
followups for two graph regressions at `(M,N,K)=(65,18944,3584)` and
`(65,19968,5632)` still showed Triton slower (about 14.7% and 3.9% latency).
Followups supplement the screen; they do not replace its less favorable rows.
See the [archived repeated measurements](residual-repeat.jsonl),
[repeat manifest](residual-repeat-manifest.json), and
[original summary](operator-summary.json) for the repeated and per-corpus results.
The per-corpus summary includes overlaps and consequently differs from the
distinct-shape summary at the top of this page.

## Correctness and raw evidence

The separate [full-output sweep](full-correctness.jsonl) passed on all 2,513
shapes: every output value compared with Torch RowWise, finite checks, and up
to 16 sampled rows × 32 columns compared with FP32 accumulation, using
`rtol=atol=1e-2`. Inputs were seeded per shape. These pass records do not include
maximum/mean error measurements.

The [186 controls](validation-controls.json) comprise 176 dtype/bias/scale-layout
cases, two compiled dynamic-shape configurations, and eight graph-replay cases
with changed inputs/scales. They are separate coverage; BF16/no-bias full-output
validation is not claimed for every dtype/layout combination over the full sweep.

- Original timing corpora: [broad](validation-broad.jsonl),
  [additional synthetic](validation-blind.jsonl),
  [boundaries](validation-boundaries.jsonl),
  [prospective set 2](validation-prospective2.jsonl).
- [Original scripts](validate.py), [input generator](experiment.py),
  [shape plan](plan.json), [correctness driver](correctness.py), and
  [environment/measurement manifest](validation-controls-manifest.json).
- [Frozen source hashes](frozen-candidate.json), [tested source](tested-source/),
  [full sweep status](full-correctness-status.json), and
  [original correctness log](full-correctness.log).
- [Checksum manifest](sha256.json) covers all other bundle files. Original
  records, drivers, and source snapshots are unchanged.

Using an existing uv-managed environment, reproduce this summary/table without
a GPU, then audit the saved evidence (refresh checksums after intentional edits):

```bash
uv run --no-project --python /path/to/.venv/bin/python \\
  /path/to/.venv/bin/python summarize_results.py
uv run --no-project --python /path/to/.venv/bin/python \\
  /path/to/.venv/bin/python audit.py
```

The original GPU drivers retain historical workspace paths/API names. For
historical reproduction, check out commit `6a1dbbe971` as
`WORKSPACE/vllm-fp8-heuristic-refine`, copy this bundle to
`WORKSPACE/logs/fp8-heuristic-refine-20261027`, and install its compatible ROCm
environment/compiled extensions using uv. Included `before/` and `baseline/`
snapshots supply comparison dependencies. Expose physical GPU1 alone using
`HIP_VISIBLE_DEVICES=1`. In a scratch copy, rename `full-correctness.jsonl`
before running `correctness.py` so its resume logic does not skip saved shapes.
Keep the four input corpora and historical tuning tables for the controls;
`validate.py controls` overwrites its output in the scratch copy.

Publication recomputed summaries and audited the saved data; it did not rerun
the GPU workload. AI assistance was used to assemble and audit this bundle.
The evidence supplements the focused dynamic-shape CI test; it does not add
these shapes to ongoing CI.
"""
)
(ROOT / "README.md").write_text(readme)
print("Wrote summary, all 2,513 Triton/RowWise table rows, and full-precision CSV.")
