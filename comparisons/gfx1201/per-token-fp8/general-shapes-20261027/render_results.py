"""Render coverage charts, tables, and CSVs from the archived correctness records."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
ROWS = [
    json.loads(line)
    for line in (ROOT / "full-correctness.jsonl").read_text().splitlines()
]
CONTROLS = json.loads((ROOT / "validation-controls.json").read_text())
assert len(ROWS) == 2513 and all(row["status"] == "pass" for row in ROWS)
assert len(CONTROLS) == 186 and all(row["status"] == "pass" for row in CONTROLS)
assert len({(row["M"], row["N"], row["K"]) for row in ROWS}) == len(ROWS)


def launch_path(row):
    config = row["config"]
    if config.get("engine") == "shared":
        return "Shared GEMM"
    if config.get("variant") == "persistent":
        return "Persistent GEMM"
    if config.get("split_k", 1) > 1:
        return "Split-K GEMM"
    return "Contiguous GEMM"


def write_csv(name, fields, records):
    with (ROOT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


by_pair = defaultdict(list)
by_m = defaultdict(list)
for row in ROWS:
    by_pair[(row["N"], row["K"])].append(row)
    by_m[row["M"]].append(row)
pairs = sorted(by_pair)
ms = sorted(by_m)
paths = ["Shared GEMM", "Contiguous GEMM", "Persistent GEMM", "Split-K GEMM"]
colors = ["#177e89", "#3675bd", "#8656a2", "#d07721"]
path_counts = Counter(launch_path(row) for row in ROWS)
pair_rows = [
    dict(
        N=n,
        K=k,
        cases=len(by_pair[(n, k)]),
        passed=len(by_pair[(n, k)]),
        failed=0,
        M_min=min(row["M"] for row in by_pair[(n, k)]),
        M_max=max(row["M"] for row in by_pair[(n, k)]),
        M_values=" ".join(
            str(row["M"]) for row in sorted(by_pair[(n, k)], key=lambda r: r["M"])
        ),
    )
    for n, k in pairs
]
m_rows = [dict(M=m, pairs=len(by_m[m]), passed=len(by_m[m]), failed=0) for m in ms]
write_csv("coverage-by-pair.csv", list(pair_rows[0]), pair_rows)
write_csv("coverage-by-m.csv", list(m_rows[0]), m_rows)
write_csv(
    "shape-results.csv",
    ["M", "N", "K", "status", "launch_path"],
    [
        dict(
            M=row["M"],
            N=row["N"],
            K=row["K"],
            status=row["status"],
            launch_path=launch_path(row),
        )
        for row in sorted(ROWS, key=lambda r: (r["N"], r["K"], r["M"]))
    ],
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)
fig = plt.figure(figsize=(16, 10), layout="constrained")
grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
ax = fig.add_subplot(grid[0, 0])
scatter = ax.scatter(
    [n for n, k in pairs],
    [k for n, k in pairs],
    s=[len(by_pair[pair]) * 8 for pair in pairs],
    c=[len(by_pair[pair]) for pair in pairs],
    cmap="viridis",
    edgecolors="white",
    linewidths=0.6,
    alpha=0.9,
)
ax.set(
    xscale="log",
    yscale="log",
    xlabel="N: output channels (log scale)",
    ylabel="K: reduction dimension (log scale)",
    title="87 tested N/K pairs",
)
fig.colorbar(scatter, ax=ax, label="Distinct M values tested per pair", shrink=0.8)
ax.grid(alpha=0.18)
ax = fig.add_subplot(grid[0, 1])
bars = ax.barh(
    paths[::-1], [path_counts[name] for name in paths[::-1]], color=colors[::-1]
)
ax.bar_label(bars, labels=[f"{path_counts[name]:,}" for name in paths[::-1]], padding=6)
ax.set(
    xlabel="Unique shape records",
    title="Selected launch paths in the full sweep",
    xlim=(0, max(path_counts.values()) * 1.18),
)
ax.text(
    0.99,
    0.03,
    "All 2,513 records passed\nColors identify launch paths, not speed.",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
)
ax = fig.add_subplot(grid[1, :])
bars = ax.bar(range(len(ms)), [len(by_m[m]) for m in ms], color="#177e89")
ax.bar_label(bars, padding=2, fontsize=8, rotation=90)
ax.set(
    xticks=range(len(ms)),
    xticklabels=[f"{m:,}" for m in ms],
    xlabel="M: token rows (categorical positions; numeric spacing is not preserved)",
    ylabel="N/K pairs tested",
    ylim=(0, 100),
    title="43 tested token counts, including boundary neighbors",
)
ax.tick_params(axis="x", rotation=65, labelsize=8)
ax.grid(axis="y", alpha=0.18)
fig.suptitle(
    "Per-token FP8 correctness coverage | 2,513 shapes passed + 186 controls passed\n"
    "R9700 / gfx1201 · historical source 6a1dbbe971 · 2026-10-27",
    fontsize=17,
)
for extension in ("png", "svg"):
    fig.savefig(ROOT / f"coverage-overview.{extension}", dpi=160)
plt.close(fig)

matrix = np.zeros((len(pairs), len(ms)), dtype=int)
lookup = {(row["N"], row["K"], row["M"]): row for row in ROWS}
for i, (n, k) in enumerate(pairs):
    for j, m in enumerate(ms):
        if (n, k, m) in lookup:
            matrix[i, j] = paths.index(launch_path(lookup[n, k, m])) + 1
assert np.count_nonzero(matrix) == 2513
fig, ax = plt.subplots(figsize=(16, 23), layout="constrained")
cmap = ListedColormap(["#edf0f3", *colors])
ax.imshow(
    matrix,
    aspect="auto",
    interpolation="nearest",
    cmap=cmap,
    norm=BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N),
)
ax.set(
    xticks=range(len(ms)),
    xticklabels=[f"{m:,}" for m in ms],
    yticks=range(len(pairs)),
    yticklabels=[f"N={n:,}  K={k:,}" for n, k in pairs],
    xlabel="M: token rows",
    ylabel="Tested N/K pairs (sorted by N, then K)",
)
ax.tick_params(axis="x", rotation=90, labelsize=9)
ax.tick_params(axis="y", labelsize=8)
ax.set_xticks(np.arange(-0.5, len(ms), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(pairs), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.4)
ax.tick_params(which="minor", bottom=False, left=False)
fig.suptitle(
    "Exact tested-shape matrix: 87 N/K pairs × 43 M values\n"
    "2,513 colored cells passed · gray cells were not tested in this sweep",
    fontsize=17,
)
fig.legend(
    handles=[
        Patch(color="#edf0f3", label="Not tested"),
        *[Patch(color=color, label=name) for color, name in zip(colors, paths)],
    ],
    loc="outside lower center",
    ncol=5,
    frameon=False,
)
for extension in ("png", "svg"):
    fig.savefig(ROOT / f"shape-matrix.{extension}", dpi=160)
plt.close(fig)

text = [
    "## Coverage at a glance",
    "",
    "![Coverage overview](coverage-overview.png)",
    "",
    (
        "[Download overview: SVG](coverage-overview.svg). Charts count saved correctness records; "
        "they do not show latency or numerical error magnitudes."
    ),
    "",
    "### Validation results",
    "",
    "| Check group | Shapes / configurations | Output / bias | Reference and check | Passed | Failed |",
    "|---|---|---|---|---:|---:|",
    "| Full-output sweep | 2,513 unique M/N/K shapes; 87 N/K pairs | BF16; no bias; untuned selector | Every output vs Torch RowWise; finite outputs; up to 16 × 32 sampled FP32 values | 2,513 | 0 |",
    "| Dtype / bias / scale layout | 11 shapes × 2 dtypes × 2 bias settings × 4 layouts | BF16 / FP16; bias off / on | Sampled FP32 reference and finite outputs, repeated three times | 176 | 0 |",
    "| Compiled dynamic shapes | Two configurations; M changes through one compiled callable | BF16; tuning maps absent / present | Sampled FP32; full output vs eager execution | 2 | 0 |",
    "| Graph replay | Eight launch cases; three replays with mutated inputs / scales | BF16; selected and explicit launches | Sampled FP32; full output vs shared Triton implementation | 8 | 0 |",
    "",
    (
        "All comparison tolerances are `rtol=atol=1e-2`. The raw records report pass/fail, "
        "not maximum or mean numerical error; no error magnitudes are inferred. "
        "FP8 E4M3FN operands use FP32 per-token/per-channel scales. Inputs are seeded per shape."
    ),
    "",
    "### Shape ranges",
    "",
    "| Dimension | Meaning | Distinct values | Minimum | Maximum |",
    "|---|---|---:|---:|---:|",
]
for axis, meaning in [
    ("M", "Token rows"),
    ("N", "Output channels"),
    ("K", "Reduction dimension"),
]:
    values = {row[axis] for row in ROWS}
    text.append(
        f"| {axis} | {meaning} | {len(values)} | {min(values):,} | {max(values):,} |"
    )
text += [
    "",
    (
        "The ranges are bounds of observed values, not exhaustive Cartesian coverage. "
        "The full sweep is BF16 without bias; the other dtypes and layouts are separate controls."
    ),
    "",
    "### Input corpora and overlap",
    "",
    "| Corpus | Recorded cases | New unique shapes in this table order | Overlap with earlier rows |",
    "|---|---:|---:|---:|",
]
seen = set()
for name, label in [
    ("broad", "Broad synthetic / model-derived"),
    ("blind", "Additional synthetic"),
    ("boundaries", "Boundary neighbors"),
    ("prospective2", "Prospective set 2"),
]:
    records = [
        json.loads(line)
        for line in (ROOT / f"validation-{name}.jsonl").read_text().splitlines()
    ]
    shapes = {(row["M"], row["N"], row["K"]) for row in records}
    text.append(
        f"| [{label}](validation-{name}.jsonl) | {len(records):,} | {len(shapes - seen):,} | {len(shapes & seen):,} |"
    )
    seen |= shapes
text += [
    "| **Total** | **2,590** | **2,513** | **77** |",
    "",
    (
        "Overlap is counted relative to preceding rows, not as independent pairwise overlaps. "
        "Some corpora informed selector refinement, so this is regression coverage rather than "
        "an untouched final holdout."
    ),
    "",
    "### Launch paths exercised",
    "",
    "| Selected launch path | Unique shapes | Share of full sweep | Passed | Failed |",
    "|---|---:|---:|---:|---:|",
]
for name in paths:
    count = path_counts[name]
    text.append(f"| {name} | {count:,} | {count / len(ROWS):.1%} | {count:,} | 0 |")
text += [
    "",
    (
        "Paths are derived from each record's saved configuration. Split-K includes "
        "18 shapes using four splits and 12 using eight; persistent covers 107 shapes. "
        "These are correctness counts, not speedups."
    ),
    "",
    "### Exact coverage matrix",
    "",
    (
        "Each colored cell is one tested, passing `(M, N, K)` combination. Gray means "
        "not tested in this sweep. The matrix has 3,741 possible displayed cells; "
        "2,513 were tested. This fraction describes only this displayed grid, not all possible shapes."
    ),
    "",
    "<details>",
    "<summary>Expand the full 87-pair × 43-token-count matrix</summary>",
    "",
    "![Exact tested-shape coverage](shape-matrix.png)",
    "",
    "[Open full-resolution PNG](shape-matrix.png) · [Download SVG](shape-matrix.svg)",
    "",
    "</details>",
    "",
    "### Results for every N/K pair",
    "",
    (
        "[Download pair summary](coverage-by-pair.csv) · [Download every shape result](shape-results.csv). "
        "The pair CSV includes the exact tested M values; min/max do not imply intervening values were tested."
    ),
    "",
    "<details>",
    "<summary>Expand all 87 N/K pairs</summary>",
    "",
    "| N | K | Tested M values | Min M | Max M | Passed | Failed |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for row in pair_rows:
    text.append(
        f"| {row['N']:,} | {row['K']:,} | {row['cases']} | {row['M_min']:,} | {row['M_max']:,} | {row['passed']} | 0 |"
    )
text += [
    "| **Total** | | **2,513 shape records** | | | **2,513** | **0** |",
    "",
    "</details>",
    "",
    "### Results by token count",
    "",
    "[Download token-count summary](coverage-by-m.csv).",
    "",
    "<details>",
    "<summary>Expand all 43 M values</summary>",
    "",
    "| M | N/K pairs tested | Passed | Failed |",
    "|---:|---:|---:|---:|",
]
for row in m_rows:
    text.append(f"| {row['M']:,} | {row['pairs']} | {row['passed']} | 0 |")
text += [
    "| **Total** | **2,513 shape records** | **2,513** | **0** |",
    "",
    "</details>",
    "",
    "### Hardware and software",
    "",
    "| Item | Recorded environment |",
    "|---|---|",
    "| GPU | Radeon AI PRO R9700 / gfx1201; one physical GPU for operator validation |",
    "| Resources | 32 reported processors; about 32 GiB VRAM |",
    "| Torch | `2.13.0+rocm10.0.0` |",
    "| HIP | `7.15.26333` |",
    "| Triton | `3.8.0` |",
    "| Tested source | `6a1dbbe97195eccd06fbcb45d47b485ea6b6ecd1` |",
    "",
    (
        "See the [original environment manifest](validation-controls-manifest.json). "
        "Charts, CSVs, and the tables above are generated from the archived records by "
        "[render_results.py](render_results.py). No new GPU measurements were collected."
    ),
    "",
    "```bash",
    "uv run --no-project --python /path/to/.venv/bin/python \\",
    "  /path/to/.venv/bin/python render_results.py",
    "```",
    "",
    (
        "The renderer requires Matplotlib and NumPy. It changes generated documentation "
        "and figures; refresh `sha256.json` after intentional regeneration before running the integrity audit."
    ),
    "",
]
readme = (ROOT / "README.md").read_text()
start = readme.index("## Coverage")
end = readme.index("## Inspect the evidence")
text = [
    "| " + " | ".join(line.strip("|").split("|")) + " |"
    if line.startswith("|") and set(line) <= set("|-:")
    else line
    for line in text
]
(ROOT / "README.md").write_text(readme[:start] + "\n".join(text) + "\n" + readme[end:])
for svg in (ROOT / "coverage-overview.svg", ROOT / "shape-matrix.svg"):
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
print("Rendered two charts (PNG/SVG), three CSVs, and comprehensive README tables.")
