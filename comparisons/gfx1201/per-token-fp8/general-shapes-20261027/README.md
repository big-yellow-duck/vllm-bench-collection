# Per-token FP8: broad shape correctness on gfx1201

Historical validation on 2026-10-27 passed **2,513 distinct M/N/K shapes** and
**186 additional controls**. This bundle makes that evidence inspectable for
[vLLM PR #61091](https://github.com/vllm-project/vllm/pull/61091), particularly
[the shape-coverage review comment](https://github.com/vllm-project/vllm/pull/61091#discussion_r4494706183).

The records were collected against
[`6a1dbbe97195eccd06fbcb45d47b485ea6b6ecd1`](https://github.com/big-yellow-duck/vllm/commit/6a1dbbe97195eccd06fbcb45d47b485ea6b6ecd1).
The archived backend, launcher, selection rules, and pytest source match the
SHA-256 hashes recorded at measurement time. This is **historical correctness
evidence**, not a rerun of the current PR tip, a guarantee for every shape or
architecture, or a new performance/accuracy claim.

## Coverage and checks

| Evidence | Coverage | Outcome |
|---|---|---|
| [Full-output sweep](full-correctness.jsonl) | 2,513 unique shapes across 87 N/K pairs | All passed |
| M | 43 distinct values, 1 through 32,768 | Includes decode, prefill, and tile-boundary neighbors |
| N | 45 distinct values, 128 through 33,792 | Includes synthetic and model-derived dimensions |
| K | 35 distinct values, 128 through 32,768 | Includes short-K, long-K, and aspect-ratio variations |
| [Dtype/bias/layout controls](validation-controls.json) | 11 shapes × BF16/FP16 × bias on/off × four scale layouts = 176 | All passed sampled reference and finite checks |
| Compiled controls | Two cases, with and without tuning maps; changing M through one compiled callable | Both passed |
| Graph replay controls | Eight cases including split-K, persistent launches, and selected decode/prefill; inputs/scales mutated between replays | All passed |

The full sweep used FP8 E4M3FN operands, FP32 per-token/per-channel scales,
BF16 output, no bias, and the untuned selector. It compared every output value
against the Torch RowWise backend, checked that outputs were finite, and checked
up to 16 sampled rows × 32 columns against FP32 accumulation. The archived
drivers use `rtol=atol=1e-2`. Inputs were seeded per shape.

The four input corpora contain 2,170 broad, 150 additional synthetic, 150
boundary, and 120 prospective cases. Their union has 2,513 unique shapes;
overlaps are counted once in the full correctness sweep. Some corpora informed
selector refinement, so they are regression coverage rather than an untouched
final holdout. Dimensions span the ranges above; the ranges do not mean that
every combination or every intervening integer was tested.

Hardware: one Radeon AI PRO R9700, gfx1201, 32 reported processors, about 32 GiB
VRAM. Software recorded in the [controls manifest](validation-controls-manifest.json):
Torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`, Triton `3.8.0`. The broader
controls are separate from the BF16/no-bias full sweep; they do not establish
full-output correctness for every dtype/layout across all 2,513 shapes.

## Inspect the evidence

- [Full sweep status](full-correctness-status.json), [records](full-correctness.jsonl),
  and [original log](full-correctness.log).
- [Control results](validation-controls.json), [log](controls.log), and
  [measurement manifest](validation-controls-manifest.json).
- Original scripts: [correctness.py](correctness.py), [validate.py](validate.py),
  [experiment.py](experiment.py), and [shape plan](plan.json).
- [Frozen source manifest](frozen-candidate.json) and [tested source files](tested-source/).
- Four original corpus files and their manifests are retained beside the scripts
  so the sweep's shape union can be audited. They also contain historical timing
  samples; this README makes no performance claim from those samples.
- [sha256.json](sha256.json) covers every other file in this bundle.

Audit counts, shape-set equality, checksums, and historical source hashes without
a GPU. From this directory, using an existing uv-managed environment:

```bash
uv run --no-project --python /path/to/.venv/bin/python \
  /path/to/.venv/bin/python audit.py
```

This audit checks the saved evidence, not numerical computation on a GPU.

## Historical reproduction

The original scripts are preserved byte-for-byte, including workspace-relative
paths and historical API names. They are not a portable benchmark for the
current PR. To reconstruct their layout in a separate scratch workspace:

1. Check out vLLM commit `6a1dbbe97195eccd06fbcb45d47b485ea6b6ecd1`
   as `WORKSPACE/vllm-fp8-heuristic-refine`, with its compatible ROCm environment
   and compiled extensions installed using uv.
2. Copy this bundle to `WORKSPACE/logs/fp8-heuristic-refine-20261027`.
   The included `before/` and `baseline/` files supply the historical comparison
   dependencies. Keep the archived tuning tables from the vLLM commit for the
   tuned controls.
3. In that scratch copy, rename `full-correctness.jsonl` to
   `full-correctness.recorded.jsonl`. Otherwise the resumable script would skip
   all recorded shapes. Preserve the four `validation-*.jsonl` input corpora.
4. Activate the ROCm environment, expose physical GPU1 as the only GPU with
   `HIP_VISIBLE_DEVICES=1`, and run `correctness.py`, then `validate.py controls`,
   through the uv-managed `.venv/bin/python` as above. The controls command
   overwrites its output in the scratch copy. The original drivers also pin
   execution to an available CPU core and require the historical hardware/API
   assumptions.

Publication rechecked artifact integrity and source provenance. It did not
rerun this historical GPU workload. AI assistance was used to assemble and
audit the publication bundle.

## Relation to the review suggestion

The reviewed test loops over M with one `torch.compile(..., dynamic=True)`
callable. Keeping that loop checks changes in token count through the same
compiled function. The archived sweep provides broader numerical evidence;
it does not add those shapes to ongoing CI. A small parametrization of N/K
can complement the M loop without replacing its dynamic-shape purpose.
