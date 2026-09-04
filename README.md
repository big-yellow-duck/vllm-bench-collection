# vLLM AMD bench collection

Benchmarks for the same model under different serving configurations on AMD GPUs.

## Repository layout

Results, plots, and comparisons are organized by hardware, then model, then serving configuration:

```
results/<hardware>/<model>/<config>/
plots/<hardware>/<model>/<config>/
comparisons/<hardware>/<model>/
```

Examples:

```
results/gfx1201/qwen3.8-27b-fp8/tp2-baseline/agent-multiturn-benchmarks.json
plots/gfx1201/qwen3.8-27b-fp8/tp2-baseline/tps_vs_token_length.png
```

### Configuration naming

Config directory names encode the serving parameters:

```
tp<N>[-dp<N>][-<kernel-tag>[-<kernel-tag>...]]
```

| Tag | Meaning |
| --- | --- |
| `tp<N>` | Tensor parallel size |
| `dp<N>` | Data parallel size |
| `baseline` | No extra kernel patches |
| `rdna4blockscalemm` | RDNA4 block-scale MM kernel |
| `rdna4fp8blockscalemm` | RDNA4 FP8 block-scale MM |
| `rdna4-fp8vblockscalemm` | RDNA4 FP8 block-scale MM variant |
| `tritonsplitkvfp8` | Triton split-KV FP8 |
| `flydslar` | Flash-attn / `flydslar` optimization |

### Hardware

| Directory | GPU |
| --- | --- |
| `gfx1201` | AMD Strix Halo / RDNA 3.5 integrated GPU |
| `gfx1151` | AMD dGPU (Gemma result) |

## Running benchmarks

1. Start the vLLM server with the desired configuration.
2. Run the benchmark driver against it:

```bash
.venv/bin/python -m guidellm.benchmark \
  --config benchmarks-configs/simple-agent-multiturn-prefix-cache.json \
  --output results/<hardware>/<model>/<config>/agent-multiturn-benchmarks.json
```

Or point the driver at the server endpoint directly and copy the resulting JSON into the right config directory.

## Generating plots

Use the plotting scripts under `plotting/`. Each script takes one result JSON and writes one plot PNG:

```bash
.venv/bin/python plotting/plot_tps_vs_token_length.py \
  results/gfx1201/qwen3.8-27b-fp8/tp2-baseline/agent-multiturn-benchmarks.json \
  --output plots/gfx1201/qwen3.8-27b-fp8/tp2-baseline/tps_vs_token_length.png

.venv/bin/python plotting/plot_ttft_vs_token_length.py \
  results/gfx1201/qwen3.8-27b-fp8/tp2-baseline/agent-multiturn-benchmarks.json \
  --output plots/gfx1201/qwen3.8-27b-fp8/tp2-baseline/ttft_vs_token_length.png
```

Keep the plot output path mirroring the result input path so comparisons across configs are easy.

## Comparing configurations

For side-by-side comparisons across configs, use `comparisons/<hardware>/<model>/`:

```bash
.venv/bin/python plotting/plot_tps_vs_token_length.py \
  --compare \
  results/gfx1201/qwen3.8-27b-fp8/tp2-baseline/agent-multiturn-benchmarks.json \
  results/gfx1201/qwen3.8-27b-fp8/tp4-baseline/agent-multiturn-benchmarks.json \
  --output comparisons/gfx1201/qwen3.8-27b-fp8/tps_vs_token_length.png
```

(Adjust the comparison invocation once the plotting scripts support `--compare`.)

## Config definitions

Workload definitions live in `benchmarks-configs/`:

- `simple-agent-multiturn-prefix-cache.json` — agent-style multiturn benchmark used across most configs.
- See `benchmarks-configs/README.md` for parameter details.

## Archive policy

If a newer run of the same config is produced, keep the most recent run as `agent-multiturn-benchmarks.json` and archive the previous run as `agent-multiturn-benchmarks.YYYYMMDD-HHMMSS.json`.
