# Plotting scripts

Generate output throughput plots against the observed input and output token
lengths:

```bash
.venv/bin/python plotting/plot_tps_vs_token_length.py \
  results/gfx1151/gemma4-26b-a4b-it-q4km/agent-multiturn-benchmarks.json \
  --output plots/gfx1151/gemma4-26b-a4b-it-q4km/tps_vs_token_length.png
```

The script creates two panels, with each configured concurrency (`streams`) as a
separate line:

- Left: mean per-request output TPS from request-level
  `output_tokens_per_second`.
- Right: concurrency-summed output TPS calculated as the sum of request-level
  `output_tokens_per_second` values within each input-length bucket. This makes
  two simultaneous requests at 18 TPS show approximately 36 aggregate TPS.

Nearby request lengths are grouped within 256 tokens by default so concurrent
requests from the same workload stage do not create noisy lines. Adjust that
behavior with `--group-tolerance 0` for exact lengths.

Generate the analogous TTFT chart:

```bash
.venv/bin/python plotting/plot_ttft_vs_token_length.py \
  results/gfx1151/gemma4-26b-a4b-it-q4km/agent-multiturn-benchmarks.json \
  --output plots/gfx1151/gemma4-26b-a4b-it-q4km/ttft_vs_token_length.png
```

The TTFT chart uses mean TTFT on the left and p95 TTFT on the right. TTFT is a
latency metric, so it is not summed across concurrent requests.

## Compare TP2 and TP4 step by step

Use `plot_compare_multiturn.py` to compare multiple result sets. The command
below compares baseline and FlyDSL-AR at TP2 and TP4. It creates separate TP2
and TP4 figures. Each figure has a 2×2 layout with one subplot for each
configured stream level and two lines per subplot: baseline and FlyDSL-AR.

```bash
.venv/bin/python plotting/plot_compare_multiturn.py \
  --report "TP2 baseline=results/gfx1201/qwen3.8-27b-fp8/tp2-baseline/agent-multiturn-benchmarks.json" \
  --report "TP2 FlyDSL-AR=results/gfx1201/qwen3.8-27b-fp8/tp2-flydslar/agent-multiturn-benchmarks.json" \
  --report "TP4 baseline=results/gfx1201/qwen3.8-27b-fp8/tp4-baseline/agent-multiturn-benchmarks.json" \
  --report "TP4 FlyDSL-AR=results/gfx1201/qwen3.8-27b-fp8/tp4-flydslar/agent-multiturn-benchmarks.json" \
  --output-dir plots/gfx1201/qwen3.8-27b-fp8/tp2-vs-tp4-flydslar
```

The script recognizes case-insensitive `TP<number>` tokens in report labels and
uses them to split the output. It writes output-TPS, TTFT, and E2E-latency charts
for each TP case. The x-axis is the exact conversation `turn_index`, with the
corresponding mean prompt length shown below it. Values from simultaneous
requests at the same step are averaged within a line. Color identifies the
result set; each subplot identifies the configured stream count.
