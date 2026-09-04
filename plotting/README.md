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
