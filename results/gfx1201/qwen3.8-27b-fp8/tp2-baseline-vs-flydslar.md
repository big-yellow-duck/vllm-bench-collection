# TP2 Baseline vs. Flydslar

This report compares the `tp2-baseline` and `tp2-flydslar` runs for
`Qwen/Qwen3.8-27B-FP8` on `gfx1201`.

## Benchmark setup

Both runs use the same benchmark configuration:

- Tensor parallelism: 2
- Workload: `simple-agent-multiturn-prefix-cache`
- Streams: 1, 2, 4, and 8
- Synthetic prompt: 2,048 tokens per turn
- Output: 512 tokens per turn
- Turns per conversation: 8

Output throughput is the benchmark's mean `output_tokens_per_second`.
Aggregate TTFT uses the mean `time_to_first_token_ms`; lower is better.
End-to-end (E2E) latency uses the mean `request_latency`, from request start
until the final response token; lower is better. Tail latency is also reported
as p99 TTFT grouped by turn, and therefore by prefix length.

## Aggregate results

The table below mixes all prefix lengths. It is useful for the overall
throughput picture, but it is not an input-length-normalized latency comparison.

| Streams | Baseline output tok/s | Flydslar output tok/s | Throughput change | Mean TTFT (ms), baseline → flydslar | TTFT change | Mean E2E latency (s), baseline → flydslar | E2E change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.11 | 6.38 | **+4.5% (1.05×)** | 1,687 → 2,997 | +77.7% | 84.03 → 80.60 | **−4.1%** |
| 2 | 11.91 | 13.05 | **+9.6% (1.10×)** | 2,460 → 4,260 | +73.2% | 86.02 → 78.68 | **−8.5%** |
| 4 | 11.85 | 13.38 | **+12.9% (1.13×)** | 3,059 → 4,641 | +51.7% | 87.66 → 76.56 | **−12.7%** |
| 8 | 21.96 | 25.35 | **+15.4% (1.15×)** | 8,624 → 6,166 | **−28.5%** | 100.31 → 82.95 | **−17.3%** |

```text
Output throughput (higher is better; each █ is approximately 2 tok/s)

1 stream   baseline ███ 6.11       flydslar ███ 6.38
2 streams  baseline ██████ 11.91    flydslar ███████ 13.05
4 streams  baseline ██████ 11.85    flydslar ███████ 13.38
8 streams  baseline ███████████ 21.96  flydslar █████████████ 25.35
```

## Input-length-normalized p99 TTFT

Each turn appends another prompt and response to the prefix, so prompt length
increases from about 2.1k to 20.1k tokens. The following comparisons group
requests by `turn_index`, which holds the prefix length constant within each
row. Negative values improve tail TTFT; positive values regress it.

Each prefix-length bucket contains only one to four requests per run. Therefore,
the displayed p99 is effectively the slowest request in that bucket and should
be treated as a tail observation rather than a statistically stable percentile.

### 1 stream

| Prompt tokens (approx.) | Baseline p99 TTFT (ms) | Flydslar p99 TTFT (ms) | Change |
|---:|---:|---:|---:|
| 2.1k | 1,569 | 741 | **−52.8%** |
| 4.7k | 1,042 | 1,598 | +53.4% |
| 7.2k | 1,294 | 2,326 | +79.7% |
| 9.8k | 1,321 | 2,535 | +91.9% |
| 12.4k | 1,611 | 2,555 | +58.6% |
| 15.0k | 2,652 | 3,500 | +32.0% |
| 17.5k | 1,825 | 3,603 | +97.4% |
| 20.1k | 2,183 | 3,466 | +58.8% |

### 2 streams

| Prompt tokens (approx.) | Baseline p99 TTFT (ms) | Flydslar p99 TTFT (ms) | Change |
|---:|---:|---:|---:|
| 2.1k | 1,377 | 1,300 | **−5.6%** |
| 4.7k | 2,037 | 2,702 | +32.6% |
| 7.2k | 2,546 | 4,282 | +68.2% |
| 9.8k | 2,595 | 4,802 | +85.1% |
| 12.4k | 3,153 | 4,658 | +47.7% |
| 15.0k | 5,226 | 6,846 | +31.0% |
| 17.5k | 3,621 | 7,430 | +105.2% |
| 20.1k | 4,269 | 6,364 | +49.1% |

### 4 streams

| Prompt tokens (approx.) | Baseline p99 TTFT (ms) | Flydslar p99 TTFT (ms) | Change |
|---:|---:|---:|---:|
| 2.1k | 1,860 | 2,253 | +21.2% |
| 4.7k | 3,024 | 3,832 | +26.7% |
| 7.2k | 3,758 | 6,055 | +61.1% |
| 9.8k | 3,852 | 6,885 | +78.8% |
| 12.4k | 4,761 | 6,507 | +36.7% |
| 15.0k | 6,361 | 9,895 | +55.6% |
| 17.5k | 5,379 | 9,900 | +84.1% |
| 20.1k | 6,383 | 9,461 | +48.2% |

### 8 streams

| Prompt tokens (approx.) | Baseline p99 TTFT (ms) | Flydslar p99 TTFT (ms) | Change |
|---:|---:|---:|---:|
| 2.1k | 4,424 | 4,056 | **−8.3%** |
| 4.7k | 7,529 | 6,957 | **−7.6%** |
| 7.2k | 12,126 | 11,273 | **−7.0%** |
| 9.8k | 9,966 | 9,077 | **−8.9%** |
| 12.4k | 12,300 | 8,941 | **−27.3%** |
| 15.0k | 17,766 | 14,886 | **−16.2%** |
| 17.5k | 15,674 | 12,282 | **−21.6%** |
| 20.1k | 17,740 | 9,995 | **−43.7%** |

## Interpretation

- Flydslar improves aggregate output throughput at every stream level, from **+4.5%** at one stream to **+15.4%** at eight streams. It also lowers aggregate E2E latency at every level.
- At one, two, and four streams, this comes with a substantial mean-TTFT regression. The prefix-length-normalized p99 comparison is also worse for nearly every bucket above the initial prompt.
- At eight streams, Flydslar improves all three aggregate measures: output throughput, mean TTFT, and E2E latency. Its p99 TTFT improves at every prefix length.

## Source data

- [`tp2-baseline/agent-multiturn-benchmarks.json`](tp2-baseline/agent-multiturn-benchmarks.json)
- [`tp2-flydslar/agent-multiturn-benchmarks.json`](tp2-flydslar/agent-multiturn-benchmarks.json)
