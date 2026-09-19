# TP2 Baseline vs. RDNA4 FP8 Block-Scale GEMM

This report compares the `tp2-baseline` and `tp2-rdna4fp8blockscalemm` runs for
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

| Streams | Baseline output tok/s | RDNA4 output tok/s | Throughput change | Mean TTFT (ms), baseline → RDNA4 | TTFT change | Mean E2E latency (s), baseline → RDNA4 | E2E change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.11 | 6.30 | **+3.1% (1.03×)** | 1,687 → 3,211 | +90.3% | 84.03 → 81.68 | **−2.8%** |
| 2 | 11.91 | 12.92 | **+8.5% (1.08×)** | 2,460 → 4,276 | +73.8% | 86.02 → 79.48 | **−7.6%** |
| 4 | 11.85 | 24.60 | **+107.5% (2.08×)** | 3,059 → 6,288 | +105.5% | 87.66 → 81.48 | **−7.0%** |
| 8 | 21.96 | 25.26 | **+15.0% (1.15×)** | 8,624 → 6,367 | **−26.2%** | 100.31 → 82.94 | **−17.3%** |

```text
Output throughput (higher is better; each █ is approximately 2 tok/s)

1 stream   baseline ███ 6.11       RDNA4 ███ 6.30
2 streams  baseline ██████ 11.91    RDNA4 ███████ 12.92
4 streams  baseline ██████ 11.85    RDNA4 ████████████ 24.60
8 streams  baseline ███████████ 21.96  RDNA4 █████████████ 25.26
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

| Prompt tokens (approx.) | Baseline p99 TTFT (ms) | RDNA4 p99 TTFT (ms) | Change |
|---:|---:|---:|---:|
| 2.1k | 1,569 | 758 | **−51.7%** |
| 4.7k | 1,042 | 1,559 | +49.6% |
| 7.2k | 1,294 | 2,294 | +77.2% |
| 9.8k | 1,321 | 2,488 | +88.3% |
| 12.4k | 1,611 | 2,494 | +54.8% |
| 15.0k | 2,652 | 3,485 | +31.4% |
| 17.5k | 1,825 | 3,589 | +96.6% |
| 20.1k | 2,183 | 4,916 | +125.2% |

### 2 streams

| Prompt tokens (approx.) | Baseline p99 TTFT (ms) | RDNA4 p99 TTFT (ms) | Change |
|---:|---:|---:|---:|
| 2.1k | 1,377 | 1,370 | **−0.5%** |
| 4.7k | 2,037 | 2,784 | +36.7% |
| 7.2k | 2,546 | 4,233 | +66.3% |
| 9.8k | 2,595 | 4,793 | +84.7% |
| 12.4k | 3,153 | 4,570 | +44.9% |
| 15.0k | 5,226 | 6,946 | +32.9% |
| 17.5k | 3,621 | 7,345 | +102.8% |
| 20.1k | 4,269 | 6,487 | +51.9% |

### 4 streams

| Prompt tokens (approx.) | Baseline p99 TTFT (ms) | RDNA4 p99 TTFT (ms) | Change |
|---:|---:|---:|---:|
| 2.1k | 1,860 | 2,972 | +59.8% |
| 4.7k | 3,024 | 4,778 | +58.0% |
| 7.2k | 3,758 | 7,861 | +109.2% |
| 9.8k | 3,852 | 8,363 | +117.1% |
| 12.4k | 4,761 | 8,703 | +82.8% |
| 15.0k | 6,361 | 12,511 | +96.7% |
| 17.5k | 5,379 | 12,231 | +127.4% |
| 20.1k | 6,383 | 10,028 | +57.1% |

### 8 streams

| Prompt tokens (approx.) | Baseline p99 TTFT (ms) | RDNA4 p99 TTFT (ms) | Change |
|---:|---:|---:|---:|
| 2.1k | 4,424 | 4,028 | **−8.9%** |
| 4.7k | 7,529 | 6,843 | **−9.1%** |
| 7.2k | 12,126 | 11,623 | **−4.1%** |
| 9.8k | 9,966 | 9,155 | **−8.1%** |
| 12.4k | 12,300 | 10,062 | **−18.2%** |
| 15.0k | 17,766 | 16,199 | **−8.8%** |
| 17.5k | 15,674 | 12,194 | **−22.2%** |
| 20.1k | 17,740 | 11,589 | **−34.7%** |

## Interpretation

- RDNA4 improves aggregate output throughput at every stream level and lowers aggregate E2E latency at every level.
- The throughput highlight is four streams: **+107.5% output throughput** and **+118.2% total-token throughput**, with a 7.0% E2E reduction. It also has a large TTFT regression, both in aggregate and across every prefix-length p99 bucket.
- At eight streams, RDNA4 improves output throughput by 15.0%, mean TTFT by 26.2%, E2E latency by 17.3%, and p99 TTFT at every prefix length.

## Source data

- [`tp2-baseline/agent-multiturn-benchmarks.json`](tp2-baseline/agent-multiturn-benchmarks.json)
- [`tp2-rdna4fp8blockscalemm/agent-multiturn-benchmarks.json`](tp2-rdna4fp8blockscalemm/agent-multiturn-benchmarks.json)
