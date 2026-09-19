# TP4 Baseline vs. Flydslar

This report compares the `tp4-baseline` and `tp4-flydslar` runs for
`Qwen/Qwen3.8-27B-FP8` on `gfx1201`.

## Benchmark setup

Both runs use the same benchmark configuration:

- Tensor parallelism: 4
- Workload: `simple-agent-multiturn-prefix-cache`
- Streams: 1, 2, 4, and 8
- Synthetic prompt: 2,048 tokens per turn
- Output: 512 tokens per turn
- Turns per conversation: 8

Output throughput is reported as the benchmark's mean
`output_tokens_per_second`. The aggregate TTFT view uses the mean
`time_to_first_token_ms`; lower is better. Tail latency is also reported
below as p99 TTFT grouped by turn, and therefore by prefix length.
End-to-end (E2E) latency uses the mean `request_latency`, measured from request
start until the final response token; lower is better.

## Aggregate results

The table below mixes all prefix lengths. It is useful for the overall
throughput picture, but it is not an input-length-normalized latency comparison.

> **Comparability note:** this original comparison is not normalized for
> achieved request concurrency. The baseline reached 2.04 concurrent requests
> at the four-stream target, while the previous Flydslar run reached 4.00.
> Therefore, the four-stream throughput delta cannot be attributed to Flydslar
> alone. See
> [`tp4-flydslar-rerun-vs-previous.md`](tp4-flydslar-rerun-vs-previous.md).

| Streams | Baseline output tok/s | Flydslar output tok/s | Throughput change | Mean TTFT (ms), baseline → flydslar | TTFT change | Mean E2E latency (s), baseline → flydslar | E2E change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.87 | 7.03 | −10.6% | 2,340 → 1,958 | **−16.3%** | 65.14 → 73.12 | +12.2% |
| 2 | 15.09 | 14.63 | −3.1% | 3,066 → 2,829 | **−7.8%** | 67.86 → 70.13 | +3.3% |
| 4 | 15.10 | 28.37 | **+87.9% (1.88×)** | 3,391 → 4,150 | +22.4% | 69.17 → 70.78 | +2.3% |
| 8 | 27.96 | 29.23 | **+4.6% (1.05×)** | 5,226 → 4,123 | **−21.1%** | 75.93 → 71.32 | **−6.1%** |

```text
Output throughput (higher is better; each █ is approximately 2 tok/s)

1 stream   baseline ████ 7.87      flydslar ████ 7.03
2 streams  baseline ████████ 15.09  flydslar ███████ 14.63
4 streams  baseline ████████ 15.10  flydslar ██████████████ 28.37
8 streams  baseline ██████████████ 27.96  flydslar ███████████████ 29.23
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
| 2.1k | 388 | 564 | +45.6% |
| 4.7k | 2,288 | 1,214 | **−47.0%** |
| 7.2k | 2,556 | 1,554 | **−39.2%** |
| 9.8k | 2,477 | 1,696 | **−31.5%** |
| 12.4k | 2,441 | 1,825 | **−25.2%** |
| 15.0k | 2,977 | 2,309 | **−22.4%** |
| 17.5k | 2,878 | 2,404 | **−16.5%** |
| 20.1k | 2,716 | 1,963 | **−27.7%** |

### 2 streams

| Prompt tokens (approx.) | Baseline p99 TTFT (ms) | Flydslar p99 TTFT (ms) | Change |
|---:|---:|---:|---:|
| 2.1k | 617 | 887 | +43.9% |
| 4.7k | 2,928 | 1,989 | **−32.1%** |
| 7.2k | 3,867 | 2,923 | **−24.4%** |
| 9.8k | 3,786 | 3,132 | **−17.3%** |
| 12.4k | 3,686 | 3,048 | **−17.3%** |
| 15.0k | 4,872 | 4,484 | **−8.0%** |
| 17.5k | 4,654 | 4,345 | **−6.6%** |
| 20.1k | 4,330 | 4,114 | **−5.0%** |

### 4 streams

| Prompt tokens (approx.) | Baseline p99 TTFT (ms) | Flydslar p99 TTFT (ms) | Change |
|---:|---:|---:|---:|
| 2.1k | 1,488 | 1,760 | +18.3% |
| 4.7k | 3,803 | 3,455 | **−9.2%** |
| 7.2k | 5,165 | 5,446 | +5.4% |
| 9.8k | 4,999 | 5,564 | +11.3% |
| 12.4k | 4,731 | 5,556 | +17.4% |
| 15.0k | 6,656 | 8,730 | +31.2% |
| 17.5k | 6,256 | 7,358 | +17.6% |
| 20.1k | 5,732 | 6,414 | +11.9% |

### 8 streams

| Prompt tokens (approx.) | Baseline p99 TTFT (ms) | Flydslar p99 TTFT (ms) | Change |
|---:|---:|---:|---:|
| 2.1k | 1,942 | 2,624 | +35.1% |
| 4.7k | 7,013 | 4,918 | **−29.9%** |
| 7.2k | 9,612 | 7,816 | **−18.7%** |
| 9.8k | 8,975 | 6,212 | **−30.8%** |
| 12.4k | 8,751 | 5,869 | **−32.9%** |
| 15.0k | 13,118 | 9,979 | **−23.9%** |
| 17.5k | 10,650 | 7,367 | **−30.8%** |
| 20.1k | 10,569 | 6,469 | **−38.8%** |

## Interpretation

- At one and two streams, Flydslar reduces aggregate mean TTFT by 16.3% and 7.8%, respectively, but output throughput is slightly lower. Once the prefix exceeds 2.1k tokens, every grouped p99 TTFT comparison improves.
- At four streams, the recorded Flydslar throughput is 87.9% higher, but this is not a valid kernel-only comparison because it achieved roughly twice the baseline's concurrency. Its grouped p99 TTFT is worse for all but the 4.7k-token bucket, especially above 12k tokens.
- At eight streams, Flydslar has a modest 4.6% output-throughput gain and reduces aggregate mean TTFT by 21.1%. The grouped p99 TTFT also improves from 4.7k through 20.1k tokens.

## Source data

- [`tp4-baseline/agent-multiturn-benchmarks.json`](tp4-baseline/agent-multiturn-benchmarks.json)
- [`tp4-flydslar/agent-multiturn-benchmarks.json`](tp4-flydslar/agent-multiturn-benchmarks.json)
