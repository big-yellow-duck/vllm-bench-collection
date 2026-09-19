# TP4 Baseline vs. Flydslar Rerun

This report compares `tp4-baseline` with the new TP4 Flydslar rerun stored at
the repository root as `agent-multiturn-benchmarks.json`, for
`Qwen/Qwen3.8-27B-FP8` on `gfx1201`.

## Benchmark setup

Both runs use the same client-side benchmark configuration:

- Tensor parallelism: 4
- Workload: `simple-agent-multiturn-prefix-cache`
- Streams: 1, 2, 4, and 8
- Synthetic prompt: 2,048 tokens per turn
- Output: 512 tokens per turn
- Turns per conversation: 8
- Seed: `20260715`

Output throughput is the benchmark's mean `output_tokens_per_second`.
Aggregate TTFT uses mean `time_to_first_token_ms`; E2E latency uses mean
`request_latency`. Lower latency is better. Achieved concurrency is the
benchmark's observed mean `request_concurrency` and is included because the
server does not reach every requested stream level.

## Aggregate results

| Streams | Baseline output tok/s | Flydslar output tok/s | Throughput change | Mean TTFT (ms), baseline → flydslar | TTFT change | Mean E2E latency (s), baseline → flydslar | E2E change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.87 | 7.01 | −10.9% | 2,340 → 2,063 | **−11.8%** | 65.14 → 73.40 | +12.7% |
| 2 | 15.09 | 14.64 | −3.0% | 3,066 → 2,826 | **−7.8%** | 67.86 → 70.12 | +3.3% |
| 4 | 15.10 | 15.05 | −0.3% | 3,391 → 3,060 | **−9.8%** | 69.17 → 67.78 | **−2.0%** |
| 8 | 27.96 | 31.94 | **+14.2% (1.14×)** | 5,226 → 4,617 | **−11.7%** | 75.93 → 72.85 | **−4.0%** |

## Achieved concurrency validation

| Target streams | Baseline achieved concurrency | Flydslar rerun achieved concurrency |
|---:|---:|---:|
| 1 | 1.00 | 1.00 |
| 2 | 2.00 | 2.00 |
| 4 | 2.04 | 2.01 |
| 8 | 4.14 | 4.53 |

```text
Achieved concurrency (target shown at left; each █ is approximately one request)

1 target   baseline █ 1.00       rerun █ 1.00
2 target   baseline ██ 2.00      rerun ██ 2.00
4 target   baseline ██ 2.04      rerun ██ 2.01
8 target   baseline ████ 4.14    rerun █████ 4.53
```

## Input-length-normalized p99 TTFT

Each turn appends another prompt and response to the prefix, so prompt length
increases from about 2.1k to 20.1k tokens. Rows are grouped by `turn_index`,
which holds the prefix length constant. Negative values improve tail TTFT;
positive values regress it.

Each bucket has only one to four requests per run, so its p99 is effectively
the slowest request and is a tail observation rather than a stable percentile.

### 1 stream

| Prompt tokens (approx.) | Baseline p99 TTFT (ms) | Rerun p99 TTFT (ms) | Change |
|---:|---:|---:|---:|
| 2.1k | 388 | 541 | +39.5% |
| 4.7k | 2,288 | 1,200 | **−47.6%** |
| 7.2k | 2,556 | 1,600 | **−37.4%** |
| 9.8k | 2,477 | 1,714 | **−30.8%** |
| 12.4k | 2,441 | 1,843 | **−24.5%** |
| 15.0k | 2,977 | 2,366 | **−20.5%** |
| 17.5k | 2,878 | 2,495 | **−13.3%** |
| 20.1k | 2,716 | 2,360 | **−13.1%** |

### 2 streams

| Prompt tokens (approx.) | Baseline p99 TTFT (ms) | Rerun p99 TTFT (ms) | Change |
|---:|---:|---:|---:|
| 2.1k | 617 | 1,009 | +63.6% |
| 4.7k | 2,928 | 2,007 | **−31.4%** |
| 7.2k | 3,867 | 2,924 | **−24.4%** |
| 9.8k | 3,786 | 3,077 | **−18.7%** |
| 12.4k | 3,686 | 3,061 | **−17.0%** |
| 15.0k | 4,872 | 4,398 | **−9.7%** |
| 17.5k | 4,654 | 4,428 | **−4.9%** |
| 20.1k | 4,330 | 4,021 | **−7.1%** |

### 4 streams

| Prompt tokens (approx.) | Baseline p99 TTFT (ms) | Rerun p99 TTFT (ms) | Change |
|---:|---:|---:|---:|
| 2.1k | 1,488 | 1,313 | **−11.8%** |
| 4.7k | 3,803 | 2,731 | **−28.2%** |
| 7.2k | 5,165 | 4,173 | **−19.2%** |
| 9.8k | 4,999 | 4,415 | **−11.7%** |
| 12.4k | 4,731 | 4,175 | **−11.8%** |
| 15.0k | 6,656 | 6,426 | **−3.5%** |
| 17.5k | 6,256 | 5,959 | **−4.7%** |
| 20.1k | 5,732 | 5,976 | +4.3% |

### 8 streams

| Prompt tokens (approx.) | Baseline p99 TTFT (ms) | Rerun p99 TTFT (ms) | Change |
|---:|---:|---:|---:|
| 2.1k | 1,942 | 3,049 | +57.0% |
| 4.7k | 7,013 | 5,438 | **−22.5%** |
| 7.2k | 9,612 | 9,171 | **−4.6%** |
| 9.8k | 8,975 | 6,156 | **−31.4%** |
| 12.4k | 8,751 | 6,679 | **−23.7%** |
| 15.0k | 13,118 | 11,514 | **−12.2%** |
| 17.5k | 10,650 | 7,323 | **−31.2%** |
| 20.1k | 10,569 | 7,434 | **−29.7%** |

## Finding

- At one and two streams, achieved concurrency matches exactly. The rerun reduces mean and grouped p99 TTFT after the initial prompt, but output throughput is 3.0–10.9% lower and E2E latency is higher.
- At the four-stream target, both runs achieve only about two concurrent requests. This is an apples-to-apples concurrency comparison: output throughput is unchanged (−0.3%), while the rerun improves mean TTFT by 9.8% and E2E latency by 2.0%.
- At the eight-stream target, the rerun has 9.2% more achieved concurrency (4.53 vs. 4.14). Its +14.2% output-throughput result cannot be treated as a kernel-only gain; it coincides with the higher active-request level. It does improve mean TTFT, E2E latency, and most prefix-length p99 TTFT buckets.

The rerun confirms that TP4 currently plateaus near two active requests at the
four-stream target and near four to five at the eight-stream target. It does
not confirm a TP4 throughput scaling gain over baseline at matched concurrency.

## Source data

- [`tp4-baseline/agent-multiturn-benchmarks.json`](tp4-baseline/agent-multiturn-benchmarks.json)
- [`agent-multiturn-benchmarks.json`](../../../agent-multiturn-benchmarks.json) — TP4 Flydslar rerun
