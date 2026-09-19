# TP4 Flydslar Rerun vs. Previous Run

This report compares the rerun stored at the repository root as
`agent-multiturn-benchmarks.json` with the previous
`tp4-flydslar/agent-multiturn-benchmarks.json` run.

## Configuration and run health

The client-side benchmark configuration is identical in both runs:

- Tensor parallelism: 4
- Workload: `simple-agent-multiturn-prefix-cache`
- Streams: 1, 2, 4, and 8
- Prompt/output/turns: 2,048 / 512 / 8
- Seed: `20260715`
- GuideLLM version: `0.7.3`

Neither run reports request errors. The discrepancy comes from achieved request
concurrency, which is measured by the benchmark rather than requested by the
client.

## Results

| Target streams | Achieved concurrency, previous → rerun | Output tok/s, previous → rerun | Throughput change | Mean TTFT (ms), previous → rerun | TTFT change | Mean E2E latency (s), previous → rerun | E2E change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 → 1.00 | 7.03 → 7.01 | −0.3% | 1,958 → 2,063 | +5.3% | 73.12 → 73.40 | +0.4% |
| 2 | 2.00 → 2.00 | 14.63 → 14.64 | +0.0% | 2,829 → 2,826 | −0.1% | 70.13 → 70.12 | −0.0% |
| 4 | **4.00 → 2.01** | 28.37 → 15.05 | −47.0% | 4,150 → 3,060 | **−26.3%** | 70.78 → 67.78 | **−4.2%** |
| 8 | 4.07 → 4.53 | 29.23 → 31.94 | +9.3% | 4,123 → 4,617 | +12.0% | 71.32 → 72.85 | +2.2% |

```text
Achieved concurrency (target shown at left; each █ is approximately one request)

1 target   previous █ 1.00       rerun █ 1.00
2 target   previous ██ 2.00      rerun ██ 2.00
4 target   previous ████ 4.00    rerun ██ 2.01
8 target   previous ████ 4.07    rerun █████ 4.53
```

## Finding

The previous run is not botched in the sense of a failed benchmark: it has the
same configuration and no request errors. Its four-stream result actually
reached approximately four concurrent requests.

The rerun only reaches approximately two concurrent requests at the four-stream
target, which explains its 47.0% lower aggregate throughput and its lower TTFT.
Likewise, neither eight-stream run reaches the requested eight concurrent
requests; both plateau around four to five.

This also invalidates the earlier TP4 baseline-versus-Flydslar four-stream
throughput claim as a kernel-only comparison. The baseline reached only 2.04
actual concurrent requests, while the previous Flydslar run reached 4.00. The
apparent 87.9% Flydslar gain is therefore primarily an achieved-concurrency
difference. At one and two streams, the rerun and previous results are nearly
identical, which supports that conclusion.

The likely source is a server-side scheduling or launch-capacity setting that
changed between runs. Before making TP4 scaling claims, verify the server's
effective maximum active sequence count and report achieved concurrency beside
each target stream level.

## Source data

- [`agent-multiturn-benchmarks.json`](../../../agent-multiturn-benchmarks.json) — TP4 Flydslar rerun
- [`tp4-flydslar/agent-multiturn-benchmarks.json`](tp4-flydslar/agent-multiturn-benchmarks.json) — previous TP4 Flydslar run
- [`tp4-baseline/agent-multiturn-benchmarks.json`](tp4-baseline/agent-multiturn-benchmarks.json) — prior TP4 baseline run
