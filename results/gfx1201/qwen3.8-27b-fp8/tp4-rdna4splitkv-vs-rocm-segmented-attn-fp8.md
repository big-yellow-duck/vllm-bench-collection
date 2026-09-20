# TP4 RDNA4 SplitKV vs ROCm segmented attention

Run date: 2026-09-19. Lower TTFT and E2E latency are better; higher output TPS is better. Values below are arithmetic means over successful request records.

## Setup and compatibility

- Hardware: AMD Radeon AI PRO R9700 (`gfx1201`), TP4.
- Model: `Qwen/Qwen3.8-27B-FP8`; KV cache: FP8.
- Candidate: `ROCM_SEGMENTED_ATTN` on vLLM branch `perf/rdna4-prefill-autotune-splitkv`, source HEAD `25a3bb1b4264cdaadc0068349045385a71697a64` plus uncommitted milestone changes; prefix caching and startup autotuning enabled; `--max-num-seqs 128`.
- Workload: streams 1/2/4/8; 8 turns; 2048 new prompt tokens per turn; 512 requested output tokens; seed 20260715; one warmup conversation per stream.
- Both reports use GuideLLM 0.7.3, matching model/workload/profile settings, and have 8/16/32/64 successful requests with zero incomplete or errored requests.

Raw reports: [RDNA4 SplitKV](tp4-rdna4splitkv_fp8/agent-multiturn-benchmarks.json), [segmented](tp4-rocm-segmented-attn-fp8/agent-multiturn-benchmarks.json).

## Successful-request means

| Streams | Output tok/s, SplitKV → segmented | Output tok/s change | Mean TTFT (ms), SplitKV → segmented | Mean TTFT change | Mean E2E (s), SplitKV → segmented | Mean E2E change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.46 → 29.29 | −25.8% | 1,853 → 897 | **−51.6%** | 13.03 → 17.49 | +34.2% |
| 2 | 34.71 → 40.70 | **+17.3%** | 2,524 → 1,047 | **−58.5%** | 14.87 → 12.59 | **−15.3%** |
| 4 | 28.68 → 35.73 | **+24.6%** | 3,989 → 1,117 | **−72.0%** | 18.18 → 14.36 | **−21.0%** |
| 8 | 26.60 → 34.36 | **+29.2%** | 4,566 → 1,036 | **−77.3%** | 20.25 → 15.07 | **−25.6%** |

This is a mixed but favorable result. Segmented attention wins mean TTFT at every stream level and wins all three metrics at streams 2, 4, and 8. Prior SplitKV remains better for single-stream decode: it leads TPS on all eight turns and E2E on all eight turns. Across all stream/turn points, segmented wins 23/32 TPS, 30/32 TTFT, and 23/32 E2E points. The other TPS/E2E loss is the first turn at two streams.

## Charts

- [Output TPS by turn](../../../plots/gfx1201/qwen3.8-27b-fp8/tp4-rdna4splitkv-vs-rocm-segmented-attn-fp8/compare_tps_by_turn.png)
- [TTFT by turn](../../../plots/gfx1201/qwen3.8-27b-fp8/tp4-rdna4splitkv-vs-rocm-segmented-attn-fp8/compare_ttft_by_turn.png)
- [E2E latency by turn](../../../plots/gfx1201/qwen3.8-27b-fp8/tp4-rdna4splitkv-vs-rocm-segmented-attn-fp8/compare_e2e_by_turn.png)

## Caveats

This is end-to-end serving performance, not an isolated attention-kernel comparison, and historical launch/source metadata is not embedded in the report. Candidate first-use JIT warnings affected segmented stage/reduction and unrelated GDN/sampling kernels. The single-stream result indicates that the prior SplitKV route should remain a useful decode option even though segmented attention dominates TTFT and concurrent-agent cases.
