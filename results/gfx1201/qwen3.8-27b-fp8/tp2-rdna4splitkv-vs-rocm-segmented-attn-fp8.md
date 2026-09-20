# TP2 RDNA4 SplitKV vs ROCm segmented attention

Run date: 2026-09-19. Lower TTFT and E2E latency are better; higher output TPS is better. Values below are arithmetic means over successful request records.

## Setup and compatibility

- Hardware: AMD Radeon AI PRO R9700 (`gfx1201`), TP2.
- Model: `Qwen/Qwen3.8-27B-FP8`; KV cache: FP8.
- Candidate: `ROCM_SEGMENTED_ATTN` on vLLM branch `perf/rdna4-prefill-autotune-splitkv`, source HEAD `25a3bb1b4264cdaadc0068349045385a71697a64` plus uncommitted milestone changes; prefix caching and startup autotuning enabled; `--max-num-seqs 128`.
- Workload: streams 1/2/4/8; 8 turns; 2048 new prompt tokens per turn; 512 requested output tokens; seed 20260715; one warmup conversation per stream.
- Both reports use GuideLLM 0.7.3 and matching model/workload/profile settings.
- Candidate counts are 8/16/32/64 successful, zero errors. Historical SplitKV counts are 8/16/32/63 successful, with one error at eight streams; means use only successful records.

Raw reports: [RDNA4 SplitKV](tp2-rdna4-fp8vblockscalemm-splitkvfp8/agent-multiturn-benchmarks.json), [segmented](tp2-rocm-segmented-attn-fp8/agent-multiturn-benchmarks.json).

## Successful-request means

| Streams | Output tok/s, SplitKV → segmented | Output tok/s change | Mean TTFT (ms), SplitKV → segmented | Mean TTFT change | Mean E2E (s), SplitKV → segmented | Mean E2E change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.91 → 22.55 | **+13.2%** | 2,825 → 1,196 | **−57.7%** | 25.91 → 22.71 | **−12.3%** |
| 2 | 17.36 → 22.80 | **+31.3%** | 4,476 → 958 | **−78.6%** | 29.88 → 22.46 | **−24.9%** |
| 4 | 16.05 → 26.59 | **+65.7%** | 4,945 → 1,304 | **−73.6%** | 33.02 → 19.47 | **−41.0%** |
| 8 | 12.88 → 22.71 | **+76.4%** | 7,413 → 1,397 | **−81.1%** | 43.01 → 22.80 | **−47.0%** |

Segmented attention wins 31/32 per-turn points for each of TPS, TTFT, and E2E. The sole loss is the first request at one stream: TPS is 3.6% lower, TTFT is 334.4% higher, and E2E is 3.7% higher. It wins every later turn and every point at streams 2/4/8.

## Charts

- [Output TPS by turn](../../../plots/gfx1201/qwen3.8-27b-fp8/tp2-rdna4splitkv-vs-rocm-segmented-attn-fp8/compare_tps_by_turn.png)
- [TTFT by turn](../../../plots/gfx1201/qwen3.8-27b-fp8/tp2-rdna4splitkv-vs-rocm-segmented-attn-fp8/compare_ttft_by_turn.png)
- [E2E latency by turn](../../../plots/gfx1201/qwen3.8-27b-fp8/tp2-rdna4splitkv-vs-rocm-segmented-attn-fp8/compare_e2e_by_turn.png)

## Caveats

The historical TP2 directory is tagged `fp8vblockscalemm-splitkvfp8`, so it represents the combined prior RDNA4 work rather than an isolated SplitKV kernel change. Its eight-stream report has one errored request. The benchmark is end-to-end serving, and historical launch/source details are not embedded in the JSON. The candidate emitted first-use JIT warnings for segmented stage/reduction and unrelated sampling kernels.
