# TP2 baseline vs ROCm segmented attention

Run date: 2026-09-19. Lower TTFT and E2E latency are better; higher output TPS is better. Values below are arithmetic means over successful request records, not GuideLLM's aggregate `benchmark.metrics` fields.

## Setup and compatibility

- Hardware: AMD Radeon AI PRO R9700 (`gfx1201`), TP2.
- Model: `Qwen/Qwen3.8-27B-FP8`; KV cache: FP8.
- Candidate: vLLM branch `perf/rdna4-prefill-autotune-splitkv`, source HEAD `25a3bb1b4264cdaadc0068349045385a71697a64` plus the uncommitted segmented-attention milestone changes.
- Candidate backend: `ROCM_SEGMENTED_ATTN`, prefix caching enabled, startup autotuning enabled, `--max-num-seqs 128`.
- Workload: `simple-agent-multiturn-prefix-cache.json`; streams 1/2/4/8; 8 turns; 2048 new prompt tokens per turn; 512 requested output tokens; seed 20260715; one warmup conversation per stream.
- Both reports use GuideLLM 0.7.3 and the same model/workload/profile. Both contain 8/16/32/64 successful requests and zero incomplete or errored requests.

Raw reports: [baseline](tp2-baseline/agent-multiturn-benchmarks.json), [segmented](tp2-rocm-segmented-attn-fp8/agent-multiturn-benchmarks.json).

## Successful-request means

| Streams | Output tok/s, baseline → candidate | Output tok/s change | Mean TTFT (ms), baseline → candidate | Mean TTFT change | Mean E2E (s), baseline → candidate | Mean E2E change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.37 → 22.55 | **+205.8%** | 2,460 → 1,196 | **−51.4%** | 82.57 → 22.71 | **−72.5%** |
| 2 | 7.07 → 22.80 | **+222.7%** | 3,863 → 958 | **−75.2%** | 86.23 → 22.46 | **−74.0%** |
| 4 | 6.92 → 26.59 | **+284.6%** | 4,224 → 1,304 | **−69.1%** | 87.81 → 19.47 | **−77.8%** |
| 8 | 6.30 → 22.71 | **+260.4%** | 5,060 → 1,397 | **−72.4%** | 95.79 → 22.80 | **−76.2%** |

The candidate wins all 32 per-turn TPS and E2E points. It wins 31/32 TTFT points; the exception is the first request at one stream, where TTFT is 75.3% higher. Subsequent turns, whose mean prompt length rises from about 4.7K to 20.1K tokens, consistently favor segmented attention.

## Charts

- [Output TPS by turn](../../../plots/gfx1201/qwen3.8-27b-fp8/tp2-baseline-vs-rocm-segmented-attn-fp8/compare_tps_by_turn.png)
- [TTFT by turn](../../../plots/gfx1201/qwen3.8-27b-fp8/tp2-baseline-vs-rocm-segmented-attn-fp8/compare_ttft_by_turn.png)
- [E2E latency by turn](../../../plots/gfx1201/qwen3.8-27b-fp8/tp2-baseline-vs-rocm-segmented-attn-fp8/compare_e2e_by_turn.png)

## Caveats

This is an end-to-end serving comparison, so it includes model, sampling, communication, graph, and attention effects. The candidate server emitted one-time JIT warnings for segmented stage/reduction and sampling kernels during early requests. The historical server launch command and source revision were not embedded in its report. The candidate used `--max-num-seqs 128`; the workload itself peaks at eight streams.
