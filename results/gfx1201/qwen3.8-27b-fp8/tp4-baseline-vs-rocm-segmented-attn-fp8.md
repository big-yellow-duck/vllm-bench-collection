# TP4 baseline vs ROCm segmented attention

Run date: 2026-09-19. Lower TTFT and E2E latency are better; higher output TPS is better. Values below are arithmetic means over successful request records.

## Setup and compatibility

- Hardware: AMD Radeon AI PRO R9700 (`gfx1201`), TP4.
- Model: `Qwen/Qwen3.8-27B-FP8`; KV cache: FP8.
- Candidate: `ROCM_SEGMENTED_ATTN` on vLLM branch `perf/rdna4-prefill-autotune-splitkv`, source HEAD `25a3bb1b4264cdaadc0068349045385a71697a64` plus uncommitted milestone changes; prefix caching and startup autotuning enabled; `--max-num-seqs 128`.
- Workload: streams 1/2/4/8; 8 turns; 2048 new prompt tokens per turn; 512 requested output tokens; seed 20260715.
- Both reports use GuideLLM 0.7.3 and contain 8/16/32/64 successful requests with zero incomplete or errored requests.
- Warmup differs: the candidate used one warmup conversation per stream, while the stored TP4 baseline report records no explicit warmup. Treat early-turn TTFT as less strictly controlled than the later prefix-cache steps.

Raw reports: [baseline](tp4-baseline/agent-multiturn-benchmarks.json), [segmented](tp4-rocm-segmented-attn-fp8/agent-multiturn-benchmarks.json).

## Successful-request means

| Streams | Output tok/s, baseline → candidate | Output tok/s change | Mean TTFT (ms), baseline → candidate | Mean TTFT change | Mean E2E (s), baseline → candidate | Mean E2E change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.03 → 29.29 | **+192.1%** | 2,340 → 897 | **−61.7%** | 65.14 → 17.49 | **−73.2%** |
| 2 | 9.53 → 40.70 | **+327.0%** | 3,066 → 1,047 | **−65.8%** | 67.86 → 12.59 | **−81.4%** |
| 4 | 9.26 → 35.73 | **+285.7%** | 3,391 → 1,117 | **−67.1%** | 69.17 → 14.36 | **−79.2%** |
| 8 | 8.32 → 34.36 | **+312.9%** | 5,226 → 1,036 | **−80.2%** | 75.93 → 15.07 | **−80.2%** |

The candidate wins all 32 TPS and E2E turn-level points. It wins 29/32 TTFT points; the three losses are first-turn requests at streams 1, 2, and 4. Every subsequent turn favors segmented attention.

## Charts

- [Output TPS by turn](../../../plots/gfx1201/qwen3.8-27b-fp8/tp4-baseline-vs-rocm-segmented-attn-fp8/compare_tps_by_turn.png)
- [TTFT by turn](../../../plots/gfx1201/qwen3.8-27b-fp8/tp4-baseline-vs-rocm-segmented-attn-fp8/compare_ttft_by_turn.png)
- [E2E latency by turn](../../../plots/gfx1201/qwen3.8-27b-fp8/tp4-baseline-vs-rocm-segmented-attn-fp8/compare_e2e_by_turn.png)

## Caveats

The warmup mismatch and absent historical launch/source metadata limit strict first-turn attribution. This is an end-to-end serving comparison, not isolated attention-kernel timing. The candidate emitted one-time JIT warnings for segmented stage/reduction, GDN, and sampling kernels. The candidate used `--max-num-seqs 128`; the workload peaks at eight streams.
