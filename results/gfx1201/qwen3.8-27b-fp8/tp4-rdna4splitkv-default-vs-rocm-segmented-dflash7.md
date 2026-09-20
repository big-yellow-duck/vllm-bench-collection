# TP4 ROCm segmented attention with DFlash2-7

## Outcome

The opt-in `ROCM_SEGMENTED_ATTN` backend with DFlash2 (`num_speculative_tokens=7`) completed the full `simple-agent-multiturn-prefix-cache` sweep without errors. Against the stored TP4 RDNA4 SplitKV default-decode result, measured-window aggregate output throughput improves by 4.1-21.6%, mean per-request output rate improves by 14.7-29.9%, mean TTFT drops by 35.4-73.0%, and mean end-to-end latency drops by 9.6-23.9% across all configured stream counts.

This is a combined backend-plus-speculation comparison. The older segmented-attention default-decode result is included as a secondary reference, but it was collected from commit `dc919e3f5`, while the DFlash run used commit `7ee10d843e`; it is therefore directional rather than a strict same-binary speculative-decoding A/B.

## Setup

- Model: `Qwen/Qwen3.8-27B-FP8`
- Hardware: 4 x AMD Radeon R9700 (`gfx1201`), tensor parallel size 4
- KV cache: FP8, LBHNC layout, prefix caching enabled
- Attention backend: `ROCM_SEGMENTED_ATTN`
- Speculator: `incoai/Qwen3.8-27B-DFlash2`, 7 speculative tokens
- Workload: 8 turns, 2,048 new prompt tokens per turn, 512 output tokens per turn
- Streams: 1, 2, 4, 8; max requests: 8, 16, 32, 64; seed: 20260715
- All three reports use the same workload, stream levels, request limits, prompt/output lengths, turn count, and seed.
- All reports contain 8/16/32/64 successful requests and zero errored or incomplete requests.

The cold DFlash startup took 422.17 seconds. Segmented-attention autotuning covered 118/118 planned workloads across four TP ranks in 144.39 seconds with zero failures. During the benchmark, periodic server telemetry summed to 36,949 accepted draft tokens out of 172,011 drafted tokens (21.48% acceptance). The server warned that speculative settings capped `max_num_scheduled_tokens` at 2,048, so this result retains the default setting but may not be the best attainable DFlash throughput.

## Primary comparison: RDNA4 SplitKV default decode to segmented DFlash2-7

GuideLLM's measured-window aggregate output throughput, after excluding warmup and cooldown requests:

| Streams | Aggregate output tok/s, baseline to candidate | Change |
|---:|---:|---:|
| 1 | 38.99 to 40.57 | **+4.1%** |
| 2 | 68.09 to 77.58 | **+13.9%** |
| 4 | 106.95 to 120.64 | **+12.8%** |
| 8 | 107.87 to 131.15 | **+21.6%** |

Mean per-request statistics from the successful request samples:

| Streams | Output tok/s, baseline to candidate | Change | Mean TTFT (ms), baseline to candidate | Change | Mean E2E (s), baseline to candidate | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.46 to 45.27 | **+14.7%** | 1,853 to 1,197 | **-35.4%** | 13.03 to 11.77 | **-9.6%** |
| 2 | 34.71 to 41.30 | **+19.0%** | 2,524 to 1,067 | **-57.7%** | 14.87 to 12.85 | **-13.6%** |
| 4 | 28.68 to 34.36 | **+19.8%** | 3,989 to 1,279 | **-67.9%** | 18.18 to 15.60 | **-14.2%** |
| 8 | 26.60 to 34.55 | **+29.9%** | 4,566 to 1,234 | **-73.0%** | 20.25 to 15.41 | **-23.9%** |

## Secondary reference: older segmented default decode to segmented DFlash2-7

| Streams | Output tok/s, baseline to candidate | Change | Mean TTFT (ms), baseline to candidate | Change | Mean E2E (s), baseline to candidate | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29.29 to 45.27 | **+54.5%** | 897 to 1,197 | +33.5% | 17.49 to 11.77 | **-32.7%** |
| 2 | 40.70 to 41.30 | **+1.5%** | 1,047 to 1,067 | +1.9% | 12.59 to 12.85 | +2.1% |
| 4 | 35.73 to 34.36 | -3.8% | 1,117 to 1,279 | +14.5% | 14.36 to 15.60 | +8.6% |
| 8 | 34.36 to 34.55 | **+0.6%** | 1,036 to 1,234 | +19.1% | 15.07 to 15.41 | +2.3% |

The by-turn plots show why the aggregate result is mixed against the older segmented default: DFlash is strongest during the early, shorter-prefix turns, while its advantage shrinks or reverses as the conversation reaches roughly 12K-20K prompt tokens. With only about 21.5% draft-token acceptance, draft and verification overhead dominate more often at higher concurrency and longer prefixes. A strict follow-up should rerun default decode from commit `7ee10d843e` with all other server settings identical.

## Artifacts

- [DFlash GuideLLM report](tp4-rocm-segmented-attn-fp8-dflash7/agent-multiturn-benchmarks.json)
- [DFlash benchmark log](tp4-rocm-segmented-attn-fp8-dflash7/benchmark.log)
- [DFlash server log](tp4-rocm-segmented-attn-fp8-dflash7/server.log)
- [Output TPS by turn](../../../plots/gfx1201/qwen3.8-27b-fp8/tp4-rdna4splitkv-default-vs-rocm-segmented-dflash7/compare_tps_by_turn.png)
- [TTFT by turn](../../../plots/gfx1201/qwen3.8-27b-fp8/tp4-rdna4splitkv-default-vs-rocm-segmented-dflash7/compare_ttft_by_turn.png)
- [End-to-end latency by turn](../../../plots/gfx1201/qwen3.8-27b-fp8/tp4-rdna4splitkv-default-vs-rocm-segmented-dflash7/compare_e2e_by_turn.png)
