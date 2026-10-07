# ROCm segmented attention milestone: agent multi-turn prefix cache

This milestone compares the new `ROCM_SEGMENTED_ATTN` backend against the stored vanilla baselines and prior RDNA4 SplitKV work using `simple-agent-multiturn-prefix-cache.json`, Qwen3.8-27B-FP8, FP8 KV cache, and TP4. The superseded TP2 results have been removed; the current TP2 comparison uses ROCm 7.2.3, segmented attention with autotuning, and AITER unified attention.

## Result

- Against stored baseline, segmented attention wins mean TPS, TTFT, and E2E at every tested stream level on TP4.
- Against prior SplitKV on TP4, segmented wins TTFT at all streams and wins TPS/E2E at streams 2/4/8. The exception is single-stream decode, where it is 25.8% lower in TPS and 34.2% higher in E2E despite 51.6% lower TTFT.
- The important workload behavior is visible by turn: segmented performance stays relatively flat as the cached conversation prefix grows from roughly 2.1K to 20.1K prompt tokens, while the older paths degrade much more sharply.

## Detailed comparisons

- [Current TP2 ROCm 7.2.3 segmented autotune vs AITER unified attention](../../../comparisons/gfx1201/qwen3.8-27b-fp8/rocm723-tp2-segmented-autotune-vs-aiter-unified/comparison.md)
- [TP4 baseline vs segmented](tp4-baseline-vs-rocm-segmented-attn-fp8.md)
- [TP4 RDNA4 SplitKV vs segmented](tp4-rdna4splitkv-vs-rocm-segmented-attn-fp8.md)

## Startup tuning observation

The cold TP4 startup tuning search completed in 151.2 seconds across four ranks, producing 138 merged tuned shapes with no failures. Work was distributed 35/35/36/32 shapes across ranks. The new Q=4, Q=16, and Q=64 query capacities were present in the TP4 search.

## Known limitations

- The stored TP4 baseline lacks the explicit one-conversation warmup used by the other reports.
- Historical reports do not embed full vLLM launch commands or source revisions.
- Candidate runs emitted one-time inference JIT warnings, so first-turn data includes some cold-specialization noise.
