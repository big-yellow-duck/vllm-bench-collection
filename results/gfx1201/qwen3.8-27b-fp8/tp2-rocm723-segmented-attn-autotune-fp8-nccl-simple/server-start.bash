NCCL_PROTO=Simple \
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 \
vllm serve \
Qwen/Qwen3.8-27B-FP8 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.92 \
    --tensor-parallel-size 2 \
    --max-num-batched-tokens 8192 \
    --max-model-len auto \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --mm-encoder-tp-mode data \
    --kv-cache-dtype fp8 \
    --load-format fastsafetensors \
    --attention-backend ROCM_SEGMENTED_ATTN \
    --kernel-config '{"enable_rocm_segmented_attn_autotune": true}' \
    2>&1 | tee qwen3.8-27b-fp8-segmented-attn.log
