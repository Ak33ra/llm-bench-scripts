VLLM_USE_V1=0 vllm serve Qwen/Qwen3.5-9B --dtype bfloat16 --max-model-len 4096 --no-enable-chunked-prefill
