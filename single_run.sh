#!/bin/bash
set -euo pipefail

model="Qwen/Qwen3.5-9B"

request_rate=40
num_prompts=1000
input_len=512
output_len=256
burstiness=0.1

vllm bench serve \
  --model Qwen/Qwen3-8B \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len $input_len \
  --random-output-len $output_len \
  --ignore-eos \
  --num-prompts $num_prompts \
  --request-rate $request_rate \
  --burstiness $burstiness \
  --seed 0 \
