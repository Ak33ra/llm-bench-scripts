#!/bin/bash 
set -euo pipefail

model="Qwen/Qwen3.5-9B"
model_name="qwen"
#model="zai-org/GLM-4.7-Flash"
#model_name="glm-4.7-flash"
engine="vllm"
gpu="gh200"

request_rate=20
num_prompts=1000
result_root="./${engine}/${gpu}/${model_name}/sharegpt/rate${request_rate}"

for burstiness in 0.01; do
for seed in 0; do

  out_dir="${result_root}/burst${burstiness}"
  mkdir -p "$out_dir"

  vllm bench serve \
    --model Qwen/Qwen3.5-9B \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --dataset-name sharegpt \
    --dataset-path ~/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts $num_prompts \
    --request-rate $request_rate \
    --burstiness $burstiness \
    --seed $seed \
    --save-result \
    --save-detailed \
    --metric-percentiles "50,90,95,99" \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --result-dir "$out_dir" \
    --result-filename "seed${seed}.json"
done
done
