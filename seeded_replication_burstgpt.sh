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
result_root="./${engine}/${gpu}/${model_name}/burstgpt/rate${request_rate}"

for burstiness in 1.0 0.5 0.1; do
for seed in 0 1 2; do

  out_dir="${result_root}/burst${burstiness}"
  mkdir -p "$out_dir"

  vllm bench serve \
    --model $model \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --dataset-name burstgpt \
    --dataset-path ~/BurstGPT_without_fails_2.csv \
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
