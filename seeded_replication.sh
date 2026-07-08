#!/bin/bash 
set -euo pipefail

#model="Qwen/Qwen3.5-9B"
#model_name="qwen"
#model="zai-org/GLM-4.7-Flash"
#model_name="glm-4.7-flash"
model="meta-llama/Llama-3.1-8B"
model_name="llama-3.1-8b"
#model="Qwen/Qwen3-32B"
#model_name="qwen-3-32b"
engine="vllm-crash-test"
gpu="h200"

request_rate=120
num_prompts=5000
input_len=1024
output_len=2
result_root="./${engine}/${gpu}/${model_name}/in${input_len}out${output_len}/rate${request_rate}"

for burstiness in 1.0 0.1 0.01; do
for seed in 0 1 2 3 4 5 ; do

  out_dir="${result_root}/burst${burstiness}"
  mkdir -p "$out_dir"

  vllm bench serve \
    --model $model \
    --backend openai \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len $input_len \
    --random-output-len $output_len \
    --ignore-eos \
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
