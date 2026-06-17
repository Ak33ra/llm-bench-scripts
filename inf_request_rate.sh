#!/bin/bash 
set -euo pipefail

# Infinite request rate spawns all jobs at time 0, so burstiness parameter 
# has no effect. We set burstiness = 0.0 since CV = 1/sqrt(burstiness), 
# to signal "infinite" variance.

model="Qwen/Qwen3.5-9B"
model_name="qwen"
#model="zai-org/GLM-4.7-Flash"
#model_name="glm-4.7-flash"
engine="vllm"
gpu="gh200"

request_rate="inf"
#num_prompts=1000
input_len=128
output_len=128
result_root="./${engine}/${gpu}/${model_name}/in${input_len}out${output_len}/rate${request_rate}"
burstiness=0.0
seed=0

for num_prompts in 1000 500 250 100 ; do

out_dir="${result_root}/burst${burstiness}"
mkdir -p "$out_dir"

vllm bench serve \
  --model $model \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len $input_len \
  --random-output-len $output_len \
  --ignore-eos \
  --num-prompts $num_prompts \
  --request-rate $request_rate \
  --save-result \
  --save-detailed \
  --metric-percentiles "50,90,95,99" \
  --percentile-metrics "ttft,tpot,itl,e2el" \
  --result-dir "$out_dir" \
  --result-filename "seed${num_prompts}.json"

done
