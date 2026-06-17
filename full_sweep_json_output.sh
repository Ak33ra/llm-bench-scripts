#!/bin/bash 
set -euo pipefail

model="Qwen/Qwen3.5-9B"
model_name="qwen"
#model="zai-org/GLM-4.7-Flash"
#model_name="glm-4.7-flash"
engine="vllm"
gpu="gh200"

# Capture a standard dataset over more input/output lengths
# This is not the extreme variance study (inf vs uniform)

#request_rate=80
num_prompts=1000
#input_len=128
#output_len=128

for input_len in 128 256 1024; do 
for output_len in 1 128 256 1024; do
for request_rate in 10 20 40; do
for burstiness in 1.0 0.1 0.01; do
for seed in 6 7 8 9 10; do

  result_root="./${engine}/${gpu}/${model_name}/in${input_len}out${output_len}/rate${request_rate}"

  out_dir="${result_root}/burst${burstiness}"
  mkdir -p "$out_dir"

  result_file="${out_dir}/seed${seed}.json"
  if [[ -f "$result_file" ]]; then
    echo "Skipping: $result_file already exists"
    continue
  fi

  vllm bench serve \
    --model Qwen/Qwen3.5-9B \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
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
    #--plot-timeline \
done
done
done
done
done
