model_name="Qwen/Qwen3.5-9B"

python3 examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py  \
     --model $model_name  \
     --prefill localhost:8100   \
     --decode localhost:8200    \
     --port 8000
