if [ -z "$1" ]; then
	echo "Usage: $0 <log_output_file>" >&2
	exit 1
fi
log_output_file="$1"

vllm serve Qwen/Qwen3.5-9B --dtype bfloat16 --max-model-len 4096 --enable-logging-iteration-details 2>&1 \
  | grep --line-buffered "Iteration(" > "$log_output_file"
