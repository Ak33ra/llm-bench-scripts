#!/usr/bin/env bash
# Local copy script for run_lambda_worker.py (--copy-script).
# Runs on YOUR machine after a clean worker run. The orchestrator exports the
# instance details as env vars; we use them instead of hardcoding host/IP:
#   LAMBDA_INSTANCE_IP, LAMBDA_SSH_USER, LAMBDA_SSH_KEY_PATH, LAMBDA_SSH_OPTS,
#   LAMBDA_REMOTE_WORKDIR, LAMBDA_REMOTE_RESULTS_DIR, LAMBDA_LOCAL_RESULTS_DIR
set -euo pipefail

remote="${LAMBDA_SSH_USER}@${LAMBDA_INSTANCE_IP}"

# run_iterlog_sweep.sh writes results under a RELATIVE path from where it runs
# (~/llm-bench-scripts), so the benchmark outputs live in
#   ~/llm-bench-scripts/<engine>/<gpu>/<model>/in<..>out<..>/rate<..>/...
# The top of that tree is the sweep's $engine dir (currently vllm-torch-profiler).
# Pull that whole tree into ./vllm-torch-profiler in the launch cwd. (NOT ~/vllm,
# which is just the vLLM source checkout — no results there.)
results_subdir="vllm-torch-profiler"

rsync -azP -e "$LAMBDA_SSH_OPTS" \
  "${remote}:${LAMBDA_REMOTE_WORKDIR}/llm-bench-scripts/${results_subdir}/" \
  "./${results_subdir}/"

# --- scp equivalent, if you prefer it (note: must pass the key + host-key opts) ---
# scp -r -i "$LAMBDA_SSH_KEY_PATH" \
#   -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
#   "${remote}:${LAMBDA_REMOTE_WORKDIR}/llm-bench-scripts/${results_subdir}/" .
