#!/usr/bin/env bash
set -euo pipefail

cd ~
source venv/bin/activate
cd llm-bench-scripts/
./run_iterlog_sweep.sh 
