#!/usr/bin/env bash
set -euo pipefail

cd ~
python3 -m venv venv
source venv/bin/activate 
git clone https://github.com/Ak33ra/vllm.git 
cd vllm/
VLLM_USE_PRECOMPILED=1 pip install -e .
pip install vllm[bench]
pip install "fastapi<0.137"
cd ~
git clone https://github.com/Ak33ra/llm-bench-scripts 
