#!/usr/bin/env bash
#
# run_iterlog_sweep.sh — for each workload/burstiness/seed config:
#   1. start vLLM with iteration logging piped to burst<b>/log_<seed>.txt
#   2. wait until the server is ready
#   3. (optional) start nsys and/or py-spy profiling of the SERVER
#   4. run the matching benchmark -> burst<b>/seed<seed>.json
#   5. (optional) stop profiling -> burst<b>/<base>.nsys-rep / .pyspy.<ext>
#   6. stop the server
# then move to the next config. One full server restart per config.
#
# Profiling is opt-in via --profiler and targets the vLLM *server* (where the
# iteration-level decode work lives), not the bench client. Scope = the
# benchmark window only: collection starts just before `vllm bench serve` and
# stops just after it returns, so server startup / weight load is excluded.
# A <base>.profmeta.json sidecar records the wall-clock epoch at collection
# start (profiler time t=0) so the profiler timeline can be aligned to the
# wall-clock timestamps in the iteration log.
#
# Examples:
#   ./run_iterlog_sweep.sh                     # no profiling (default, unchanged behavior)
#   ./run_iterlog_sweep.sh --profiler pyspy    # py-spy only: chrometrace @100Hz, --subprocesses
#   ./run_iterlog_sweep.sh --profiler nsys     # nsys only: -t cuda,nvtx,osrt --sample none
#   ./run_iterlog_sweep.sh --profiler both     # nsys + py-spy together (higher overhead; see usage())
#   ./run_iterlog_sweep.sh --profiler torch    # vLLM built-in PyTorch profiler (server-gated)
#
#   # tweak profiler defaults:
#   ./run_iterlog_sweep.sh --profiler pyspy --pyspy-rate 250 --pyspy-extra "--nonblocking"
#   ./run_iterlog_sweep.sh --profiler pyspy --pyspy-format speedscope
#   ./run_iterlog_sweep.sh --profiler nsys  --nsys-trace cuda,nvtx --nsys-sample process-tree
#   ./run_iterlog_sweep.sh --profiler torch --torch-delay-iters 2000 --torch-max-iters 300
#                                              # torch: profile a 300-iter steady-state slice
#
# Per-config outputs land in in<I>out<O>/rate<R>/burst<b>/ sharing the result basename:
#   prompts<N>seed<S>.json  (+ .nsys-rep / .pyspy.<ext> / .profmeta.json when profiling)
#
# The torch profiler is different from nsys/py-spy: it is NOT an attach. vLLM's
# own engine writes the traces, gated by `vllm serve --profiler-config` (enables
# it) plus `vllm bench serve --profile` (sends /start_profile + /stop_profile
# around the request loop). vLLM names the trace files itself (rank suffix +
# timestamp), so to follow the basename scheme each config's traces are dumped
# into their OWN per-config subdir:
#   burst<b>/prompts<N>seed<S>_torchprof/
#     <dp..tp..>_rank<R>.<ts>.pt.trace.json.gz   <- per-worker GPU+CPU traces (the useful ones)
#     <host>_<pid>.async_llm.<ts>.pt.trace.json.gz  <- front-end CPU-only trace
#     profiler_out_<rank>.txt                       <- text summary table
# These are gzip'd Perfetto/Chrome traces (NOT tar.gz) — view directly at
# https://ui.perfetto.dev/ , no untar needed. NOTE: torch profiles the ENTIRE
# benchmark window with no iteration cap, so at high --num-prompts the traces get
# large and slow to flush; VLLM_RPC_TIMEOUT is bumped automatically to cover it.
# Run `./run_iterlog_sweep.sh --help` for the full option list.
#
# Other knobs are environment variables (not CLI flags), e.g.:
#   HEALTH_PATH=/v1/models READY_TIMEOUT=900 SETTLE_SECS=5 ./run_iterlog_sweep.sh --profiler nsys
#
set -euo pipefail

# ============================ shared constants ============================
model="meta-llama/Meta-Llama-3-70B"
model_name="llama-3-70b"
engine="vllm-tp2"
gpu="h200"

# Workload sweep values. Single-value arrays preserve the old defaults while
# keeping every workload parameter in the sweep loop where paths are built.
request_rates=(10 20)
num_prompts_values=(750)
input_lens=(128)
output_lens=(128)

# server launch flags (kept from your serve script; server must serve the
# SAME model the benchmark hits, so both are driven from $model)
dtype="bfloat16"
max_model_len=4096

# where the server listens / how we detect readiness
host="localhost"
base_port="${BASE_PORT:-8000}"
case "$base_port" in
  ''|*[!0-9]*) echo "BASE_PORT must be a non-negative integer (got '${base_port}')" >&2; exit 2;;
esac
cuda_port_offset() {
  local visible="${CUDA_VISIBLE_DEVICES:-}"
  [ -z "$visible" ] && { echo 0; return 0; }

  visible="${visible//[[:space:]]/}"
  local selected="" device
  local -a devices
  IFS=',' read -ra devices <<< "$visible"
  for device in "${devices[@]}"; do
    [ -z "$device" ] && continue
    case "$device" in
      *[!0-9]*)
        echo "!!! CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' is not a numeric index list; using base port ${base_port}" >&2
        echo 0
        return 0
        ;;
    esac
    if [ -z "$selected" ] || [ "$device" -lt "$selected" ]; then
      selected="$device"
    fi
  done

  echo "${selected:-0}"
}
port=$((base_port + $(cuda_port_offset)))
bench_base_url="http://${host}:${port}"
health_path="${HEALTH_PATH:-/health}"   # set HEALTH_PATH=/v1/models if /health reports ready before weights finish loading
ready_timeout="${READY_TIMEOUT:-600}"   # seconds to wait for readiness before skipping the config
poll_interval=2
settle_secs="${SETTLE_SECS:-3}"         # grace after shutdown for GPU memory to release

# ============================ profiling config ===========================
# All overridable on the command line (see usage()). Defaults keep the old
# behavior: profiler=none means nothing about the run changes.
profiler="none"                  # none | nsys | pyspy | both | torch
nsys_trace="cuda,nvtx,osrt"      # nsys -t (CUDA + NVTX ranges + OS runtime)
nsys_sample="none"               # nsys --sample; 'none' disables CPU backtrace
                                 #   sampling -> smaller/cheaper traces, the
                                 #   CUDA timeline is the point here.
nsys_extra=""                    # extra args passed to `nsys launch`
pyspy_format="chrometrace"       # flamegraph | raw | speedscope | chrometrace
                                 #   chrometrace/speedscope keep a time axis;
                                 #   raw/flamegraph are aggregate-only.
pyspy_rate="100"                 # py-spy --rate (samples/sec)
pyspy_extra=""                   # extra args to `py-spy record`
                                 #   e.g. "--nonblocking" (don't pause the
                                 #   server -> less timing distortion) or
                                 #   "--native" (include C/CUDA-ext frames).
pyspy_subprocesses=1             # vLLM v1 runs decode in an EngineCore *child*
                                 #   process, so --subprocesses is required to
                                 #   profile the GPU loop. --no-subprocesses
                                 #   turns it off.

# torch profiler (vLLM built-in). Enabled via `vllm serve --profiler-config` on
# the server + `vllm bench serve --profile` on the client; vLLM writes the
# traces itself into a per-config subdir. VLLM_RPC_TIMEOUT is bumped so the
# (potentially slow) trace flush on /stop_profile doesn't time out.
torch_rpc_timeout="${VLLM_RPC_TIMEOUT:-1800000}"  # ms; vLLM-recommended large value (30 min)
# Iteration-bounded collection: the bench's --profile arms the profiler at the
# very start of the request loop, but the server can skip the first
# delay_iterations engine steps and then profile only max_iterations of them,
# carving a steady-state slice [D, D+M) out of a long run instead of tracing the
# whole thing. One engine step == one "Iteration(" line in the iter log, so to
# land in the middle of a run: T=$(grep -c 'Iteration(' <a prior log>); set
# --torch-delay-iters ~T/2 and --torch-max-iters to the slice width (e.g. 300).
torch_delay_iters=0              # engine steps to skip after /start_profile (0 = none)
torch_max_iters=0                # engine steps to actually profile (0 = until /stop_profile)
                                 #   when either is >0, ignore_frontend is forced
                                 #   true (else the CPU front-end trace captures
                                 #   the entire run and defeats the slicing).

usage() {
  cat <<'EOF'
Usage: run_iterlog_sweep.sh [options]

  --profiler MODE       none | nsys | pyspy | both | torch  (default: none)
  --nsys-trace LIST     nsys -t trace selectors        (default: cuda,nvtx,osrt)
  --nsys-sample MODE    nsys --sample                  (default: none)
  --nsys-extra "ARGS"   extra args forwarded to 'nsys launch'
  --pyspy-format FMT    flamegraph|raw|speedscope|chrometrace (default: chrometrace)
  --pyspy-rate HZ       py-spy --rate, samples/sec      (default: 100)
  --pyspy-extra "ARGS"  extra args to 'py-spy record'  (e.g. "--nonblocking --native")
  --no-subprocesses     don't pass --subprocesses to py-spy
                        (NOTE: vLLM v1 needs it to see the EngineCore process)
  --torch-delay-iters N engine steps to skip before torch starts profiling (default: 0)
  --torch-max-iters N   engine steps to profile, then auto-stop      (default: 0 = whole run)
                        (1 engine step == 1 "Iteration(" line in the iter log.
                         For a steady-state middle slice of a long run:
                           T=$(grep -c 'Iteration(' SOME_PRIOR_LOG.txt)
                           --torch-delay-iters $((T/2)) --torch-max-iters 300
                         Either >0 forces ignore_frontend=true.)
  -h, --help

Profiler outputs land next to the JSON result, sharing its basename:
  prompts<N>seed<S>.nsys-rep      (nsys report)
  prompts<N>seed<S>.pyspy.<ext>   (svg/txt/json depending on --pyspy-format)
  prompts<N>seed<S>_torchprof/    (dir of *.pt.trace.json.gz + profiler_out_<rank>.txt)
  prompts<N>seed<S>.profmeta.json (epochs to align profiler time <-> iter log)

Notes:
  * nsys cannot attach to a running PID, so when nsys is on the server is
    launched under `nsys launch` and collection is gated by start/stop around
    the benchmark window.
  * 'both' runs nsys + py-spy simultaneously. py-spy briefly pauses the server
    to read stacks (unless --pyspy-extra "--nonblocking") and nsys/CUPTI adds
    launch overhead, so profiled runs' TPOT/ITL are perturbed — treat them as
    "where does the time go" runs, not as clean timing data.
  * 'torch' is standalone (not combinable with nsys/pyspy). The server is
    launched with --profiler-config and the bench with --profile; vLLM writes
    one gzip'd Perfetto trace per worker (rank*) plus a front-end trace and a
    profiler_out_<rank>.txt summary into the per-config _torchprof/ dir. By
    default it profiles the whole benchmark window; use --torch-delay-iters /
    --torch-max-iters to instead capture a bounded steady-state slice [D, D+M)
    of engine iterations while the full run still executes.
EOF
}

# ------------------------------ CLI parsing ------------------------------
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profiler)        profiler="$2";           shift 2;;
    --nsys-trace)      nsys_trace="$2";         shift 2;;
    --nsys-sample)     nsys_sample="$2";        shift 2;;
    --nsys-extra)      nsys_extra="$2";         shift 2;;
    --pyspy-format)    pyspy_format="$2";       shift 2;;
    --pyspy-rate)      pyspy_rate="$2";         shift 2;;
    --pyspy-extra)     pyspy_extra="$2";        shift 2;;
    --no-subprocesses) pyspy_subprocesses=0;    shift;;
    --torch-delay-iters) torch_delay_iters="$2"; shift 2;;
    --torch-max-iters)   torch_max_iters="$2";   shift 2;;
    -h|--help)         usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

# resolve which profilers are on
NSYS_ON=0
PYSPY_ON=0
TORCH_ON=0
case "$profiler" in
  none)  ;;
  nsys)  NSYS_ON=1;;
  pyspy) PYSPY_ON=1;;
  both)  NSYS_ON=1; PYSPY_ON=1;;
  torch) TORCH_ON=1;;   # vLLM built-in; server-gated, not an attach (standalone)
  *) echo "bad --profiler '$profiler' (want: none|nsys|pyspy|both|torch)" >&2; exit 2;;
esac

# vLLM recommends a large RPC timeout so the (possibly slow) trace flush on
# /stop_profile doesn't time out. Export it so the server child inherits it.
if [ "$TORCH_ON" = 1 ]; then
  export VLLM_RPC_TIMEOUT="$torch_rpc_timeout"
  for v in torch_delay_iters torch_max_iters; do
    case "${!v}" in
      ''|*[!0-9]*) echo "--${v//_/-} must be a non-negative integer (got '${!v}')" >&2; exit 2;;
    esac
  done
fi
# whether torch collection is bounded to a slice (vs. the whole window)
TORCH_BOUNDED=0
if [ "$TORCH_ON" = 1 ] && { [ "$torch_delay_iters" -gt 0 ] || [ "$torch_max_iters" -gt 0 ]; }; then
  TORCH_BOUNDED=1
fi

# fail fast if a requested profiler isn't installed
if [ "$NSYS_ON" = 1 ] && ! command -v nsys >/dev/null 2>&1; then
  echo "!!! --profiler '$profiler' needs nsys, but it is not on PATH" >&2
  echo "    install NVIDIA Nsight Systems, or pick --profiler pyspy/none" >&2
  exit 3
fi
if [ "$PYSPY_ON" = 1 ] && ! command -v py-spy >/dev/null 2>&1; then
  echo "!!! --profiler '$profiler' needs py-spy, but it is not on PATH" >&2
  exit 3
fi

# map a py-spy format to the file extension it writes
pyspy_ext() {
  case "$1" in
    flamegraph)            echo "svg";;
    raw)                   echo "txt";;
    speedscope|chrometrace) echo "json";;
    *)                     echo "out";;
  esac
}

output_root="./${engine}/${gpu}/${model_name}"
failures_log="./sweep_failures.log"
: > "$failures_log"

# ============================ server lifecycle ============================
CURRENT_SERVER_PID=""
PYSPY_PID=""          # py-spy record process, while collecting
NSYS_SESSION=""       # named nsys session for the current config, while live

# Tear down any profiler still running for the current config. Safe to call
# repeatedly / when nothing is running. Invoked both on the normal path and
# from cleanup() so Ctrl-C mid-benchmark never leaves a profiler attached.
stop_profilers() {
  if [ -n "${PYSPY_PID:-}" ]; then
    kill -INT "$PYSPY_PID" 2>/dev/null || true   # SIGINT -> py-spy flushes its file
    wait "$PYSPY_PID" 2>/dev/null || true
    PYSPY_PID=""
  fi
  if [ "$NSYS_ON" = 1 ] && [ -n "${NSYS_SESSION:-}" ]; then
    # ends the session and SIGTERMs the launched server's process group
    nsys shutdown --session "$NSYS_SESSION" --kill sigterm >/dev/null 2>&1 || true
    NSYS_SESSION=""
  fi
}

stop_server() {
  stop_profilers
  local pid="$CURRENT_SERVER_PID"
  [ -z "$pid" ] && return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true            # SIGTERM: let vLLM release the GPU cleanly
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true   # force if still alive
  fi
  wait "$pid" 2>/dev/null || true
  CURRENT_SERVER_PID=""
  # wait until the old server stops answering so the next launch can bind
  for _ in $(seq 1 30); do
    curl -sf "http://${host}:${port}${health_path}" >/dev/null 2>&1 || break
    sleep 1
  done
  sleep "$settle_secs"
}

cleanup() {
  echo ">>> cleanup: stopping server (pid ${CURRENT_SERVER_PID:-none})" >&2
  stop_server
}
trap cleanup EXIT INT TERM   # never leave vLLM holding GPU memory on Ctrl-C / error

start_server() {
  local log_file="$1"
  # When nsys is on, launch the server under `nsys launch` so a later
  # `nsys start`/`nsys stop` can gate collection to the benchmark window.
  # launch instruments but does NOT collect, so weight load runs un-traced and
  # /health still comes up normally.
  local launch_prefix=()
  if [ "$NSYS_ON" = 1 ]; then
    # NSYS_SESSION is set by the loop before this call.
    nsys shutdown --session "$NSYS_SESSION" --kill sigterm >/dev/null 2>&1 || true  # drop any stale same-name session
    # -t/--trace is set here on launch; --sample is set on `nsys start` below
    # (recent nsys deprecates --sample on launch).
    launch_prefix=(nsys launch --session-new "$NSYS_SESSION"
                   -t "$nsys_trace" $nsys_extra)
  fi
  # When torch profiling is on, enable vLLM's built-in profiler and point it at
  # this config's per-config trace dir (TORCH_DIR_ABS, set by the loop). vLLM
  # requires an absolute path; collection itself is gated client-side by the
  # bench's --profile flag, so weight load / warmup is not captured.
  local torch_args=()
  if [ "$TORCH_ON" = 1 ]; then
    local cfg="\"profiler\": \"torch\", \"torch_profiler_dir\": \"${TORCH_DIR_ABS}\""
    # When slicing to a middle window, hand the server the iteration bounds and
    # disable front-end profiling (it can't track iterations and would otherwise
    # capture the whole run).
    if [ "$TORCH_BOUNDED" = 1 ]; then
      cfg="${cfg}, \"delay_iterations\": ${torch_delay_iters}, \"max_iterations\": ${torch_max_iters}, \"ignore_frontend\": true"
    fi
    torch_args=(--profiler-config "{${cfg}}")
  fi
  # Merge stdout+stderr, filter to iteration lines, write to log_file.
  # Process substitution keeps $! pointing at vLLM (or the nsys launcher that
  # owns it), not grep, so we can kill it.
  ${launch_prefix[@]+"${launch_prefix[@]}"} vllm serve "$model" \
    --dtype "$dtype" \
    --max-model-len "$max_model_len" \
    --tensor-parallel-size 2 \
    --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}' \
    --port "$port" \
    --enable-logging-iteration-details \
    ${torch_args[@]+"${torch_args[@]}"} \
    > >(grep --line-buffered "Iteration(" > "$log_file") 2>&1 &
  CURRENT_SERVER_PID=$!
}

wait_ready() {
  local url="http://${host}:${port}${health_path}"
  local waited=0
  until curl -sf "$url" >/dev/null 2>&1; do
    kill -0 "$CURRENT_SERVER_PID" 2>/dev/null || return 1   # server died during load
    [ "$waited" -ge "$ready_timeout" ] && return 1          # timed out
    sleep "$poll_interval"
    waited=$((waited + poll_interval))
  done
  return 0
}

# ============================ profiler control ===========================
# Globals set by start_profiling for use by the sidecar writer.
PYSPY_OUT=""
NSYS_OUT=""
TORCH_DIR_ABS=""     # per-config torch trace dir (absolute); set by the loop
PROF_START_EPOCH=""

# Begin collection right before the benchmark. $1 = output path prefix (no ext).
start_profiling() {
  local base="$1"
  PYSPY_OUT=""; NSYS_OUT=""
  PROF_START_EPOCH=$(date +%s.%N)   # wall-clock anchor = profiler time t=0

  if [ "$NSYS_ON" = 1 ]; then
    NSYS_OUT="${base}.nsys-rep"
    # nsys appends .nsys-rep itself, so hand it the bare base. --sample is set
    # here (not on launch) per recent nsys; 'none' disables CPU backtrace sampling.
    if ! nsys start --session "$NSYS_SESSION" --sample "$nsys_sample" -o "$base" >/dev/null 2>&1; then
      echo "!!! nsys start failed for ${base} (session ${NSYS_SESSION})" >&2
    fi
  fi

  if [ "$PYSPY_ON" = 1 ]; then
    local ext; ext=$(pyspy_ext "$pyspy_format")
    PYSPY_OUT="${base}.pyspy.${ext}"
    local sub=(); [ "$pyspy_subprocesses" = 1 ] && sub=(--subprocesses)
    # attach to the live server (and, with --subprocesses, its EngineCore child)
    py-spy record --pid "$CURRENT_SERVER_PID" ${sub[@]+"${sub[@]}"} \
      --rate "$pyspy_rate" --format "$pyspy_format" \
      --output "$PYSPY_OUT" $pyspy_extra >/dev/null 2>&1 &
    PYSPY_PID=$!
  fi
}

# End collection right after the benchmark returns; finalizes both files.
PROF_STOP_EPOCH=""
stop_profiling() {
  PROF_STOP_EPOCH=$(date +%s.%N)
  if [ -n "${PYSPY_PID:-}" ]; then
    kill -INT "$PYSPY_PID" 2>/dev/null || true
    wait "$PYSPY_PID" 2>/dev/null || true
    PYSPY_PID=""
  fi
  if [ "$NSYS_ON" = 1 ]; then
    if ! nsys stop --session "$NSYS_SESSION" >/dev/null 2>&1; then
      echo "!!! nsys stop failed for session ${NSYS_SESSION}" >&2
    fi
  fi
}

# Write the alignment sidecar. $1=base $2=bench_start_epoch $3=bench_end_epoch
write_profmeta() {
  [ "$profiler" = "none" ] && return 0
  local base="$1" bench_start="$2" bench_end="$3"
  local meta="${base}.profmeta.json"

  local pyspy_json="null" nsys_json="null" torch_json="null"
  if [ "$TORCH_ON" = 1 ]; then
    torch_json=$(printf '{"trace_dir":"%s","glob":"%s","rpc_timeout_ms":"%s","bounded":%s,"delay_iterations":%s,"max_iterations":%s,"ignore_frontend":%s}' \
      "$(basename "$TORCH_DIR_ABS")" "*.pt.trace.json.gz" "${VLLM_RPC_TIMEOUT:-}" \
      "$([ "$TORCH_BOUNDED" = 1 ] && echo true || echo false)" \
      "$torch_delay_iters" "$torch_max_iters" \
      "$([ "$TORCH_BOUNDED" = 1 ] && echo true || echo false)")
  fi
  if [ "$PYSPY_ON" = 1 ]; then
    pyspy_json=$(printf '{"output":"%s","format":"%s","rate":%s,"subprocesses":%s,"extra":"%s"}' \
      "$(basename "$PYSPY_OUT")" "$pyspy_format" "$pyspy_rate" \
      "$([ "$pyspy_subprocesses" = 1 ] && echo true || echo false)" "$pyspy_extra")
  fi
  if [ "$NSYS_ON" = 1 ]; then
    nsys_json=$(printf '{"output":"%s","trace":"%s","sample":"%s","session":"%s","extra":"%s"}' \
      "$(basename "$NSYS_OUT")" "$nsys_trace" "$nsys_sample" "$NSYS_SESSION" "$nsys_extra")
  fi

  cat > "$meta" <<EOF
{
  "profiler": "${profiler}",
  "config": {
    "model": "${model}", "engine": "${engine}", "gpu": "${gpu}",
    "burstiness": ${burstiness}, "seed": ${seed}, "num_prompts": ${num_prompts},
    "request_rate": ${request_rate}, "input_len": ${input_len}, "output_len": ${output_len}
  },
  "iter_log": "$(basename "$log_file")",
  "result_json": "$(basename "$base").json",
  "epochs": {
    "_note": "unix epoch seconds; collection_start == profiler timeline t=0",
    "collection_start": ${PROF_START_EPOCH},
    "collection_stop": ${PROF_STOP_EPOCH},
    "bench_start": ${bench_start},
    "bench_end": ${bench_end}
  },
  "pyspy": ${pyspy_json},
  "nsys": ${nsys_json},
  "torch": ${torch_json}
}
EOF
}

# ================================ sweep ==================================
echo ">>> sweep: model=${model} root=${output_root} port=${port} profiler=${profiler}" >&2

for input_len in "${input_lens[@]}"; do
for output_len in "${output_lens[@]}"; do
for request_rate in "${request_rates[@]}"; do
for num_prompts in "${num_prompts_values[@]}"; do
for burstiness in 1.0 0.5 0.1; do
for seed in 0 1 2; do
  result_root="${output_root}/in${input_len}out${output_len}"
  out_dir="${result_root}/rate${request_rate}/burst${burstiness}"
  mkdir -p "$out_dir"
  log_file="${out_dir}/num_prompts${num_prompts}log_${seed}.txt"
  base="${out_dir}/prompts${num_prompts}seed${seed}"   # shared basename for json + profiler outputs
  tag="in=${input_len} out=${output_len} rate=${request_rate} prompts=${num_prompts} burst=${burstiness} seed=${seed}"
  burst_tag="${burstiness//./p}"
  NSYS_SESSION="vllmprof_i${input_len}_o${output_len}_r${request_rate}_p${num_prompts}_b${burst_tag}_s${seed}"  # per-config nsys session name

  # Per-config torch trace dir. vLLM needs an absolute path and writes its own
  # rank*/async_llm trace files + profiler_out_<rank>.txt into it, so the
  # basename scheme is carried by the dir name. Created up front (empty dirs are
  # harmless when torch is off).
  TORCH_DIR_ABS=""
  if [ "$TORCH_ON" = 1 ]; then
    torch_dir="${base}_torchprof"
    mkdir -p "$torch_dir"
    TORCH_DIR_ABS="$(cd "$torch_dir" && pwd)"
  fi

  echo ">>> [${tag}] starting server, iter log -> ${log_file}" >&2
  start_server "$log_file"

  if ! wait_ready; then
    echo "!!! [${tag}] server not ready within ${ready_timeout}s — skipping" >&2
    echo "${tag}: server-not-ready" >> "$failures_log"
    stop_server
    continue
  fi

  echo ">>> [${tag}] server ready, profiler=${profiler}, running benchmark" >&2
  start_profiling "$base"
  # torch profiling is gated by the bench's --profile flag (/start_profile +
  # /stop_profile around the request loop); other modes leave this empty.
  bench_profile_flag=()
  [ "$TORCH_ON" = 1 ] && bench_profile_flag=(--profile)
  bench_start_epoch=$(date +%s.%N)
  if vllm bench serve \
      --model "$model" \
      --backend openai \
      --base-url "$bench_base_url" \
      --endpoint /v1/completions \
      --dataset-name random \
      --random-input-len "$input_len" \
      --random-output-len "$output_len" \
      --ignore-eos \
      --num-prompts "$num_prompts" \
      --request-rate "$request_rate" \
      --seed "$seed" \
      --burstiness "$burstiness" \
      --save-result \
      --save-detailed \
      --metric-percentiles "50,90,95,99" \
      --percentile-metrics "ttft,tpot,itl,e2el" \
      --result-dir "$out_dir" \
      --result-filename "prompts${num_prompts}seed${seed}.json" \
      ${bench_profile_flag[@]+"${bench_profile_flag[@]}"}
  then
    echo ">>> [${tag}] benchmark done" >&2
  else
    rc=$?
    echo "!!! [${tag}] benchmark failed (exit ${rc}) — continuing" >&2
    echo "${tag}: benchmark-exit-${rc}" >> "$failures_log"
  fi
  bench_end_epoch=$(date +%s.%N)

  stop_profiling
  write_profmeta "$base" "$bench_start_epoch" "$bench_end_epoch"

  stop_server
done
done
done
done
done
done

echo ">>> sweep complete" >&2
if [ -s "$failures_log" ]; then
  echo "!!! some configs failed:" >&2
  cat "$failures_log" >&2
  exit 1
fi
