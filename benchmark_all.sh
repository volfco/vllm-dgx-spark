#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# vLLM Multi-Model Benchmark Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Benchmarks multiple models and creates a comparison matrix.
#
# Usage:
#   ./benchmark_all.sh [OPTIONS]
#
# Options:
#   --single-node     Only benchmark single-node models (TP=1)
#   --multi-node      Only benchmark multi-node models (TP=2)
#   --models "1,2,3"  Benchmark specific models by number
#   --skip-token      Skip models requiring HF token
#   --profile PROFILE Use specific benchmark profile (quick/short/medium/throughput)
#   --dry-run         Show what would be benchmarked without running
#   -h, --help        Show this help
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment
WORKER_HOST="${WORKER_HOST:-}"
WORKER_USER="${WORKER_USER:-$(whoami)}"
HF_CACHE="${HF_CACHE:-/raid/hf-cache}"
HF_TOKEN="${HF_TOKEN:-}"

# Results directory
RESULTS_DIR="${SCRIPT_DIR}/benchmark_results"
mkdir -p "${RESULTS_DIR}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model Definitions (copied from switch_model.sh)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODELS=(
  "openai/gpt-oss-120b"
  "openai/gpt-oss-20b"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "mistralai/Mistral-Nemo-Instruct-2407"
  "mistralai/Mixtral-8x7B-Instruct-v0.1"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.1-70B-Instruct"
  "microsoft/phi-4"
  "google/gemma-2-27b-it"
  "CohereForAI/c4ai-command-r-plus-08-2024"
  "nvidia/Llama-3.1-405B-Instruct-FP4"
  "meta-llama/Llama-3.3-70B-Instruct"
)

MODEL_NAMES=(
  "GPT-OSS-120B"
  "GPT-OSS-20B"
  "Qwen2.5-7B"
  "Qwen2.5-14B"
  "Qwen2.5-32B"
  "Qwen2.5-72B"
  "Mistral-7B"
  "Mistral-Nemo-12B"
  "Mixtral-8x7B"
  "Llama-3.1-8B"
  "Llama-3.1-70B"
  "Phi-4"
  "Gemma2-27B"
  "Command-R-Plus"
  "Llama-3.1-405B-FP4"
  "Llama-3.3-70B"
)

# Nodes required per model. Models that fit in a single DGX Spark's
# ~120GB VRAM use 1 node; larger models that must be split across two
# Sparks use 2 nodes.
MODEL_NODES=(
  1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2
)

MODEL_NEEDS_TOKEN=(
  false false false false false false false false false true true false true true true true
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SINGLE_NODE_ONLY=false
MULTI_NODE_ONLY=false
SKIP_TOKEN_MODELS=false
SPECIFIC_MODELS=""
PROFILE="quick"
DRY_RUN=false

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

check_hf_token() {
  [ -n "${HF_TOKEN:-}" ]
}

# Wait for API with progress updates
wait_for_api() {
  local timeout=$1
  local elapsed=0
  local last_log=""

  echo "  Waiting for API (timeout: ${timeout}s)..."

  while [ $elapsed -lt $timeout ]; do
    if curl -sf "http://localhost:8000/health" >/dev/null 2>&1; then
      echo "  API ready after ${elapsed}s"
      return 0
    fi

    # Show progress every 30 seconds with latest container log
    if [ $((elapsed % 30)) -eq 0 ] && [ $elapsed -gt 0 ]; then
      if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "ray-head"; then
        last_log=$(docker logs --tail 5 ray-head 2>&1 | grep -v "^$" | grep -v "HTTP/1.1" | tail -1 || echo "Loading...")
        echo "  [${elapsed}s] ${last_log:0:80}"
      else
        echo "  [${elapsed}s] Waiting for container..."
      fi
    fi

    sleep 5
    elapsed=$((elapsed + 5))
  done

  echo "  Timeout after ${timeout}s"
  return 1
}

usage() {
  cat << EOF
Usage: $0 [OPTIONS]

Benchmark multiple models and create a comparison matrix.

Options:
  --single-node     Only benchmark single-node models (TP=1)
  --multi-node      Only benchmark multi-node models (TP=2)
  --models "1,2,3"  Benchmark specific models by number
  --skip-token      Skip models requiring HuggingFace token
  --profile PROFILE Benchmark profile: quick, short, medium, throughput (default: quick)
  --dry-run         Show what would be benchmarked without running
  -h, --help        Show this help

Profiles:
  quick       10 prompts, low concurrency (fast sanity check)
  short       50 prompts, medium concurrency
  medium      100 prompts, high concurrency
  throughput  500 prompts, max concurrency

Examples:
  $0                           # Benchmark all models with quick profile
  $0 --single-node             # Only single-node models
  $0 --models "1,3,5"          # Specific models only
  $0 --profile throughput      # Full throughput test
  $0 --dry-run --single-node   # See what would be benchmarked

EOF
  exit 0
}

# Parse profile to benchmark args
get_profile_args() {
  local profile="$1"
  case "$profile" in
    quick)
      echo "-n 10 -c 10"
      ;;
    short)
      echo "-n 50 -c 32"
      ;;
    medium)
      echo "-n 100 -c 64"
      ;;
    throughput)
      echo "-n 500 -c 100"
      ;;
    *)
      echo "-n 10 -c 10"
      ;;
  esac
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Parse Arguments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

while [[ $# -gt 0 ]]; do
  case $1 in
    --single-node)
      SINGLE_NODE_ONLY=true
      shift
      ;;
    --multi-node)
      MULTI_NODE_ONLY=true
      shift
      ;;
    --models)
      SPECIFIC_MODELS="$2"
      shift 2
      ;;
    --skip-token)
      SKIP_TOKEN_MODELS=true
      shift
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Build Model List
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODELS_TO_BENCHMARK=()

if [ -n "${SPECIFIC_MODELS}" ]; then
  # Parse comma-separated list
  IFS=',' read -ra MODEL_NUMS <<< "${SPECIFIC_MODELS}"
  for num in "${MODEL_NUMS[@]}"; do
    num=$(echo "$num" | tr -d ' ')
    if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "${#MODELS[@]}" ]; then
      MODELS_TO_BENCHMARK+=("$((num-1))")
    fi
  done
else
  # Add all models based on filters
  for i in "${!MODELS[@]}"; do
    # Check single-node filter
    if [ "${SINGLE_NODE_ONLY}" = "true" ] && [ "${MODEL_NODES[$i]}" -ne 1 ]; then
      continue
    fi

    # Check multi-node filter
    if [ "${MULTI_NODE_ONLY}" = "true" ] && [ "${MODEL_NODES[$i]}" -ne 2 ]; then
      continue
    fi

    # Check token filter
    if [ "${SKIP_TOKEN_MODELS}" = "true" ] && [ "${MODEL_NEEDS_TOKEN[$i]}" = "true" ]; then
      continue
    fi

    MODELS_TO_BENCHMARK+=("$i")
  done
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Display Plan
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " vLLM Multi-Model Benchmark"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Profile:    ${PROFILE}"
echo "Models:     ${#MODELS_TO_BENCHMARK[@]} models to benchmark"
echo "Results:    ${RESULTS_DIR}"
echo ""

if [ "${#MODELS_TO_BENCHMARK[@]}" -eq 0 ]; then
  echo "No models selected for benchmarking."
  exit 1
fi

echo "Models to benchmark:"
for idx in "${MODELS_TO_BENCHMARK[@]}"; do
  MARKER=""
  if [ "${MODEL_NODES[$idx]}" -gt 1 ]; then
    MARKER=" [TP=2]"
  fi
  if [ "${MODEL_NEEDS_TOKEN[$idx]}" = "true" ]; then
    MARKER="${MARKER} [HF TOKEN]"
  fi
  printf "  %2d. %s%s\n" "$((idx+1))" "${MODEL_NAMES[$idx]}" "${MARKER}"
done
echo ""

if [ "${DRY_RUN}" = "true" ]; then
  echo "Dry run - exiting without benchmarking."
  exit 0
fi

# Confirm before starting
read -p "Start benchmarking ${#MODELS_TO_BENCHMARK[@]} models? This may take a while. (y/N): " CONFIRM
if [ "${CONFIRM}" != "y" ] && [ "${CONFIRM}" != "Y" ]; then
  echo "Cancelled."
  exit 0
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run Benchmarks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${RESULTS_DIR}/summary_${TIMESTAMP}.txt"
CSV_FILE="${RESULTS_DIR}/results_${TIMESTAMP}.csv"
JSON_FILE="${RESULTS_DIR}/results_${TIMESTAMP}.json"

# Initialize results arrays
declare -A BENCH_RESULTS

# Initialize CSV
echo "Model,TP,Output_TPS,Total_TPS,TTFT_ms,TPOT_ms,Status" > "${CSV_FILE}"

# Initialize JSON
echo '{"timestamp":"'"${TIMESTAMP}"'","profile":"'"${PROFILE}"'","results":[' > "${JSON_FILE}"

FIRST_RESULT=true
SUCCESSFUL=0
FAILED=0

PROFILE_ARGS=$(get_profile_args "${PROFILE}")

for idx in "${MODELS_TO_BENCHMARK[@]}"; do
  MODEL="${MODELS[$idx]}"
  MODEL_NAME="${MODEL_NAMES[$idx]}"
  MODEL_NUM=$((idx+1))

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " Benchmarking: ${MODEL_NAME} (${MODEL_NUM}/${#MODELS_TO_BENCHMARK[@]})"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  # Check for HF token if required
  if [ "${MODEL_NEEDS_TOKEN[$idx]}" = "true" ] && ! check_hf_token; then
    log "SKIPPED: ${MODEL_NAME} requires HF token"
    echo "${MODEL_NAME},${MODEL_NODES[$idx]},N/A,N/A,N/A,N/A,SKIPPED" >> "${CSV_FILE}"
    continue
  fi

  # Switch to model (runs start_cluster.sh internally)
  log "Switching to model (this may take several minutes)..."
  if ! "${SCRIPT_DIR}/switch_model.sh" --skip-download "${MODEL_NUM}" 2>&1 | tee /tmp/switch_${MODEL_NUM}.log; then
    log "FAILED: Could not switch to ${MODEL_NAME}"
    echo "${MODEL_NAME},${MODEL_NODES[$idx]},N/A,N/A,N/A,N/A,SWITCH_FAILED" >> "${CSV_FILE}"
    FAILED=$((FAILED + 1))
    continue
  fi

  # Wait for API to be ready (60s timeout - cluster should already be starting)
  log "Verifying API is ready..."
  if ! wait_for_api 60; then
    log "FAILED: API not ready for ${MODEL_NAME}"
    echo "${MODEL_NAME},${MODEL_NODES[$idx]},N/A,N/A,N/A,N/A,API_TIMEOUT" >> "${CSV_FILE}"
    FAILED=$((FAILED + 1))
    continue
  fi

  # Run benchmark
  log "Running benchmark with profile: ${PROFILE}..."
  BENCH_OUTPUT="${RESULTS_DIR}/bench_${MODEL_NUM}_${MODEL_NAME// /_}_${TIMESTAMP}.txt"

  if "${SCRIPT_DIR}/benchmark_current.sh" ${PROFILE_ARGS} 2>&1 | tee "${BENCH_OUTPUT}"; then
    # Extract metrics from output
    OUTPUT_TPS=$(grep -E "Output.*throughput" "${BENCH_OUTPUT}" | head -1 | grep -oE '[0-9]+\.?[0-9]*' | head -1 || echo "N/A")
    TOTAL_TPS=$(grep -E "Total.*throughput" "${BENCH_OUTPUT}" | head -1 | grep -oE '[0-9]+\.?[0-9]*' | head -1 || echo "N/A")
    MEAN_TTFT=$(grep -E "Mean TTFT" "${BENCH_OUTPUT}" | head -1 | grep -oE '[0-9]+\.?[0-9]*' | head -1 || echo "N/A")
    MEAN_TPOT=$(grep -E "Mean TPOT" "${BENCH_OUTPUT}" | head -1 | grep -oE '[0-9]+\.?[0-9]*' | head -1 || echo "N/A")

    log "Results: Output=${OUTPUT_TPS} tok/s, TTFT=${MEAN_TTFT}ms"

    # Store results
    BENCH_RESULTS["${MODEL_NAME}_output_tps"]="${OUTPUT_TPS}"
    BENCH_RESULTS["${MODEL_NAME}_total_tps"]="${TOTAL_TPS}"
    BENCH_RESULTS["${MODEL_NAME}_ttft"]="${MEAN_TTFT}"
    BENCH_RESULTS["${MODEL_NAME}_tpot"]="${MEAN_TPOT}"

    # Add to CSV
    echo "${MODEL_NAME},${MODEL_NODES[$idx]},${OUTPUT_TPS},${TOTAL_TPS},${MEAN_TTFT},${MEAN_TPOT},SUCCESS" >> "${CSV_FILE}"

    # Add to JSON
    if [ "${FIRST_RESULT}" != "true" ]; then
      echo "," >> "${JSON_FILE}"
    fi
    FIRST_RESULT=false

    cat >> "${JSON_FILE}" << EOF
  {
    "model": "${MODEL}",
    "name": "${MODEL_NAME}",
    "tp": ${MODEL_NODES[$idx]},
    "output_throughput_tps": ${OUTPUT_TPS:-null},
    "total_throughput_tps": ${TOTAL_TPS:-null},
    "mean_ttft_ms": ${MEAN_TTFT:-null},
    "mean_tpot_ms": ${MEAN_TPOT:-null},
    "status": "success"
  }
EOF

    SUCCESSFUL=$((SUCCESSFUL + 1))
  else
    log "FAILED: Benchmark failed for ${MODEL_NAME}"
    echo "${MODEL_NAME},${MODEL_NODES[$idx]},N/A,N/A,N/A,N/A,BENCH_FAILED" >> "${CSV_FILE}"
    FAILED=$((FAILED + 1))
  fi
done

# Close JSON
echo "" >> "${JSON_FILE}"
echo "]}" >> "${JSON_FILE}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generate Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Benchmark Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

{
  echo "vLLM Multi-Model Benchmark Results"
  echo "=================================="
  echo ""
  echo "Timestamp: $(date)"
  echo "Profile:   ${PROFILE}"
  echo "Successful: ${SUCCESSFUL}"
  echo "Failed:     ${FAILED}"
  echo ""
  echo "Results Matrix:"
  echo "---------------"
  echo ""
  printf "%-20s %4s %10s %10s %10s %10s\n" "Model" "TP" "Out TPS" "Tot TPS" "TTFT(ms)" "TPOT(ms)"
  printf "%-20s %4s %10s %10s %10s %10s\n" "--------------------" "----" "----------" "----------" "----------" "----------"

  # Read CSV and format
  tail -n +2 "${CSV_FILE}" | while IFS=',' read -r model tp out_tps tot_tps ttft tpot status; do
    if [ "${status}" = "SUCCESS" ]; then
      printf "%-20s %4s %10s %10s %10s %10s\n" "${model}" "${tp}" "${out_tps}" "${tot_tps}" "${ttft}" "${tpot}"
    else
      printf "%-20s %4s %10s %10s %10s %10s\n" "${model}" "${tp}" "-" "-" "-" "(${status})"
    fi
  done

  echo ""
  echo "Files Generated:"
  echo "  Summary: ${SUMMARY_FILE}"
  echo "  CSV:     ${CSV_FILE}"
  echo "  JSON:    ${JSON_FILE}"
} | tee "${SUMMARY_FILE}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Benchmark Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "View results:"
echo "  cat ${CSV_FILE}"
echo "  cat ${JSON_FILE} | python3 -m json.tool"
echo ""
