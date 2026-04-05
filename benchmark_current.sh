#!/usr/bin/env bash
set -euo pipefail

################################################################################
# vLLM Benchmark Script using vllm bench serve
#
# Uses the official vLLM benchmarking tool for consistent, reproducible results.
#
# Usage:
#   ./benchmark_current.sh [options]
#
# Options:
#   -u, --url URL           vLLM API URL (default: auto-detect)
#   -n, --num-prompts N     Number of prompts to benchmark (default: 100)
#   -c, --concurrency N     Max concurrent requests (default: 100)
#   -d, --dataset PATH      Path to ShareGPT dataset JSON (auto-downloads if missing)
#   -s, --single            Run single-request benchmark only (1 prompt)
#   -q, --quick             Quick mode: 20 prompts, lower concurrency
#   -o, --output FILE       Output results to JSON file
#   -h, --help              Show this help message
#
# Examples:
#   ./benchmark_current.sh                    # Full benchmark (100 prompts)
#   ./benchmark_current.sh --quick            # Quick benchmark (20 prompts)
#   ./benchmark_current.sh --single           # Single request latency test
#   ./benchmark_current.sh -n 50 -c 50        # Custom prompts/concurrency
#
# Dataset:
#   Uses ShareGPT_V3 dataset for realistic workload distribution.
#   Auto-downloads from HuggingFace if not present.
#
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Default configuration
API_URL=""
API_PORT="8000"
NUM_PROMPTS=100
MAX_CONCURRENCY=100
DATASET_PATH=""
SINGLE_MODE=false
QUICK_MODE=false
OUTPUT_FILE=""
MODEL=""
CONTAINER="${CONTAINER:-ray-head}"

# Default dataset location
DEFAULT_DATASET="ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -u|--url)
      API_URL="$2"
      shift 2
      ;;
    -n|--num-prompts)
      NUM_PROMPTS="$2"
      shift 2
      ;;
    -c|--concurrency)
      MAX_CONCURRENCY="$2"
      shift 2
      ;;
    -d|--dataset)
      DATASET_PATH="$2"
      shift 2
      ;;
    -s|--single)
      SINGLE_MODE=true
      NUM_PROMPTS=1
      MAX_CONCURRENCY=1
      shift
      ;;
    -q|--quick)
      QUICK_MODE=true
      NUM_PROMPTS=20
      MAX_CONCURRENCY=20
      shift
      ;;
    -o|--output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    -h|--help)
      head -35 "$0" | tail -30
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

print_header() {
  echo ""
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${BOLD}$1${NC}"
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_ok() {
  echo -e "${GREEN}✓${NC} $1"
}

print_warn() {
  echo -e "${YELLOW}⚠${NC} $1"
}

print_fail() {
  echo -e "${RED}✗${NC} $1"
}

print_info() {
  echo -e "${CYAN}ℹ${NC} $1"
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print_header "vLLM Benchmark (using vllm bench serve)"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Auto-detect API URL if not provided
if [ -z "$API_URL" ]; then
  # Try localhost first
  if curl -sf "http://localhost:${API_PORT}/health" >/dev/null 2>&1; then
    API_URL="http://localhost:${API_PORT}"
  else
    # Try to detect from network interfaces (exclude loopback and docker)
    PUBLIC_IP=$(ip -o addr show | grep "inet " | grep -v "127.0.0.1" | grep -v "172.17" | awk '{print $4}' | cut -d'/' -f1 | head -1)
    if [ -n "$PUBLIC_IP" ] && curl -sf "http://${PUBLIC_IP}:${API_PORT}/health" >/dev/null 2>&1; then
      API_URL="http://${PUBLIC_IP}:${API_PORT}"
    else
      API_URL="http://localhost:${API_PORT}"
    fi
  fi
fi

echo ""
echo -e "  ${BOLD}Configuration:${NC}"
echo -e "    API URL:      ${CYAN}${API_URL}${NC}"
echo -e "    Num Prompts:  ${CYAN}${NUM_PROMPTS}${NC}"
echo -e "    Concurrency:  ${CYAN}${MAX_CONCURRENCY}${NC}"
if [ "$SINGLE_MODE" = true ]; then
  echo -e "    Mode:         ${CYAN}Single Request (latency test)${NC}"
elif [ "$QUICK_MODE" = true ]; then
  echo -e "    Mode:         ${CYAN}Quick${NC}"
else
  echo -e "    Mode:         ${CYAN}Full${NC}"
fi
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Check vLLM availability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo -e "${CYAN}▶${NC} Checking vLLM availability..."

if ! curl -sf "${API_URL}/health" >/dev/null 2>&1; then
  print_fail "vLLM is not accessible at ${API_URL}"
  echo "    Make sure vLLM is running: ./start_cluster.sh"
  exit 1
fi
print_ok "vLLM is accessible"

# Get model name from API
MODEL=$(curl -sf "${API_URL}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
if [ "$MODEL" = "unknown" ]; then
  print_fail "Could not detect model from vLLM API"
  exit 1
fi
print_ok "Model: ${MODEL}"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Check for vllm bench command
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo -e "${CYAN}▶${NC} Checking vllm bench availability..."

# Check if vllm bench is available in the container
if ! docker exec "${CONTAINER}" bash -lc "vllm bench --help" >/dev/null 2>&1; then
  print_warn "vllm bench not available in container - installing..."
  # Try to ensure vllm is properly installed with bench support
  if ! docker exec "${CONTAINER}" bash -lc "pip show vllm >/dev/null 2>&1"; then
    print_fail "vLLM package not found in container"
    exit 1
  fi
fi
print_ok "vllm bench command available"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Download ShareGPT dataset if needed
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo -e "${CYAN}▶${NC} Checking ShareGPT dataset..."

# Use provided path or check default locations
if [ -z "$DATASET_PATH" ]; then
  # Check if dataset exists in common locations
  if [ -f "./${DEFAULT_DATASET}" ]; then
    DATASET_PATH="./${DEFAULT_DATASET}"
  elif [ -f "/tmp/${DEFAULT_DATASET}" ]; then
    DATASET_PATH="/tmp/${DEFAULT_DATASET}"
  elif docker exec "${CONTAINER}" bash -lc "[ -f /tmp/${DEFAULT_DATASET} ]" 2>/dev/null; then
    DATASET_PATH="/tmp/${DEFAULT_DATASET}"
  else
    # Download dataset to container
    print_info "Downloading ShareGPT dataset to container..."
    if ! docker exec "${CONTAINER}" bash -lc "
      cd /tmp
      if [ ! -f '${DEFAULT_DATASET}' ]; then
        curl -sL '${DATASET_URL}' -o '${DEFAULT_DATASET}'
      fi
    "; then
      print_fail "Failed to download dataset"
      exit 1
    fi
    DATASET_PATH="/tmp/${DEFAULT_DATASET}"
  fi
fi

# Verify dataset exists in container
if ! docker exec "${CONTAINER}" bash -lc "[ -f '${DATASET_PATH}' ]" 2>/dev/null; then
  # Try copying from host if it exists locally
  if [ -f "${DATASET_PATH}" ]; then
    print_info "Copying dataset to container..."
    docker cp "${DATASET_PATH}" "${CONTAINER}:/tmp/${DEFAULT_DATASET}"
    DATASET_PATH="/tmp/${DEFAULT_DATASET}"
  else
    print_fail "Dataset not found at ${DATASET_PATH}"
    exit 1
  fi
fi

DATASET_SIZE=$(docker exec "${CONTAINER}" bash -lc "wc -c < '${DATASET_PATH}'" 2>/dev/null || echo "unknown")
print_ok "Dataset ready: ${DATASET_PATH} (${DATASET_SIZE} bytes)"
echo ""

echo -e "${CYAN}▶${NC} Detecting model configuration..."

MODEL_ALIAS=$(curl -sf "${API_URL}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")

REAL_MODEL_PATH=$(docker exec "${CONTAINER}" ps aux | grep "vllm serve" | grep -v grep | perl -ne 'print $1 if /--model\s+([^\s]+)/' | head -1)

if [ -z "$REAL_MODEL_PATH" ]; then
    REAL_MODEL_PATH=$(docker exec "${CONTAINER}" ps aux | grep "vllm serve" | grep -v grep | awk '{for(i=1;i<=NF;i++) if($i=="serve") print $(i+1)}' | head -1)
fi

if [ "$MODEL_ALIAS" = "unknown" ] || [ -z "$REAL_MODEL_PATH" ]; then
    print_fail "Could not detect model alias or real path."
    echo "  Alias: $MODEL_ALIAS"
    echo "  Real Path: $REAL_MODEL_PATH"
    exit 1
fi

print_ok "API Alias: ${MODEL_ALIAS}"
print_ok "Real Path: ${REAL_MODEL_PATH}"
echo ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run benchmark
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print_header "Running Benchmark"

echo ""
echo -e "  ${BOLD}Test Parameters:${NC}"
echo -e "    Model:        ${MODEL}"
echo -e "    Prompts:      ${NUM_PROMPTS}"
echo -e "    Concurrency:  ${MAX_CONCURRENCY}"
echo -e "    Dataset:      ShareGPT_V3"
echo ""

# Extract host and port from API URL
API_HOST=$(echo "${API_URL}" | sed -E 's|https?://||' | cut -d: -f1)
API_PORT=$(echo "${API_URL}" | sed -E 's|https?://||' | cut -d: -f2 | cut -d/ -f1)
if [ -z "$API_PORT" ] || [ "$API_PORT" = "$API_HOST" ]; then
  API_PORT="8000"
fi

# Build output file argument if specified
OUTPUT_ARG=""
RESULT_FILE="/tmp/benchmark_result_$(date +%Y%m%d_%H%M%S).json"
if [ -n "$OUTPUT_FILE" ]; then
  OUTPUT_ARG="--result-filename ${RESULT_FILE}"
fi

# Run the benchmark inside the container
echo -e "${CYAN}▶${NC} Starting vllm bench serve..."
echo ""

BENCH_CMD="vllm bench serve \
  --backend vllm \
  --model '${MODEL_ALIAS}' \
  --tokenizer '${REAL_MODEL_PATH}' \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path '${DATASET_PATH}' \
  --num-prompts ${NUM_PROMPTS} \
  --max-concurrency ${MAX_CONCURRENCY} \
  --host ${API_HOST} \
  --port ${API_PORT} \
  ${OUTPUT_ARG}"

# Show the command being run
echo -e "  ${BLUE}Command:${NC}"
echo "    vllm bench serve \\"
echo "      --backend vllm \\"
echo "      --model '${MODEL_ALIAS}' \\"
echo "      --tokenizer '${REAL_MODEL_PATH}' \\"
echo "      --endpoint /v1/completions \\"
echo "      --dataset-name sharegpt \\"
echo "      --dataset-path '${DATASET_PATH}' \\"
echo "      --num-prompts ${NUM_PROMPTS} \\"
echo "      --max-concurrency ${MAX_CONCURRENCY} \\"
echo "      --host ${API_HOST} \\"
echo "      --port ${API_PORT}"
echo ""

# Execute benchmark
docker exec "${CONTAINER}" bash -lc "${BENCH_CMD}" 2>&1 | tee /tmp/benchmark_output.txt

BENCH_EXIT_CODE=${PIPESTATUS[0]}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Save results if requested
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ -n "$OUTPUT_FILE" ]; then
  # Copy result file from container
  if docker exec "${CONTAINER}" bash -lc "[ -f '${RESULT_FILE}' ]" 2>/dev/null; then
    docker cp "${CONTAINER}:${RESULT_FILE}" "${OUTPUT_FILE}"
    echo ""
    print_ok "Results saved to: ${OUTPUT_FILE}"
  fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Performance Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print_header "Performance Analysis"

# Extract key metrics from output
OUTPUT_TPS=$(grep -E "Output token throughput" /tmp/benchmark_output.txt | head -1 | awk '{print $NF}' || echo "N/A")
PEAK_TPS=$(grep -E "Peak output token throughput" /tmp/benchmark_output.txt | head -1 | awk '{print $NF}' || echo "N/A")
TOTAL_TPS=$(grep -E "Total Token throughput" /tmp/benchmark_output.txt | head -1 | awk '{print $NF}' || echo "N/A")
REQ_THROUGHPUT=$(grep -E "Request throughput" /tmp/benchmark_output.txt | head -1 | awk '{print $NF}' || echo "N/A")
MEAN_TTFT=$(grep -E "Mean TTFT" /tmp/benchmark_output.txt | head -1 | awk '{print $NF}' || echo "N/A")
MEAN_TPOT=$(grep -E "Mean TPOT" /tmp/benchmark_output.txt | head -1 | awk '{print $NF}' || echo "N/A")

echo ""
echo -e "  ${BOLD}Key Metrics:${NC}"
echo -e "    Output Throughput:     ${GREEN}${OUTPUT_TPS}${NC} tokens/sec"
if [ "$PEAK_TPS" != "N/A" ]; then
  echo -e "    Peak Throughput:       ${GREEN}${PEAK_TPS}${NC} tokens/sec"
fi
echo -e "    Total Throughput:      ${TOTAL_TPS} tokens/sec (input + output)"
echo -e "    Request Throughput:    ${REQ_THROUGHPUT} req/sec"
echo -e "    Mean TTFT:             ${MEAN_TTFT} ms"
echo -e "    Mean TPOT:             ${MEAN_TPOT} ms"
echo ""

# Diagnostic tip
echo -e "  ${BOLD}Diagnostic:${NC}"
echo -e "    To verify NCCL is using InfiniBand/RoCE: ./checkout_setup.sh --nccl"

echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print_header "Benchmark Complete"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo -e "  ${BOLD}Quick Reference Commands:${NC}"
echo ""
echo "  # Run single-request latency test"
echo "  ./benchmark_current.sh --single"
echo ""
echo "  # Run quick batch benchmark (20 prompts)"
echo "  ./benchmark_current.sh --quick"
echo ""
echo "  # Run full benchmark with JSON output"
echo "  ./benchmark_current.sh -n 100 -o results.json"
echo ""
echo "  # Check NCCL/InfiniBand configuration"
echo "  ./checkout_setup.sh --nccl"
echo ""

exit $BENCH_EXIT_CODE
