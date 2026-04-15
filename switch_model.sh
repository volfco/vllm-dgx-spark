#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# vLLM Model Switching Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Allows switching between different models on the vLLM cluster.
# Handles tensor parallelism, model downloading, and rsync to worker nodes.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration
if [ -f "${SCRIPT_DIR}/setup-env.sh" ]; then
  # Get WORKER_HOST and WORKER_USER from environment setup
  WORKER_HOST="${WORKER_HOST:-}"
  WORKER_USER="${WORKER_USER:-$(whoami)}"
  HF_CACHE="${HF_CACHE:-/raid/hf-cache}"
  HF_TOKEN="${HF_TOKEN:-}"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model Definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Model HuggingFace IDs
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

# Human-readable model descriptions
MODEL_NAMES=(
  "GPT-OSS-120B (120B params, MoE, native MXFP4 ~65GB, high quality)"
  "GPT-OSS-20B (21B params, MoE, ~16-20GB, fast)"
  "Qwen2.5-7B (7B params, ~7GB, very fast)"
  "Qwen2.5-14B (14B params, ~14GB, fast)"
  "Qwen2.5-32B (32B params, ~30GB, strong mid-size)"
  "Qwen2.5-72B (72B params, ~70GB, slow, high quality)"
  "Mistral-7B v0.3 (7B params, ~7GB, very fast)"
  "Mistral-Nemo-12B (12B params, ~12GB, 128k context)"
  "Mixtral-8x7B (47B total, 12B active, ~45GB, MoE, fast)"
  "Llama-3.1-8B (8B params, ~8GB, very fast)"
  "Llama-3.1-70B (70B params, ~65GB, high quality)"
  "Phi-4 (15B params, ~14-16GB, small but smart)"
  "Gemma2-27B (27B params, ~24-28GB, strong mid-size)"
  "Command-R-Plus (104B params, BF16 ~208GB, requires 2 Sparks)"
  "Llama-3.1-405B-FP4 (405B params, FP4 ~200GB, requires 2 Sparks)"
  "Llama-3.3-70B (70B params, BF16 ~141GB, requires 2 Sparks)"
)

# Tensor Parallelism (number of GPUs needed)
# Models that fit in a single DGX Spark's ~120GB VRAM use TP=1;
# larger models that must be split across two Sparks use TP=2.
MODEL_TP=(
  1    # gpt-oss-120b - native MXFP4 ~65GB
  1    # gpt-oss-20b - ~16-20GB
  1    # Qwen2.5-7B - ~7GB
  1    # Qwen2.5-14B - ~14GB
  1    # Qwen2.5-32B - ~30GB
  1    # Qwen2.5-72B - ~70GB
  1    # Mistral-7B - ~7GB
  1    # Mistral-Nemo-12B - ~12GB
  1    # Mixtral-8x7B - ~45GB
  1    # Llama-3.1-8B - ~8GB
  1    # Llama-3.1-70B - ~65GB
  1    # Phi-4 - ~14-16GB
  1    # Gemma2-27B - ~24-28GB
  2    # Command-R-Plus - BF16 ~208GB, needs 2 Sparks
  2    # Llama-3.1-405B-FP4 - ~200GB, needs 2 Sparks
  2    # Llama-3.3-70B - BF16 ~141GB, needs 2 Sparks
)

# Number of nodes required (1 for models that fit on a single Spark, 2 otherwise)
MODEL_NODES=(
  1    # gpt-oss-120b
  1    # gpt-oss-20b
  1    # Qwen2.5-7B
  1    # Qwen2.5-14B
  1    # Qwen2.5-32B
  1    # Qwen2.5-72B
  1    # Mistral-7B
  1    # Mistral-Nemo-12B
  1    # Mixtral-8x7B
  1    # Llama-3.1-8B
  1    # Llama-3.1-70B
  1    # Phi-4
  1    # Gemma2-27B
  2    # Command-R-Plus
  2    # Llama-3.1-405B-FP4
  2    # Llama-3.3-70B
)

# GPU Memory Utilization (0.90 default)
MODEL_GPU_MEM=(
  0.90  # gpt-oss-120b
  0.90  # gpt-oss-20b
  0.90  # Qwen2.5-7B
  0.90  # Qwen2.5-14B
  0.90  # Qwen2.5-32B
  0.90  # Qwen2.5-72B
  0.90  # Mistral-7B
  0.90  # Mistral-Nemo-12B
  0.90  # Mixtral-8x7B
  0.90  # Llama-3.1-8B
  0.90  # Llama-3.1-70B
  0.90  # Phi-4
  0.90  # Gemma2-27B
  0.90  # Command-R-Plus
  0.90  # Llama-3.1-405B-FP4
  0.90  # Llama-3.3-70B
)

# Max model length (context window)
MODEL_MAX_LEN=(
  8192   # gpt-oss-120b
  8192   # gpt-oss-20b
  32768  # Qwen2.5-7B - supports up to 131k, but limit for memory
  32768  # Qwen2.5-14B
  32768  # Qwen2.5-32B
  32768  # Qwen2.5-72B
  32768  # Mistral-7B v0.3
  131072 # Mistral-Nemo-12B - 128k context
  32768  # Mixtral-8x7B
  131072 # Llama-3.1-8B - 128k context
  131072 # Llama-3.1-70B - 128k context
  16384  # Phi-4
  8192   # Gemma2-27B
  32768  # Command-R-Plus - supports 128k, limit for KV cache memory
  16384  # Llama-3.1-405B-FP4 - very large, keep context modest
  32768  # Llama-3.3-70B - supports 128k, limit for KV cache memory
)

# Trust remote code flag
MODEL_TRUST_REMOTE=(
  false  # gpt-oss-120b
  false  # gpt-oss-20b
  false  # Qwen2.5-7B
  false  # Qwen2.5-14B
  false  # Qwen2.5-32B
  false  # Qwen2.5-72B
  false  # Mistral-7B
  false  # Mistral-Nemo-12B
  false  # Mixtral-8x7B
  false  # Llama-3.1-8B
  false  # Llama-3.1-70B
  true   # Phi-4 - requires trust_remote_code
  false  # Gemma2-27B
  false  # Command-R-Plus
  false  # Llama-3.1-405B-FP4
  false  # Llama-3.3-70B
)

# Requires HF token (gated models)
MODEL_NEEDS_TOKEN=(
  false  # gpt-oss-120b
  false  # gpt-oss-20b
  false  # Qwen2.5-7B
  false  # Qwen2.5-14B
  false  # Qwen2.5-32B
  false  # Qwen2.5-72B
  false  # Mistral-7B
  false  # Mistral-Nemo-12B
  false  # Mixtral-8x7B
  true   # Llama-3.1-8B - gated
  true   # Llama-3.1-70B - gated
  false  # Phi-4
  true   # Gemma2-27B - gated
  true   # Command-R-Plus - gated
  true   # Llama-3.1-405B-FP4 - gated (Meta Llama license)
  true   # Llama-3.3-70B - gated
)

# Enable expert parallel for MoE models
MODEL_EXPERT_PARALLEL=(
  true   # gpt-oss-120b - MoE
  true   # gpt-oss-20b - MoE
  false  # Qwen2.5-7B
  false  # Qwen2.5-14B
  false  # Qwen2.5-32B
  false  # Qwen2.5-72B
  false  # Mistral-7B
  false  # Mistral-Nemo-12B
  true   # Mixtral-8x7B - MoE
  false  # Llama-3.1-8B
  false  # Llama-3.1-70B
  false  # Phi-4
  false  # Gemma2-27B
  false  # Command-R-Plus
  false  # Llama-3.1-405B-FP4
  false  # Llama-3.3-70B
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

get_current_model() {
  # Try to get from running container
  if docker ps | grep -q ray-head; then
    LOADED_MODEL=$(docker exec ray-head bash -lc "curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"data\"][0][\"id\"])'" 2>/dev/null || echo "")
    if [ -n "${LOADED_MODEL}" ]; then
      echo "${LOADED_MODEL}"
      return
    fi
  fi

  # Fall back to reading start_cluster.sh
  if [ -f "${SCRIPT_DIR}/start_cluster.sh" ]; then
    grep '^MODEL=' "${SCRIPT_DIR}/start_cluster.sh" 2>/dev/null | head -1 | sed 's/MODEL="\${MODEL:-//' | sed 's/}"$//' || echo ""
  else
    echo ""
  fi
}

check_hf_token() {
  if [ -n "${HF_TOKEN:-}" ]; then
    return 0
  fi
  return 1
}

# Get the HF cache path for a model
get_model_cache_path() {
  local model="$1"
  local cache_name="models--$(echo "${model}" | sed 's|/|--|g')"
  echo "${HF_CACHE}/hub/${cache_name}"
}

# Check if model is downloaded locally
is_model_downloaded() {
  local model="$1"
  local cache_path=$(get_model_cache_path "${model}")

  if [ -d "${cache_path}/snapshots" ]; then
    # Check if there's at least one snapshot with model files
    local snapshot_count=$(find "${cache_path}/snapshots" -name "config.json" 2>/dev/null | wc -l)
    [ "${snapshot_count}" -gt 0 ]
  else
    return 1
  fi
}

# Download model using huggingface-cli
download_model() {
  local model="$1"
  local token_arg=""

  if [ -n "${HF_TOKEN:-}" ]; then
    token_arg="--token ${HF_TOKEN}"
  fi

  log "Downloading model: ${model}"
  log "  Destination: ${HF_CACHE}/hub/"
  log "  Excluding: original/*, metal/* (to save space)"

  # Use huggingface-cli for reliable downloads
  HF_HOME="${HF_CACHE}" huggingface-cli download "${model}" ${token_arg} --exclude "original/*" --exclude "metal/*" 2>&1 | tail -10
}

# Rsync model to worker node
rsync_model_to_worker() {
  local model="$1"
  local worker_host="${WORKER_HOST}"
  local worker_user="${WORKER_USER:-$(whoami)}"
  local worker_hf_cache="${WORKER_HF_CACHE:-${HF_CACHE}}"

  if [ -z "${worker_host}" ]; then
    log "  No WORKER_HOST configured, skipping rsync"
    return 0
  fi

  local cache_name="models--$(echo "${model}" | sed 's|/|--|g')"
  local local_path="${HF_CACHE}/hub/${cache_name}"

  if [ ! -d "${local_path}" ]; then
    log "  ERROR: Model not found at ${local_path}"
    return 1
  fi

  log "Syncing model to worker..."
  log "  Worker: ${worker_user}@${worker_host}"
  log "  Source: ${local_path}"
  log "  Dest:   ${worker_hf_cache}/hub/"

  # Ensure destination directory exists
  ssh "${worker_user}@${worker_host}" "mkdir -p ${worker_hf_cache}/hub" 2>/dev/null || true

  # Rsync with progress (exclude locks)
  if ! rsync -a --info=progress2 --human-readable \
    --no-perms --no-owner --no-group \
    --exclude='.locks' \
    --exclude='*.lock' \
    "${local_path}" \
    "${worker_user}@${worker_host}:${worker_hf_cache}/hub/"; then
    log "  WARNING: rsync failed, but continuing..."
    return 1
  fi

  log "  Model synced to worker"
  return 0
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Parse Arguments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SKIP_RESTART=false
LIST_ONLY=false
DOWNLOAD_ONLY=false
SKIP_DOWNLOAD=false
MODEL_NUMBER=""
DOWNLOAD_ALL_MODELS=false

usage() {
  cat << EOF
Usage: $0 [OPTIONS] [MODEL_NUMBER]

Switch between different models on the vLLM cluster.

Options:
  -l, --list          List available models without switching
  -s, --skip-restart  Update config only, don't restart cluster
  -d, --download-only Download model only, don't switch or restart
  --skip-download     Skip download step (use existing cached model)
  -h, --help          Show this help

Examples:
  $0                  # Interactive model selection
  $0 1                # Switch to model #1 (GPT-OSS-120B)
  $0 --list           # List all available models
  $0 -s 3             # Update config for model #3 without restarting
  $0 -d 5             # Download model #5 only (no restart)
  $0 all              # Download all models (batch download mode)

Environment:
  WORKER_HOST         Worker node hostname/IP for rsync
  WORKER_USER         SSH username for worker (default: current user)
  HF_CACHE            HuggingFace cache directory (default: /raid/hf-cache)
  HF_TOKEN            HuggingFace token for gated models

EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -l|--list)
      LIST_ONLY=true
      shift
      ;;
    -s|--skip-restart)
      SKIP_RESTART=true
      shift
      ;;
    -d|--download-only)
      DOWNLOAD_ONLY=true
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    [0-9]*)
      MODEL_NUMBER="$1"
      shift
      ;;
  all)
      DOWNLOAD_ALL_MODELS=true
      MODEL_NUMBER="all"
      shift
      ;;
  *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " vLLM Model Switcher"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show current model
CURRENT_MODEL=$(get_current_model)
if [ -n "${CURRENT_MODEL}" ]; then
  echo "Current model: ${CURRENT_MODEL}"
else
  echo "Current model: (not configured)"
fi
echo ""

# Display available models
echo "Available models:"
echo ""
echo "  Single-Node Models (TP=1):"
for i in "${!MODELS[@]}"; do
  if [ "${MODEL_NODES[$i]}" -eq 1 ]; then
    MARKER=""
    if [ "${MODELS[$i]}" = "${CURRENT_MODEL}" ]; then
      MARKER=" [CURRENT]"
    fi
    if [ "${MODEL_NEEDS_TOKEN[$i]}" = "true" ]; then
      MARKER="${MARKER} [HF TOKEN]"
    fi
    # Check if downloaded
    if is_model_downloaded "${MODELS[$i]}"; then
      MARKER="${MARKER} [CACHED]"
    fi
    printf "    %2d. %s%s\n" "$((i+1))" "${MODEL_NAMES[$i]}" "${MARKER}"
  fi
done

echo ""
echo "  Multi-Node Models (TP=2, requires 2 DGX Spark nodes):"
for i in "${!MODELS[@]}"; do
  if [ "${MODEL_NODES[$i]}" -eq 2 ]; then
    MARKER=""
    if [ "${MODELS[$i]}" = "${CURRENT_MODEL}" ]; then
      MARKER=" [CURRENT]"
    fi
    if [ "${MODEL_NEEDS_TOKEN[$i]}" = "true" ]; then
      MARKER="${MARKER} [HF TOKEN]"
    fi
    # Check if downloaded
    if is_model_downloaded "${MODELS[$i]}"; then
      MARKER="${MARKER} [CACHED]"
    fi
    printf "    %2d. %s%s\n" "$((i+1))" "${MODEL_NAMES[$i]}" "${MARKER}"
  fi
done
echo ""

# Exit if list only
if [ "${LIST_ONLY}" = "true" ]; then
  exit 0
fi

# Batch download mode
if [ "${DOWNLOAD_ALL_MODELS}" = "true" ]; then
  echo "Downloading all models (this may take a while)..."
  echo ""
  for i in "${!MODELS[@]}"; do
    IDX=$((i + 1))
    MODEL="${MODELS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Downloading [${IDX}/${#MODELS[@]}]: ${MODEL_NAME}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check if already downloaded
    if is_model_downloaded "${MODEL}"; then
      log "  Model already cached, skipping..."
      continue
    fi
    
    # Download model
    if download_model "${MODEL}"; then
      log "  ✅ Downloaded successfully"
    else
      log "  ⚠️  Download failed, but continuing..."
    fi
    echo ""
  done
  echo "Batch download complete!"
  exit 0
fi

# Get model selection
if [ -z "${MODEL_NUMBER}" ]; then
  read -p "Select model (1-${#MODELS[@]}), or 'q' to quit: " MODEL_NUMBER
fi

if [ "${MODEL_NUMBER}" = "q" ] || [ "${MODEL_NUMBER}" = "Q" ]; then
  echo "Cancelled."
  exit 0
fi

# Validate selection
if ! [[ "${MODEL_NUMBER}" =~ ^[0-9]+$ ]] || [ "${MODEL_NUMBER}" -lt 1 ] || [ "${MODEL_NUMBER}" -gt "${#MODELS[@]}" ]; then
  echo "ERROR: Invalid selection. Please enter a number between 1 and ${#MODELS[@]}."
  exit 1
fi

# Get model configuration
IDX=$((MODEL_NUMBER - 1))
NEW_MODEL="${MODELS[$IDX]}"
NEW_MODEL_NAME="${MODEL_NAMES[$IDX]}"
NEW_TP="${MODEL_TP[$IDX]}"
NEW_NODES="${MODEL_NODES[$IDX]}"
NEW_GPU_MEM="${MODEL_GPU_MEM[$IDX]}"
NEW_MAX_LEN="${MODEL_MAX_LEN[$IDX]}"
NEW_TRUST="${MODEL_TRUST_REMOTE[$IDX]}"
NEEDS_TOKEN="${MODEL_NEEDS_TOKEN[$IDX]}"
NEW_EXPERT_PARALLEL="${MODEL_EXPERT_PARALLEL[$IDX]}"

# Check if model needs HF token
if [ "${NEEDS_TOKEN}" = "true" ]; then
  if ! check_hf_token; then
    echo ""
    echo "WARNING: ${NEW_MODEL} requires a HuggingFace token."
    echo ""
    echo "Please set HF_TOKEN before continuing:"
    echo "  export HF_TOKEN=hf_your_token_here"
    echo ""
    read -p "Continue anyway? (y/N): " CONTINUE
    if [ "${CONTINUE}" != "y" ] && [ "${CONTINUE}" != "Y" ]; then
      echo "Cancelled."
      exit 1
    fi
  fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Switching to: ${NEW_MODEL_NAME}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration:"
echo "  Model:             ${NEW_MODEL}"
echo "  Tensor Parallel:   ${NEW_TP}"
echo "  Nodes Required:    ${NEW_NODES}"
echo "  GPU Memory Util:   ${NEW_GPU_MEM}"
echo "  Max Context:       ${NEW_MAX_LEN}"
[ "${NEW_TRUST}" = "true" ] && echo "  Trust Remote Code: yes"
[ "${NEW_EXPERT_PARALLEL}" = "true" ] && echo "  Expert Parallel:   yes (MoE)"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Download Model (if needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ "${SKIP_DOWNLOAD}" != "true" ]; then
  log "Step 1: Checking/Downloading model..."

  if is_model_downloaded "${NEW_MODEL}"; then
    log "  Model already cached locally"
  else
    log "  Model not found in cache, downloading..."
    if ! download_model "${NEW_MODEL}"; then
      echo "ERROR: Failed to download model"
      exit 1
    fi
    log "  Download complete"
  fi

  # Rsync to worker if multi-node
  if [ "${NEW_NODES}" -gt 1 ] && [ -n "${WORKER_HOST:-}" ]; then
    log ""
    log "Step 2: Syncing model to worker node..."
    rsync_model_to_worker "${NEW_MODEL}" || true
  fi
else
  log "Skipping download (--skip-download specified)"
fi

# Exit if download only
if [ "${DOWNLOAD_ONLY}" = "true" ]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " Download Complete"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "Model downloaded: ${NEW_MODEL}"
  echo "Cache location:   $(get_model_cache_path "${NEW_MODEL}")"
  echo ""
  echo "To switch to this model and start the cluster:"
  echo "  $0 --skip-download ${MODEL_NUMBER}"
  echo ""
  exit 0
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: Update Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log ""
log "Updating start_cluster.sh configuration..."

# Update the MODEL, TENSOR_PARALLEL, MAX_MODEL_LEN, GPU_MEMORY_UTIL in start_cluster.sh
START_SCRIPT="${SCRIPT_DIR}/start_cluster.sh"

if [ -f "${START_SCRIPT}" ]; then
  # Update MODEL
  sed -i "s|^MODEL=.*|MODEL=\"\${MODEL:-${NEW_MODEL}}\"|" "${START_SCRIPT}"

  # Update TENSOR_PARALLEL
  sed -i "s|^TENSOR_PARALLEL=.*|TENSOR_PARALLEL=\"\${TENSOR_PARALLEL:-${NEW_TP}}\"|" "${START_SCRIPT}"

  # Update MAX_MODEL_LEN
  sed -i "s|^MAX_MODEL_LEN=.*|MAX_MODEL_LEN=\"\${MAX_MODEL_LEN:-${NEW_MAX_LEN}}\"|" "${START_SCRIPT}"

  # Update GPU_MEMORY_UTIL
  sed -i "s|^GPU_MEMORY_UTIL=.*|GPU_MEMORY_UTIL=\"\${GPU_MEMORY_UTIL:-${NEW_GPU_MEM}}\"|" "${START_SCRIPT}"

  # Update ENABLE_EXPERT_PARALLEL (for MoE models)
  sed -i "s|^ENABLE_EXPERT_PARALLEL=.*|ENABLE_EXPERT_PARALLEL=\"\${ENABLE_EXPERT_PARALLEL:-${NEW_EXPERT_PARALLEL}}\"|" "${START_SCRIPT}"

  log "  Configuration updated in ${START_SCRIPT}"
else
  log "  WARNING: ${START_SCRIPT} not found"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: Restart Cluster (if not skipped)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ "${SKIP_RESTART}" = "true" ]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " Configuration Updated (restart skipped)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "To start the cluster with the new model:"
  echo "  ./start_cluster.sh"
  echo ""
  exit 0
fi

# Stop existing cluster
echo ""
log "Stopping existing cluster..."

# Stop head container
if docker ps | grep -q ray-head; then
  docker stop ray-head >/dev/null 2>&1 || true
  docker rm ray-head >/dev/null 2>&1 || true
  log "  Head container stopped"
else
  log "  No head container running"
fi

# Stop worker containers if WORKER_HOST is set
if [ -n "${WORKER_HOST:-}" ]; then
  log "  Stopping worker container on ${WORKER_HOST}..."
  ssh "${WORKER_USER:-$(whoami)}@${WORKER_HOST}" "docker stop ray-worker >/dev/null 2>&1; docker rm ray-worker >/dev/null 2>&1" 2>/dev/null || true
fi

# Start new cluster
echo ""
log "Starting cluster with new model..."
echo ""

# Set environment for start script
export MODEL="${NEW_MODEL}"
export TENSOR_PARALLEL="${NEW_TP}"
export MAX_MODEL_LEN="${NEW_MAX_LEN}"
export GPU_MEMORY_UTIL="${NEW_GPU_MEM}"
export ENABLE_EXPERT_PARALLEL="${NEW_EXPERT_PARALLEL}"
export SKIP_MODEL_DOWNLOAD=1  # We already downloaded

if [ "${NEW_NODES}" -gt 1 ]; then
  echo "  Starting multi-node cluster (this may take 3-5 minutes)..."
else
  echo "  Starting single-node cluster (this may take 2-3 minutes)..."
fi

"${SCRIPT_DIR}/start_cluster.sh" 2>&1 | tee /tmp/model_switch.log &
STARTUP_PID=$!

# Wait for API
echo ""
log "Waiting for API to become ready..."

MAX_WAIT=1800  # 30 minutes for large models (70B+ need more time)
ELAPSED=0
API_URL="http://127.0.0.1:8000"

while [ $ELAPSED -lt $MAX_WAIT ]; do
  if curl -sf "${API_URL}/health" >/dev/null 2>&1; then
    echo ""
    echo "  API is ready!"
    break
  fi
  sleep 10
  ELAPSED=$((ELAPSED + 10))
  if [ $((ELAPSED % 30)) -eq 0 ]; then
    # Check if startup process is still running
    if ! kill -0 $STARTUP_PID 2>/dev/null; then
      # Check if it succeeded
      if curl -sf "${API_URL}/health" >/dev/null 2>&1; then
        echo ""
        echo "  API is ready!"
        break
      fi
    fi
    echo "  Still waiting... ${ELAPSED}s elapsed"
  fi
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
  echo ""
  echo "  WARNING: API not ready after ${MAX_WAIT}s"
  echo "  Check logs: docker logs ray-head"
  echo "  Or: cat /tmp/model_switch.log"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Verify and Display Results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Model Switch Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Get loaded model info
LOADED_MODEL=$(curl -sf "${API_URL}/v1/models" 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "unknown")

echo "  Model:        ${LOADED_MODEL}"
echo "  API:          ${API_URL}"
echo "  Health:       ${API_URL}/health"
echo "  Time:         ${ELAPSED}s"
echo ""

# Quick test
echo "Testing inference..."
TEST_RESPONSE=$(curl -sf "${API_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"'"${NEW_MODEL}"'","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}' 2>/dev/null || echo "{}")

if echo "${TEST_RESPONSE}" | grep -q '"choices"'; then
  echo "  Inference test: PASSED"
else
  echo "  Inference test: FAILED (check logs)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
