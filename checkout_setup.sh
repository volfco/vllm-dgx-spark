#!/usr/bin/env bash

################################################################################
# vLLM DGX Spark Cluster Checkout & Diagnostic Script
#
# Combines hardware detection, NCCL verification, and full system diagnostics
# into a single tool for setup and troubleshooting.
#
# Usage:
#   ./checkout_setup.sh              # Interactive menu
#   ./checkout_setup.sh --full       # Full system diagnostic (generates log file)
#   ./checkout_setup.sh --nccl       # NCCL transport verification
#   ./checkout_setup.sh --infiniband # InfiniBand/RoCE hardware check
#   ./checkout_setup.sh --quick      # Quick health check
#   ./checkout_setup.sh --config     # Show recommended configuration
#
################################################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration (config.local.env overrides config.env) so we pick up RAY_PORT etc.
if [ -f "${SCRIPT_DIR}/config.local.env" ]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/config.local.env"
elif [ -f "${SCRIPT_DIR}/config.env" ]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/config.env"
fi

CONTAINER="${CONTAINER:-ray-head}"
WORKER_HOST="${WORKER_HOST:-${SECOND_DGX_HOST:-}}"
WORKER_USER="${WORKER_USER:-$(whoami)}"
RAY_PORT="${RAY_PORT:-6385}"

# Counters
ISSUES_FOUND=0
WARNINGS_FOUND=0

################################################################################
# Helper Functions
################################################################################

print_header() {
  echo ""
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${BOLD}$1${NC}"
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_section() {
  echo ""
  echo -e "${CYAN}▶ $1${NC}"
}

print_ok() {
  echo -e "  ${GREEN}✓${NC} $1"
}

print_warn() {
  echo -e "  ${YELLOW}⚠${NC} $1"
  WARNINGS_FOUND=$((WARNINGS_FOUND + 1))
}

print_fail() {
  echo -e "  ${RED}✗${NC} $1"
  ISSUES_FOUND=$((ISSUES_FOUND + 1))
}

print_info() {
  echo -e "  ${BLUE}ℹ${NC} $1"
}

# Run command either on host or in container
run_in_container() {
  docker exec "$CONTAINER" bash -c "$1" 2>/dev/null
}

has_container() {
  docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

has_cmd_in_container() {
  docker exec "$CONTAINER" bash -c "command -v $1" >/dev/null 2>&1
}

################################################################################
# InfiniBand/RoCE Hardware Check
################################################################################

check_infiniband() {
  print_header "InfiniBand/RoCE Hardware Check"
  echo -e "  ${BLUE}DGX Spark uses RoCE (RDMA over Converged Ethernet) at 200 Gb/s${NC}"

  # 1. Check for Mellanox/NVIDIA hardware
  print_section "1. ConnectX Hardware Detection"
  if lspci 2>/dev/null | grep -i -E "mellanox|nvidia.*connectx" > /dev/null; then
    print_ok "ConnectX hardware detected:"
    lspci | grep -i -E "mellanox|nvidia.*connectx" | while read -r line; do
      echo "      $line"
    done
  else
    print_fail "No Mellanox/NVIDIA ConnectX hardware found"
  fi

  # 2. Check IB tools
  print_section "2. InfiniBand Tools"
  local tools_ok=true
  if has_cmd ibdev2netdev; then
    print_ok "ibdev2netdev installed"
  else
    print_fail "ibdev2netdev NOT installed (apt-get install infiniband-diags)"
    tools_ok=false
  fi

  if has_cmd ibstat; then
    print_ok "ibstat installed"
  else
    print_warn "ibstat not installed"
  fi

  # 3. Check kernel modules
  print_section "3. RDMA Kernel Modules"
  local ib_modules=$(lsmod | grep -E '^ib_|^rdma|^mlx' || echo "")
  if [ -n "$ib_modules" ]; then
    print_ok "RDMA/InfiniBand modules loaded:"
    echo "$ib_modules" | head -10 | while read -r line; do
      echo "      $line"
    done
  else
    print_fail "No InfiniBand/RDMA kernel modules loaded"
  fi

  # 4. Check /dev/infiniband
  print_section "4. InfiniBand Device Files"
  if [ -d /dev/infiniband ]; then
    print_ok "InfiniBand devices present:"
    ls -la /dev/infiniband/ 2>/dev/null | tail -5 | while read -r line; do
      echo "      $line"
    done
  else
    print_fail "No InfiniBand devices at /dev/infiniband/"
  fi

  # 5. Detect RoCE interfaces
  print_section "5. RoCE Interface Detection"
  if has_cmd ibdev2netdev; then
    local ib_devices=$(ibdev2netdev 2>/dev/null || echo "")
    if [ -n "$ib_devices" ]; then
      echo "$ib_devices" | while read -r line; do
        if echo "$line" | grep -q "(Up)"; then
          echo -e "      ${GREEN}$line${NC}"
        else
          echo -e "      ${YELLOW}$line${NC}"
        fi
      done

      # Get active interface details
      local active_line=$(ibdev2netdev 2>/dev/null | grep '(Up)' | head -1)
      if [ -n "$active_line" ]; then
        local active_hca=$(echo "$active_line" | awk '{print $1}')
        local active_if=$(echo "$active_line" | awk '{print $5}' | tr -d '()')
        local active_ip=$(ip -o addr show "$active_if" 2>/dev/null | grep "inet " | awk '{print $4}' | cut -d'/' -f1 | head -1)
        local link_speed=$(cat "/sys/class/net/$active_if/speed" 2>/dev/null || echo "unknown")

        echo ""
        print_ok "Active RoCE Configuration:"
        echo "      HCA Device:  $active_hca"
        echo "      Interface:   $active_if"
        echo "      IP Address:  ${active_ip:-<not configured>}"
        if [ "$link_speed" != "unknown" ]; then
          echo "      Link Speed:  ${link_speed} Mbps ($((link_speed / 1000)) Gbps)"
        fi

        # Store for config recommendations
        DETECTED_HCA="$active_hca"
        DETECTED_IF="$active_if"
        DETECTED_IP="$active_ip"
      else
        print_warn "No active (Up) RoCE interfaces found"
      fi
    else
      print_fail "No RoCE devices detected"
    fi
  else
    print_warn "Cannot detect RoCE interfaces (ibdev2netdev not installed)"
  fi

  # 6. Check ibstat for port details
  print_section "6. InfiniBand Port Status"
  if has_cmd ibstat; then
    local ib_stat=$(ibstat 2>/dev/null || echo "")
    if [ -n "$ib_stat" ]; then
      local active_ports=$(echo "$ib_stat" | grep -c "State: Active" || echo "0")
      if [ "$active_ports" -gt 0 ]; then
        print_ok "$active_ports active InfiniBand/RoCE port(s)"
        echo "$ib_stat" | grep -E "^CA '|State:|Rate:" | head -15 | while read -r line; do
          if echo "$line" | grep -q "Active"; then
            echo -e "      ${GREEN}$line${NC}"
          else
            echo "      $line"
          fi
        done
      else
        print_warn "No active InfiniBand ports (State: Active)"
      fi
    fi
  else
    print_info "ibstat not available"
  fi
}

################################################################################
# NCCL Transport Verification
################################################################################

check_nccl() {
  print_header "NCCL Transport Verification"

  local check_container=false
  if has_container; then
    check_container=true
    echo -e "  Checking container: ${CYAN}${CONTAINER}${NC}"
  else
    echo -e "  Container ${CONTAINER} not running - checking host only"
  fi

  # 1. RDMA Libraries
  print_section "1. RDMA Libraries (required for NCCL IB)"

  local check_target="host"
  if [ "$check_container" = true ]; then
    check_target="container"
  fi

  if [ "$check_container" = true ]; then
    local libibverbs=$(run_in_container "ldconfig -p 2>/dev/null | grep libibverbs" || echo "")
    local librdmacm=$(run_in_container "ldconfig -p 2>/dev/null | grep librdmacm" || echo "")
  else
    local libibverbs=$(ldconfig -p 2>/dev/null | grep libibverbs || echo "")
    local librdmacm=$(ldconfig -p 2>/dev/null | grep librdmacm || echo "")
  fi

  if [ -n "$libibverbs" ]; then
    print_ok "libibverbs found ($check_target)"
  else
    print_fail "libibverbs NOT found - NCCL cannot use RDMA!"
    print_info "Install: apt-get install -y libibverbs1"
  fi

  if [ -n "$librdmacm" ]; then
    print_ok "librdmacm found ($check_target)"
  else
    print_fail "librdmacm NOT found - NCCL cannot use RDMA!"
    print_info "Install: apt-get install -y librdmacm1"
  fi

  # 2. NCCL Environment Variables
  print_section "2. NCCL Environment Variables"

  if [ "$check_container" = true ]; then
    local nccl_ib_disable=$(run_in_container "echo \${NCCL_IB_DISABLE:-}" || echo "")
    local nccl_ib_hca=$(run_in_container "echo \${NCCL_IB_HCA:-}" || echo "")
    local nccl_socket_ifname=$(run_in_container "echo \${NCCL_SOCKET_IFNAME:-}" || echo "")
    local nccl_net_gdr=$(run_in_container "echo \${NCCL_NET_GDR_LEVEL:-}" || echo "")
  else
    local nccl_ib_disable="${NCCL_IB_DISABLE:-}"
    local nccl_ib_hca="${NCCL_IB_HCA:-}"
    local nccl_socket_ifname="${NCCL_SOCKET_IFNAME:-}"
    local nccl_net_gdr="${NCCL_NET_GDR_LEVEL:-}"
  fi

  if [ "$nccl_ib_disable" = "0" ]; then
    print_ok "NCCL_IB_DISABLE=0 (IB enabled)"
  elif [ "$nccl_ib_disable" = "1" ]; then
    print_fail "NCCL_IB_DISABLE=1 (IB disabled!)"
  else
    print_warn "NCCL_IB_DISABLE not set (should be 0)"
  fi

  if [ -n "$nccl_ib_hca" ]; then
    print_ok "NCCL_IB_HCA=$nccl_ib_hca"
  else
    print_info "NCCL_IB_HCA not set"
  fi

  if [ -n "$nccl_socket_ifname" ]; then
    print_ok "NCCL_SOCKET_IFNAME=$nccl_socket_ifname"
  else
    print_info "NCCL_SOCKET_IFNAME not set"
  fi

  if [ -n "$nccl_net_gdr" ]; then
    print_ok "NCCL_NET_GDR_LEVEL=$nccl_net_gdr"
  else
    print_info "NCCL_NET_GDR_LEVEL not set"
  fi

  # 3. vLLM Log Analysis
  if [ "$check_container" = true ]; then
    print_section "3. vLLM NCCL Transport Analysis"

    if docker exec "$CONTAINER" test -f /var/log/vllm.log 2>/dev/null; then
      local ib_lines=$(docker exec "$CONTAINER" grep -E "NET/IB|IBext|GPU Direct RDMA" /var/log/vllm.log 2>/dev/null | head -5 || echo "")
      local socket_lines=$(docker exec "$CONTAINER" grep -E "NET/Socket|Using network Socket" /var/log/vllm.log 2>/dev/null | head -3 || echo "")

      if [ -n "$ib_lines" ]; then
        print_ok "vLLM is using InfiniBand/RoCE transport"
        echo "$ib_lines" | while read -r line; do
          local clean=$(echo "$line" | sed 's/.*NCCL INFO //' | cut -c1-70)
          echo -e "      ${GREEN}${clean}${NC}"
        done
      elif [ -n "$socket_lines" ]; then
        print_fail "vLLM is using Socket transport (NOT InfiniBand!)"
        print_info "NCCL is falling back to Ethernet - check RDMA libraries"
        echo "$socket_lines" | while read -r line; do
          local clean=$(echo "$line" | sed 's/.*NCCL INFO //' | cut -c1-70)
          echo -e "      ${RED}${clean}${NC}"
        done
      else
        print_info "No clear NCCL transport indication in logs"
        print_info "Look for 'NET/IB' (good) or 'NET/Socket' (bad)"
      fi

      # Check for errors
      local nccl_errors=$(docker exec "$CONTAINER" grep -iE "NCCL (error|failed|failure)" /var/log/vllm.log 2>/dev/null | grep -vi "INFO" | tail -3 || echo "")
      if [ -n "$nccl_errors" ]; then
        print_fail "NCCL errors found in logs:"
        echo "$nccl_errors" | while read -r line; do
          echo -e "      ${RED}$(echo "$line" | cut -c1-80)${NC}"
        done
      fi
    else
      print_info "No vLLM log found at /var/log/vllm.log"
    fi
  fi

  # 4. NCCL Network Test (if container running and GPUs available)
  if [ "$check_container" = true ]; then
    print_section "4. NCCL Network Detection Test"
    print_info "Running NCCL initialization test..."

    local nccl_test=$(docker exec "$CONTAINER" bash -c '
      export NCCL_DEBUG=INFO
      export NCCL_DEBUG_SUBSYS=INIT,NET
      python3 -c "
import torch
import os
os.environ[\"MASTER_ADDR\"] = \"127.0.0.1\"
os.environ[\"MASTER_PORT\"] = \"29500\"
os.environ[\"RANK\"] = \"0\"
os.environ[\"WORLD_SIZE\"] = \"1\"
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    t = torch.zeros(1).cuda()
    print(\"NCCL_TEST_OK\")
" 2>&1 | head -50' 2>/dev/null || echo "NCCL_TEST_FAILED")

    if echo "$nccl_test" | grep -q "NCCL_TEST_OK"; then
      if echo "$nccl_test" | grep -qi "ib\|infiniband\|rdma"; then
        print_ok "NCCL detected InfiniBand/RDMA support"
      elif echo "$nccl_test" | grep -qi "socket"; then
        print_warn "NCCL is using Socket transport"
      else
        print_info "NCCL test passed (transport unclear)"
      fi
    else
      print_warn "NCCL test could not run (GPUs may be in use)"
    fi
  fi
}

################################################################################
# Full System Diagnostic
################################################################################

run_full_diagnostic() {
  local timestamp=$(date +"%Y%m%d_%H%M%S")
  local log_file="${SCRIPT_DIR}/diagnostic_report_${timestamp}.log"

  print_header "Full System Diagnostic"
  echo -e "  Generating comprehensive report..."
  echo -e "  Log file: ${GREEN}${log_file}${NC}"
  echo ""

  # Start log file
  {
    echo "vLLM DGX Spark Cluster Diagnostic Report"
    echo "========================================="
    echo "Generated: $(date)"
    echo "Hostname:  $(hostname)"
    echo "User:      $(whoami)"
    echo "Worker:    ${WORKER_HOST:-<not configured>}"
    echo ""
  } > "$log_file"

  # Helper to log commands
  log_cmd() {
    local desc="$1"
    local cmd="$2"
    echo ">>> $desc" | tee -a "$log_file"
    echo "Command: $cmd" >> "$log_file"
    echo "---" >> "$log_file"
    if eval "$cmd" >> "$log_file" 2>&1; then
      echo -e "  ${GREEN}✓${NC} $desc"
    else
      echo -e "  ${YELLOW}⚠${NC} $desc (may have failed)"
    fi
    echo "" >> "$log_file"
  }

  # 1. System Information
  print_section "1. System Information"
  log_cmd "OS Release" "cat /etc/os-release"
  log_cmd "Kernel" "uname -a"
  log_cmd "Memory" "free -h"
  log_cmd "CPU" "lscpu | grep -E 'Model name|Socket|Core|Thread'"

  # 2. GPU Information
  print_section "2. GPU Configuration"
  log_cmd "GPU List" "nvidia-smi -L"
  log_cmd "GPU Status" "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv"
  log_cmd "GPU Topology" "nvidia-smi topo -m"
  log_cmd "Full nvidia-smi" "nvidia-smi"

  # 3. Docker/Container
  print_section "3. Docker Configuration"
  log_cmd "Docker Version" "docker --version"
  log_cmd "Running Containers" "docker ps"

  if has_container; then
    log_cmd "Container Env Vars" "docker exec $CONTAINER env | sort"
    log_cmd "Container GPU Access" "docker exec $CONTAINER nvidia-smi -L"
  fi

  # 4. Ray Cluster
  print_section "4. Ray Cluster Status"
  if has_container; then
    log_cmd "Ray Status" "docker exec $CONTAINER bash -c 'ray status --address=127.0.0.1:${RAY_PORT} 2>/dev/null || ray status'"
  fi

  # 5. Network Configuration
  print_section "5. Network Configuration"
  log_cmd "IP Addresses" "ip addr show"
  log_cmd "Routes" "ip route show"
  log_cmd "IB Devices" "ibdev2netdev 2>/dev/null || echo 'ibdev2netdev not available'"
  log_cmd "IB Status" "ibstat 2>/dev/null || echo 'ibstat not available'"

  # 6. NCCL Configuration
  print_section "6. NCCL Configuration"
  if has_container; then
    log_cmd "Container NCCL Vars" "docker exec $CONTAINER env | grep -E 'NCCL|UCX|GLOO' | sort"
    log_cmd "RDMA Libraries" "docker exec $CONTAINER ldconfig -p | grep -E 'ibverbs|rdmacm'"
  fi

  # 7. Remote Node (if configured)
  if [ -n "$WORKER_HOST" ]; then
    print_section "7. Remote Node ($WORKER_HOST)"
    log_cmd "Remote GPU List" "ssh ${WORKER_USER}@${WORKER_HOST} 'nvidia-smi -L' 2>/dev/null || echo 'SSH failed'"
    log_cmd "Remote IB Devices" "ssh ${WORKER_USER}@${WORKER_HOST} 'ibdev2netdev 2>/dev/null' || echo 'SSH failed'"
    log_cmd "Ping Test" "ping -c 3 $WORKER_HOST 2>/dev/null || echo 'Ping failed'"
  fi

  # 8. vLLM Logs
  print_section "8. vLLM Logs"
  if has_container && docker exec "$CONTAINER" test -f /var/log/vllm.log 2>/dev/null; then
    log_cmd "vLLM Log (last 100 lines)" "docker exec $CONTAINER tail -100 /var/log/vllm.log"
  fi

  # 9. Resource Usage
  print_section "9. System Resources"
  log_cmd "Disk Usage" "df -h"
  log_cmd "Top CPU Processes" "ps aux --sort=-%cpu | head -15"

  # Summary
  {
    echo ""
    echo "================================================================================"
    echo "  SUMMARY"
    echo "================================================================================"
    echo ""
    echo "Local GPUs:    $(nvidia-smi -L 2>/dev/null | wc -l)"
    if [ -n "$WORKER_HOST" ]; then
      echo "Remote GPUs:   $(ssh ${WORKER_USER}@${WORKER_HOST} 'nvidia-smi -L 2>/dev/null | wc -l' 2>/dev/null || echo 'unknown')"
    fi
    if has_container; then
      echo "Ray Nodes:     $(docker exec $CONTAINER bash -c "ray status --address=127.0.0.1:${RAY_PORT} 2>/dev/null | grep -c node_ || echo 0" 2>/dev/null)"
    fi
    echo "IB Ports Up:   $(ibstat 2>/dev/null | grep -c 'State: Active' || echo '0')"
    echo ""
  } | tee -a "$log_file"

  echo ""
  echo -e "${GREEN}Diagnostic complete!${NC}"
  echo -e "Full report saved to: ${GREEN}${log_file}${NC}"
}

################################################################################
# Quick Health Check
################################################################################

quick_check() {
  print_header "Quick Health Check"

  local all_ok=true

  # GPU Check
  print_section "GPUs"
  local gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
  if [ "$gpu_count" -gt 0 ]; then
    print_ok "$gpu_count GPU(s) detected"
  else
    print_fail "No GPUs detected"
    all_ok=false
  fi

  # Container Check
  print_section "Container"
  if has_container; then
    print_ok "Container '$CONTAINER' is running"

    # vLLM API Check
    if docker exec "$CONTAINER" curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
      print_ok "vLLM API is responding"
    else
      print_warn "vLLM API not responding on port 8000"
    fi
  else
    print_warn "Container '$CONTAINER' is not running"
  fi

  # Ray Check
  print_section "Ray Cluster"
  if has_container; then
    local ray_nodes=$(docker exec "$CONTAINER" bash -c "ray status --address=127.0.0.1:${RAY_PORT} 2>/dev/null | grep -c node_" 2>/dev/null || echo "0")
    if [ "$ray_nodes" -gt 0 ]; then
      print_ok "$ray_nodes Ray node(s) connected"
    else
      print_warn "No Ray nodes detected"
    fi
  fi

  # InfiniBand Check
  print_section "InfiniBand/RoCE"
  if has_cmd ibdev2netdev; then
    local ib_up=$(ibdev2netdev 2>/dev/null | grep -c "(Up)" || echo "0")
    if [ "$ib_up" -gt 0 ]; then
      print_ok "$ib_up RoCE interface(s) active"
    else
      print_warn "No active RoCE interfaces"
    fi
  else
    print_warn "ibdev2netdev not installed"
  fi

  # Worker Check
  print_section "Worker Node"
  if [ -n "$WORKER_HOST" ]; then
    if ssh -o BatchMode=yes -o ConnectTimeout=3 "${WORKER_USER}@${WORKER_HOST}" "echo ok" >/dev/null 2>&1; then
      print_ok "Worker $WORKER_HOST is reachable"
    else
      print_fail "Cannot reach worker $WORKER_HOST"
      all_ok=false
    fi
  else
    print_info "No WORKER_HOST configured"
  fi

  echo ""
  if [ "$all_ok" = true ] && [ "$WARNINGS_FOUND" -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
  elif [ "$ISSUES_FOUND" -eq 0 ]; then
    echo -e "${YELLOW}$WARNINGS_FOUND warning(s) - review above${NC}"
  else
    echo -e "${RED}$ISSUES_FOUND issue(s) found - review above${NC}"
  fi
}

################################################################################
# Show Recommended Configuration
################################################################################

show_config() {
  print_header "Recommended Configuration"

  # Detect RoCE interface
  local detected_if=""
  local detected_hca=""
  local detected_ip=""
  local all_hcas=""

  if has_cmd ibdev2netdev; then
    local active_line=$(ibdev2netdev 2>/dev/null | grep '(Up)' | head -1)
    if [ -n "$active_line" ]; then
      detected_hca=$(echo "$active_line" | awk '{print $1}')
      detected_if=$(echo "$active_line" | awk '{print $5}' | tr -d '()')
      detected_ip=$(ip -o addr show "$detected_if" 2>/dev/null | grep "inet " | awk '{print $4}' | cut -d'/' -f1 | head -1)
      all_hcas=$(ibdev2netdev 2>/dev/null | grep "(Up)" | awk '{print $1}' | sort | tr '\n' ',' | sed 's/,$//')
    fi
  fi

  if [ -z "$detected_if" ]; then
    echo ""
    echo -e "${RED}No active RoCE interface detected.${NC}"
    echo ""
    echo "Please ensure:"
    echo "  1. 200 Gb cable is connected between Spark nodes"
    echo "  2. Run: ibdev2netdev  (to check interface status)"
    echo "  3. Check link: ip link show"
    return 1
  fi

  echo ""
  echo -e "${GREEN}Detected RoCE Configuration:${NC}"
  echo "  Interface: $detected_if"
  echo "  HCA:       $detected_hca"
  echo "  IP:        ${detected_ip:-<not configured>}"
  echo ""

  echo -e "${CYAN}# Add these to your environment or start script:${NC}"
  echo ""
  echo -e "${GREEN}# RoCE/InfiniBand Interface Settings${NC}"
  echo "export NCCL_SOCKET_IFNAME=${detected_if}"
  echo "export GLOO_SOCKET_IFNAME=${detected_if}"
  echo "export TP_SOCKET_IFNAME=${detected_if}"
  echo "export UCX_NET_DEVICES=${detected_if}"
  echo "export OMPI_MCA_btl_tcp_if_include=${detected_if}"
  echo ""
  echo -e "${GREEN}# NCCL InfiniBand/RoCE Settings${NC}"
  echo "export NCCL_IB_DISABLE=0"
  echo "export NCCL_IB_HCA=${all_hcas:-$detected_hca}"
  echo "export NCCL_NET_GDR_LEVEL=5"
  echo ""
  echo -e "${GREEN}# Head Node IP (use this for HEAD_IP)${NC}"
  if [ -n "$detected_ip" ]; then
    echo "export HEAD_IP=${detected_ip}"
  else
    echo "# Run: ip addr show ${detected_if}"
    echo "# export HEAD_IP=<ip-from-above>"
  fi
  echo ""
  echo -e "${GREEN}# Debug Settings (optional)${NC}"
  echo "export NCCL_DEBUG=INFO"
  echo "export NCCL_DEBUG_SUBSYS=INIT,NET"
  echo ""
}

################################################################################
# Interactive Menu
################################################################################

show_menu() {
  print_header "vLLM DGX Spark Checkout & Diagnostic"
  echo ""
  echo "  Select an option:"
  echo ""
  echo "    1) Quick Health Check    - Fast status check of cluster"
  echo "    2) InfiniBand Check      - Hardware and RoCE detection"
  echo "    3) NCCL Verification     - Verify NCCL is using InfiniBand"
  echo "    4) Full Diagnostic       - Complete system report (log file)"
  echo "    5) Show Configuration    - Recommended environment variables"
  echo "    q) Quit"
  echo ""

  read -p "  Choice [1-5, q]: " choice

  case "$choice" in
    1) quick_check ;;
    2) check_infiniband ;;
    3) check_nccl ;;
    4) run_full_diagnostic ;;
    5) show_config ;;
    q|Q) echo "Exiting."; exit 0 ;;
    *) echo "Invalid choice"; exit 1 ;;
  esac
}

################################################################################
# Usage
################################################################################

usage() {
  cat << EOF
Usage: $0 [OPTION]

vLLM DGX Spark Cluster Checkout & Diagnostic Tool

Options:
  (no args)       Interactive menu
  --quick         Quick health check (GPUs, container, Ray, IB)
  --infiniband    InfiniBand/RoCE hardware detection
  --nccl          NCCL transport verification (check if using IB)
  --full          Full system diagnostic (generates log file)
  --config        Show recommended configuration/environment variables
  -h, --help      Show this help

Environment Variables:
  CONTAINER       Docker container name (default: ray-head)
  WORKER_HOST     Worker node hostname/IP for remote checks
  WORKER_USER     SSH username for worker (default: current user)

Examples:
  $0                        # Interactive menu
  $0 --quick                # Fast cluster health check
  $0 --nccl                 # Verify NCCL is using InfiniBand
  $0 --full                 # Generate complete diagnostic log

  WORKER_HOST=192.168.1.2 $0 --full   # Include remote node in diagnostic

EOF
  exit 0
}

################################################################################
# Main
################################################################################

case "${1:-}" in
  --quick)
    quick_check
    ;;
  --infiniband|--ib)
    check_infiniband
    ;;
  --nccl)
    check_nccl
    ;;
  --full)
    run_full_diagnostic
    ;;
  --config)
    show_config
    ;;
  -h|--help)
    usage
    ;;
  "")
    show_menu
    ;;
  *)
    echo "Unknown option: $1"
    usage
    ;;
esac

# Print summary if issues/warnings found
if [ "$ISSUES_FOUND" -gt 0 ] || [ "$WARNINGS_FOUND" -gt 0 ]; then
  echo ""
  print_header "Summary"
  if [ "$ISSUES_FOUND" -gt 0 ]; then
    echo -e "  ${RED}$ISSUES_FOUND critical issue(s) found${NC}"
  fi
  if [ "$WARNINGS_FOUND" -gt 0 ]; then
    echo -e "  ${YELLOW}$WARNINGS_FOUND warning(s) found${NC}"
  fi
  echo ""
fi

exit $ISSUES_FOUND
