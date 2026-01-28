#!/bin/bash
# Quick benchmark runner for Qwen3-Omni API Server
# This script should be run from the vllm-omni root directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_OMNI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$VLLM_OMNI_ROOT" || { echo "Error: Failed to navigate to vllm-omni directory"; exit 1; }

cd "$SCRIPT_DIR"

# Default values
MODE=${1:-benchmark}
PROMPTS_FILE=${2:-../../build_dataset/top100.txt}
NUM_PROMPTS=${3:-100}
CONCURRENCY_LEVELS=${4:-"1 4 8"}
QPS_LEVELS=${5:-"0.1 0.2 0.3 0.4"}
OUTPUT_DIR=${6:-benchmark_results_$(date +%Y%m%d_%H%M%S)}

echo "════════════════════════════════════════════════════════════"
echo "Qwen3-Omni API Server Benchmark"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Script directory:     $SCRIPT_DIR"
echo "Working directory:    $(pwd)"
echo "Mode:                 $MODE"
echo "Prompts file:         $PROMPTS_FILE"
echo "Number of prompts:    $NUM_PROMPTS"
echo "Concurrency levels:   $CONCURRENCY_LEVELS"
echo "QPS levels:           $QPS_LEVELS"
echo "Output directory:     $OUTPUT_DIR"
echo ""

if [ "$MODE" = "server" ]; then
    echo "Starting API Server on http://0.0.0.0:8000"
    echo ""
    python "$SCRIPT_DIR/qwen3_omni_api_server.py" \
        --mode server \
        --host 0.0.0.0 \
        --port 8000

elif [ "$MODE" = "benchmark" ]; then
    # Check if prompts file exists
    if [ ! -f "$PROMPTS_FILE" ]; then
        echo "Error: Prompts file not found: $PROMPTS_FILE"
        exit 1
    fi
    
    echo "Starting benchmark..."
    echo ""
    
    python "$SCRIPT_DIR/qwen3_omni_api_server.py" \
        --mode benchmark \
        --prompts_file "$PROMPTS_FILE" \
        --num_prompts $NUM_PROMPTS \
        --concurrency_levels $CONCURRENCY_LEVELS \
        --qps_levels $QPS_LEVELS \
        --output_dir "$OUTPUT_DIR"
    
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "Benchmark completed!"
    echo "Results saved to: $OUTPUT_DIR/benchmark_results.json"
    echo "════════════════════════════════════════════════════════════"

else
    echo "Usage: $0 [server|benchmark] [prompts_file] [num_prompts] [concurrency_levels] [qps_levels] [output_dir]"
    echo ""
    echo "Examples:"
    echo "  $0 server                          # Start API server"
    echo "  $0 benchmark                       # Run benchmark with default settings"
    echo "  $0 benchmark ../../build_dataset/top100.txt 100 \"1 4 8\" \"0.1 0.2 0.3 0.4\""
    exit 1
fi
