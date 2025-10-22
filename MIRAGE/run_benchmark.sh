#!/bin/bash

# MIRAGE Benchmark Runner Script

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

# Check if argument is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [test|mmlu|medqa|medmcqa|pubmedqa|bioasq|all]"
    echo ""
    echo "Examples:"
    echo "  $0 test          # Run setup test"
    echo "  $0 mmlu          # Run MMLU dataset only"
    echo "  $0 all           # Run all datasets"
    exit 1
fi

# Get the argument
COMMAND="$1"

case "$COMMAND" in
    "test")
        echo "================================"
        echo "Testing MIRAGE Setup"
        echo "================================"
        python MIRAGE/test_setup.py
        ;;
    "mmlu"|"medqa"|"medmcqa"|"pubmedqa"|"bioasq")
        echo "================================"
        echo "Running MIRAGE Benchmark: $COMMAND"
        echo "================================"
        python MIRAGE/run_benchmark_vllm.py --dataset "$COMMAND" --mode rag --k 32
        ;;
    "all")
        echo "================================"
        echo "Running MIRAGE Benchmark: All Datasets"
        echo "================================"
        python MIRAGE/run_benchmark_vllm.py --dataset all --mode rag --k 32
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        echo "Available commands: test, mmlu, medqa, medmcqa, pubmedqa, bioasq, all"
        exit 1
        ;;
esac