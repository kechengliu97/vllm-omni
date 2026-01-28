#!/usr/bin/env python3
"""
Quick usage examples for Qwen3-Omni API Server

Run this script to see how to use the API server.
"""

import sys

def print_usage():
    """Print usage examples"""
    
    examples = """
╔════════════════════════════════════════════════════════════════════════════╗
║          Qwen3-Omni API Server with Concurrent Benchmarking                ║
╚════════════════════════════════════════════════════════════════════════════╝

USAGE EXAMPLES:
==============

1. Start API Server (default on http://0.0.0.0:8000):
   python qwen3_omni_api_server.py --mode server

2. Start API Server on custom port:
   python qwen3_omni_api_server.py --mode server --port 8080

3. Run standalone benchmark with default settings (1, 4, 8 concurrency; 0.1, 0.2, 0.3, 0.4 QPS):
   python qwen3_omni_api_server.py --mode benchmark --prompts_file top100.txt

4. Run benchmark with custom concurrency levels:
   python qwen3_omni_api_server.py --mode benchmark --prompts_file top100.txt \\
     --concurrency_levels 1 2 4 8 --qps_levels 0.1 0.2 0.3 0.4

5. Run benchmark with limited prompts (e.g., first 20):
   python qwen3_omni_api_server.py --mode benchmark --prompts_file top100.txt \\
     --num_prompts 20 --output_dir benchmark_results

6. Quick benchmark (single concurrency and QPS):
   python qwen3_omni_api_server.py --mode benchmark --prompts_file top100.txt \\
     --concurrency_levels 1 --qps_levels 0.1 --num_prompts 5


API ENDPOINTS (Server Mode):
============================

Health Check:
  POST http://localhost:8000/health
  
  curl -X POST http://localhost:8000/health

Single TTS Request:
  POST http://localhost:8000/tts
  
  curl -X POST http://localhost:8000/tts \\
    -H "Content-Type: application/json" \\
    -d '{
      "prompt": "Hello, this is a test.",
      "speaker": "Ethan"
    }'

Start Benchmark (runs in background):
  POST http://localhost:8000/benchmark
  
  curl -X POST http://localhost:8000/benchmark \\
    -H "Content-Type: application/json" \\
    -d '{
      "prompts_file": "top100.txt",
      "concurrency_levels": [1, 4, 8],
      "qps_levels": [0.1, 0.2, 0.3, 0.4],
      "output_dir": "benchmark_results"
    }'

Get Benchmark Results:
  GET http://localhost:8000/benchmark/results
  
  curl http://localhost:8000/benchmark/results


SUPPORTED CONFIGURATIONS:
=========================

Concurrency Levels:  1, 4, 8
QPS Levels:         0.1, 0.2, 0.3, 0.4

This means:
- Concurrency 1:    Single request
- Concurrency 4:    4 concurrent requests
- Concurrency 8:    8 concurrent requests

- QPS 0.1:  1 request per 10 seconds
- QPS 0.2:  1 request per 5 seconds
- QPS 0.3:  1 request per ~3.3 seconds
- QPS 0.4:  1 request per 2.5 seconds


BENCHMARK OUTPUT:
=================

Results are saved to the output_dir as JSON:
  benchmark_results.json - Contains detailed metrics for each configuration

Format:
{
  "concurrency_1_qps_0.1": {
    "concurrency": 1,
    "qps": 0.1,
    "total_requests": 100,
    "successful_requests": 100,
    "failed_requests": 0,
    "min_latency": 0.5,
    "max_latency": 3.2,
    "mean_latency": 1.5,
    "median_latency": 1.4,
    "p95_latency": 2.1,
    "p99_latency": 3.0,
    "throughput": 0.95,
    "total_duration": 105.2,
    "success_rate": 1.0,
    "avg_thinker_tokens": 128,
    "avg_talker_tokens": 256,
    "avg_code2wav_tokens": 512
  },
  ...
}


QUICK START:
============

1. Prepare prompts file (one prompt per line):
   echo "Hello world" > prompts.txt
   echo "How are you?" >> prompts.txt

2. Run benchmark (first 2 prompts, quick test):
   python qwen3_omni_api_server.py --mode benchmark \\
     --prompts_file prompts.txt \\
     --num_prompts 2 \\
     --concurrency_levels 1 \\
     --qps_levels 0.1

3. Check results:
   cat benchmark_results/benchmark_results.json


PERFORMANCE METRICS EXPLAINED:
==============================

Latency (seconds):
  - min_latency:     Fastest single request
  - max_latency:     Slowest single request
  - mean_latency:    Average latency (main metric)
  - median_latency:  50th percentile
  - p95_latency:     95th percentile (important for SLA)
  - p99_latency:     99th percentile (worst case)

Throughput (req/s):
  - throughput:      Successful requests per second

Success Rate:
  - success_rate:    Percentage of successful requests (target: >95%)

Token Statistics:
  - avg_thinker_tokens:   Average tokens from thinker module
  - avg_talker_tokens:    Average tokens from talker module
  - avg_code2wav_tokens:  Average tokens from code2wav module


TROUBLESHOOTING:
================

1. Model loading fails:
   - Ensure model path is correct
   - Check disk space and memory
   - Use --model_path to specify custom model location

2. Out of memory:
   - Reduce concurrency level
   - Reduce number of prompts
   - Close other applications

3. QPS not accurate:
   - Check system load
   - Ensure no other heavy processes running
   - Increase number of prompts for more accurate measurement


ADVANCED USAGE:
===============

Run all combinations (3 concurrency × 4 QPS = 12 configurations):
  python qwen3_omni_api_server.py --mode benchmark \\
    --prompts_file top100.txt \\
    --concurrency_levels 1 4 8 \\
    --qps_levels 0.1 0.2 0.3 0.4 \\
    --output_dir results_$(date +%Y%m%d_%H%M%S)

Run server and use separate client:
  # Terminal 1: Start server
  python qwen3_omni_api_server.py --mode server --port 8000
  
  # Terminal 2: Send requests
  curl -X POST http://localhost:8000/tts \\
    -H "Content-Type: application/json" \\
    -d '{"prompt": "test"}'


NOTES:
======

- All code is contained in qwen3_omni_api_server.py
- No external configuration files needed
- Results are automatically saved as JSON
- Server supports concurrent requests with proper rate limiting
- Benchmark mode runs all configurations sequentially

"""
    
    print(examples)


if __name__ == "__main__":
    print_usage()
