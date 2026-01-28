#!/usr/bin/env python3
"""
Client for Qwen3-Omni API Server

Usage:
  python client.py --prompt "Hello" --speaker Ethan
  python client.py --health-check
"""

import argparse
import asyncio
import json
import time
from typing import Dict, Optional

import aiohttp
import requests


class QwenOmniClient:
    """Client for Qwen3-Omni API Server"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self) -> Dict:
        """Check server health"""
        try:
            response = requests.post(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def generate_tts(self, prompt: str, speaker: str = "Ethan") -> Dict:
        """Generate TTS for a single prompt"""
        try:
            data = {
                "prompt": prompt,
                "speaker": speaker,
            }
            response = requests.post(
                f"{self.base_url}/tts",
                json=data,
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def start_benchmark(
        self,
        prompts_file: str,
        concurrency_levels: list = None,
        qps_levels: list = None,
        output_dir: str = "benchmark_results",
        num_prompts: Optional[int] = None,
    ) -> Dict:
        """Start benchmark (runs in background)"""
        if concurrency_levels is None:
            concurrency_levels = [1, 4, 8]
        if qps_levels is None:
            qps_levels = [0.1, 0.2, 0.3, 0.4]
        
        try:
            data = {
                "prompts_file": prompts_file,
                "concurrency_levels": concurrency_levels,
                "qps_levels": qps_levels,
                "output_dir": output_dir,
            }
            if num_prompts:
                data["num_prompts"] = num_prompts
            
            response = requests.post(
                f"{self.base_url}/benchmark",
                json=data,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_benchmark_results(self) -> Dict:
        """Get benchmark results"""
        try:
            response = requests.get(
                f"{self.base_url}/benchmark/results",
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Client for Qwen3-Omni API Server")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--health-check", action="store_true", help="Check server health")
    parser.add_argument("--prompt", type=str, help="Text prompt for TTS")
    parser.add_argument("--speaker", type=str, default="Ethan", help="Speaker voice")
    parser.add_argument("--start-benchmark", action="store_true", help="Start benchmark")
    parser.add_argument("--prompts-file", type=str, help="Prompts file for benchmark")
    parser.add_argument("--concurrency-levels", type=int, nargs='+', default=[1, 4, 8], 
                       help="Concurrency levels")
    parser.add_argument("--qps-levels", type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4],
                       help="QPS levels")
    parser.add_argument("--num-prompts", type=int, help="Number of prompts")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--get-results", action="store_true", help="Get benchmark results")
    
    args = parser.parse_args()
    
    client = QwenOmniClient(base_url=args.base_url)
    
    if args.health_check:
        print("Checking server health...")
        result = client.health_check()
        print(json.dumps(result, indent=2))
    
    elif args.prompt:
        print(f"Generating TTS for: '{args.prompt}'")
        result = client.generate_tts(prompt=args.prompt, speaker=args.speaker)
        print(json.dumps(result, indent=2))
    
    elif args.start_benchmark:
        if not args.prompts_file:
            parser.error("--prompts-file is required for benchmark")
        
        print(f"Starting benchmark with {len(args.concurrency_levels)} concurrency levels "
              f"and {len(args.qps_levels)} QPS levels")
        result = client.start_benchmark(
            prompts_file=args.prompts_file,
            concurrency_levels=args.concurrency_levels,
            qps_levels=args.qps_levels,
            output_dir=args.output_dir,
            num_prompts=args.num_prompts,
        )
        print(json.dumps(result, indent=2))
    
    elif args.get_results:
        print("Fetching benchmark results...")
        result = client.get_benchmark_results()
        print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
