#!/usr/bin/env python3
"""
Qwen3-Omni API Server with Concurrent Request Support and Performance Benchmarking

Supports:
- Single requests and concurrent requests (1, 4, 8)
- QPS control (0.1, 0.2, 0.3, 0.4)
- Real-time performance metrics
- Benchmark mode with detailed statistics
"""

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics
from enum import Enum

import soundfile as sf
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from tqdm import tqdm
from transformers import Qwen3OmniMoeProcessor

from qwen3_omni_moe_model import Qwen3OmniMoeForConditionalGenerationWithLogging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


# ============================================================================
# Data Models
# ============================================================================

class RequestStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class PerformanceStats:
    """Performance metrics for a single request"""
    request_id: int
    prompt: str
    start_time: float
    end_time: float
    latency: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    thinker_tokens: int = 0
    talker_tokens: int = 0
    code2wav_tokens: int = 0
    thinker_time_s: float = 0.0
    talker_time_s: float = 0.0
    code2wav_time_s: float = 0.0
    
    def __post_init__(self):
        self.latency = self.end_time - self.start_time


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results for a configuration"""
    concurrency: int
    qps: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    min_latency: float = 0.0
    max_latency: float = 0.0
    mean_latency: float = 0.0
    median_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    throughput: float = 0.0
    total_duration: float = 0.0
    success_rate: float = 0.0
    avg_thinker_tokens: float = 0.0
    avg_talker_tokens: float = 0.0
    avg_code2wav_tokens: float = 0.0


# ============================================================================
# Request Models for API
# ============================================================================

class TTSRequest(BaseModel):
    prompt: str
    speaker: str = "Ethan"


class BenchmarkRequest(BaseModel):
    prompts_file: str
    num_prompts: Optional[int] = None
    concurrency_levels: List[int] = [1, 4, 8]
    qps_levels: List[float] = [0.1, 0.2, 0.3, 0.4]
    output_dir: str = "benchmark_results"


class TTSResponse(BaseModel):
    request_id: int
    status: RequestStatus
    output_text: Optional[str] = None
    audio_path: Optional[str] = None
    latency: float = 0.0


class BenchmarkResponse(BaseModel):
    status: str
    results: Dict[str, Dict]
    summary: Dict


# ============================================================================
# QPS Limiter
# ============================================================================

class QPSLimiter:
    """Rate limiter for controlling QPS"""
    
    def __init__(self, qps: float):
        self.qps = qps
        self.min_interval = 1.0 / qps if qps > 0 else 0
        self.last_request_time = 0
        self.lock = asyncio.Lock()
    
    async def wait(self):
        """Wait before allowing the next request"""
        if self.qps <= 0:
            return
        
        async with self.lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()


class ConcurrencyLimiter:
    """Limits concurrent requests"""
    
    def __init__(self, max_concurrent: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def acquire(self):
        """Acquire a slot for a request"""
        await self.semaphore.acquire()
    
    def release(self):
        """Release a slot after request completion"""
        self.semaphore.release()


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages the Qwen3-Omni model instance"""
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.loaded = False
    
    def load_model(self):
        """Load model and processor"""
        if self.loaded:
            return
        
        logger.info(f"Loading model from {self.model_path}...")
        try:
            self.model = Qwen3OmniMoeForConditionalGenerationWithLogging.from_pretrained(
                self.model_path,
                dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
            self.loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.loaded = False
            logger.info("Model unloaded")
    
    async def generate_tts(
        self,
        prompt: str,
        speaker: str = "Ethan",
        output_dir: Optional[str] = None,
    ) -> Tuple[bool, str, Dict, Optional[str]]:
        """
        Generate TTS for the given prompt
        
        Returns:
            Tuple of (success, output_text, perf_stats, audio_path)
        """
        if not self.loaded:
            return False, "", {}, None
        
        perf_stats = {}
        audio_path = None
        
        try:
            # Prepare conversation
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                },
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            
            # Process inputs
            inputs = self.processor(
                text=text,
                audio=None,
                images=None,
                videos=None,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.model.device).to(self.model.dtype) if hasattr(v, 'to') else v 
                     for k, v in inputs.items()}
            
            # Generate output
            text_ids, audio = self.model.generate(
                **inputs,
                speaker=speaker,
                thinker_return_dict_in_generate=True,
            )
            
            # Decode output text
            output_text = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            
            # Collect performance stats
            if hasattr(self.model, '_perf_stats_last'):
                perf_stats = self.model._perf_stats_last.copy()
            
            # Save audio if requested
            if audio is not None and output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                audio_path = os.path.join(output_dir, f"output_{int(time.time() * 1000)}.wav")
                audio_data = audio.reshape(-1).detach().cpu().numpy()
                sf.write(audio_path, audio_data, samplerate=24000)
            
            return True, output_text, perf_stats, audio_path
        
        except Exception as e:
            logger.error(f"Error during TTS generation: {e}")
            return False, str(e), {}, None


# ============================================================================
# FastAPI Application
# ============================================================================

class QwenOmniServer:
    """Qwen3-Omni API Server with benchmarking capabilities"""
    
    def __init__(self, model_path: str = MODEL_PATH, host: str = "0.0.0.0", port: int = 8000):
        self.app = FastAPI(
            title="Qwen3-Omni TTS API Server",
            description="API server for Qwen3-Omni TTS with concurrent request and QPS control support",
            version="1.0.0"
        )
        self.model_manager = ModelManager(model_path)
        self.host = host
        self.port = port
        self.request_counter = 0
        self.all_stats: List[PerformanceStats] = []
        self.qps_limiters: Dict[float, QPSLimiter] = {}
        self.concurrency_limiters: Dict[int, ConcurrencyLimiter] = {}
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.on_event("startup")
        async def startup():
            """Load model on startup"""
            self.model_manager.load_model()
        
        @self.app.on_event("shutdown")
        async def shutdown():
            """Unload model on shutdown"""
            self.model_manager.unload_model()
        
        @self.app.post("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "model_loaded": self.model_manager.loaded,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/tts", response_model=TTSResponse)
        async def tts_endpoint(request: TTSRequest):
            """Single TTS request endpoint"""
            self.request_counter += 1
            request_id = self.request_counter
            
            start_time = time.time()
            success, output_text, perf_stats, audio_path = await self.model_manager.generate_tts(
                prompt=request.prompt,
                speaker=request.speaker,
            )
            end_time = time.time()
            latency = end_time - start_time
            
            if success:
                return TTSResponse(
                    request_id=request_id,
                    status=RequestStatus.SUCCESS,
                    output_text=output_text,
                    audio_path=audio_path,
                    latency=latency,
                )
            else:
                raise HTTPException(status_code=500, detail=output_text)
        
        @self.app.post("/benchmark", response_model=BenchmarkResponse)
        async def benchmark_endpoint(request: BenchmarkRequest, background_tasks: BackgroundTasks):
            """Benchmark endpoint with concurrent requests and QPS control"""
            
            # Load prompts
            if not os.path.exists(request.prompts_file):
                raise HTTPException(status_code=400, detail=f"Prompts file not found: {request.prompts_file}")
            
            prompts = self._load_prompts(request.prompts_file, request.num_prompts)
            if not prompts:
                raise HTTPException(status_code=400, detail="No prompts loaded")
            
            logger.info(f"Starting benchmark with {len(prompts)} prompts")
            logger.info(f"Concurrency levels: {request.concurrency_levels}")
            logger.info(f"QPS levels: {request.qps_levels}")
            
            # Run benchmark in background
            background_tasks.add_task(
                self._run_benchmark,
                prompts=prompts,
                concurrency_levels=request.concurrency_levels,
                qps_levels=request.qps_levels,
                output_dir=request.output_dir,
            )
            
            return BenchmarkResponse(
                status="benchmark_started",
                results={},
                summary={"message": "Benchmark started in background"}
            )
        
        @self.app.get("/benchmark/results")
        async def get_benchmark_results():
            """Get benchmark results"""
            return {
                "status": "success",
                "total_requests": len(self.all_stats),
                "results": [asdict(s) for s in self.all_stats],
                "summary": self._compute_summary(),
            }
    
    def _load_prompts(self, prompts_file: str, num_prompts: Optional[int] = None) -> List[str]:
        """Load prompts from file"""
        prompts = []
        with open(prompts_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if num_prompts and i >= num_prompts:
                    break
                line = line.strip()
                if line:
                    prompts.append(line)
        logger.info(f"Loaded {len(prompts)} prompts")
        return prompts
    
    async def _run_benchmark(
        self,
        prompts: List[str],
        concurrency_levels: List[int],
        qps_levels: List[float],
        output_dir: str,
    ):
        """Run full benchmark with all configurations"""
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        
        for concurrency in concurrency_levels:
            for qps in qps_levels:
                config_name = f"concurrency_{concurrency}_qps_{qps}"
                logger.info(f"Running: {config_name}")
                
                # Create limiters
                qps_limiter = QPSLimiter(qps)
                concurrency_limiter = ConcurrencyLimiter(concurrency)
                
                # Run requests
                tasks = []
                start_time = time.time()
                
                for idx, prompt in enumerate(prompts):
                    task = self._process_request(
                        request_id=len(self.all_stats) + idx,
                        prompt=prompt,
                        qps_limiter=qps_limiter,
                        concurrency_limiter=concurrency_limiter,
                        output_dir=os.path.join(output_dir, config_name),
                    )
                    tasks.append(task)
                
                # Execute all tasks
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                total_duration = end_time - start_time
                
                # Aggregate results
                benchmark_result = self._aggregate_results(
                    results=results,
                    concurrency=concurrency,
                    qps=qps,
                    total_duration=total_duration,
                )
                
                all_results[config_name] = asdict(benchmark_result)
                
                # Log summary
                self._log_benchmark_summary(benchmark_result)
                
                # Save intermediate results
                self._save_results(output_dir, all_results)
        
        logger.info("Benchmark completed")
    
    async def _process_request(
        self,
        request_id: int,
        prompt: str,
        qps_limiter: QPSLimiter,
        concurrency_limiter: ConcurrencyLimiter,
        output_dir: str,
    ) -> PerformanceStats:
        """Process a single request with QPS and concurrency control"""
        
        # Wait for QPS rate limit
        await qps_limiter.wait()
        
        # Acquire concurrency slot
        await concurrency_limiter.acquire()
        start_time = time.time()
        
        try:
            success, output_text, perf_stats, audio_path = await self.model_manager.generate_tts(
                prompt=prompt,
                output_dir=output_dir,
            )
            
            end_time = time.time()
            
            stats = PerformanceStats(
                request_id=request_id,
                prompt=prompt[:100],
                start_time=start_time,
                end_time=end_time,
                success=success,
                error_message=None if success else output_text,
                thinker_tokens=int(perf_stats.get('thinker_tokens', 0)),
                talker_tokens=int(perf_stats.get('talker_tokens', 0)),
                code2wav_tokens=int(perf_stats.get('code2wav_tokens', 0)),
                thinker_time_s=float(perf_stats.get('thinker_time_s', 0)),
                talker_time_s=float(perf_stats.get('talker_time_s', 0)),
                code2wav_time_s=float(perf_stats.get('code2wav_time_s', 0)),
            )
            
            self.all_stats.append(stats)
            return stats
        
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            end_time = time.time()
            
            stats = PerformanceStats(
                request_id=request_id,
                prompt=prompt[:100],
                start_time=start_time,
                end_time=end_time,
                success=False,
                error_message=str(e),
            )
            
            self.all_stats.append(stats)
            return stats
        
        finally:
            concurrency_limiter.release()
    
    def _aggregate_results(
        self,
        results: List,
        concurrency: int,
        qps: float,
        total_duration: float,
    ) -> BenchmarkResult:
        """Aggregate results for a configuration"""
        
        successful = [r for r in results if isinstance(r, PerformanceStats) and r.success]
        failed = [r for r in results if isinstance(r, PerformanceStats) and not r.success]
        
        latencies = [s.latency for s in successful]
        
        if successful:
            result = BenchmarkResult(
                concurrency=concurrency,
                qps=qps,
                total_requests=len(results),
                successful_requests=len(successful),
                failed_requests=len(failed),
                min_latency=min(latencies),
                max_latency=max(latencies),
                mean_latency=statistics.mean(latencies),
                median_latency=statistics.median(latencies),
                p95_latency=self._percentile(latencies, 95),
                p99_latency=self._percentile(latencies, 99),
                throughput=len(successful) / total_duration,
                total_duration=total_duration,
                success_rate=len(successful) / len(results) if results else 0,
                avg_thinker_tokens=statistics.mean([s.thinker_tokens for s in successful]) if successful else 0,
                avg_talker_tokens=statistics.mean([s.talker_tokens for s in successful]) if successful else 0,
                avg_code2wav_tokens=statistics.mean([s.code2wav_tokens for s in successful]) if successful else 0,
            )
        else:
            result = BenchmarkResult(
                concurrency=concurrency,
                qps=qps,
                total_requests=len(results),
                successful_requests=0,
                failed_requests=len(failed),
                success_rate=0,
            )
        
        return result
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if len(data) <= 1:
            return data[0] if data else 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _compute_summary(self) -> Dict:
        """Compute summary statistics"""
        if not self.all_stats:
            return {}
        
        successful = [s for s in self.all_stats if s.success]
        latencies = [s.latency for s in successful]
        
        if latencies:
            return {
                "total_requests": len(self.all_stats),
                "successful_requests": len(successful),
                "failed_requests": len(self.all_stats) - len(successful),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "mean_latency": statistics.mean(latencies),
                "median_latency": statistics.median(latencies),
                "p95_latency": self._percentile(latencies, 95),
                "p99_latency": self._percentile(latencies, 99),
            }
        return {}
    
    def _log_benchmark_summary(self, result: BenchmarkResult):
        """Log benchmark summary"""
        logger.info(
            f"Config: concurrency={result.concurrency}, qps={result.qps} | "
            f"Requests: {result.successful_requests}/{result.total_requests} | "
            f"Latency: {result.mean_latency:.3f}s (p95: {result.p95_latency:.3f}s) | "
            f"Throughput: {result.throughput:.2f} req/s"
        )
    
    def _save_results(self, output_dir: str, results: Dict):
        """Save benchmark results to file"""
        results_file = os.path.join(output_dir, "benchmark_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")
    
    def run(self, reload: bool = False):
        """Run the server"""
        logger.info(f"Starting server on {self.host}:{self.port}")
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            reload=reload,
        )


# ============================================================================
# Standalone Benchmark Mode
# ============================================================================

async def run_standalone_benchmark(
    model_path: str,
    prompts_file: str,
    concurrency_levels: List[int],
    qps_levels: List[float],
    output_dir: str,
    num_prompts: Optional[int] = None,
):
    """Run benchmark without starting the server"""
    
    logger.info("Running standalone benchmark mode")
    
    # Initialize model
    model_manager = ModelManager(model_path)
    model_manager.load_model()
    
    # Load prompts
    prompts = []
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_prompts and i >= num_prompts:
                break
            line = line.strip()
            if line:
                prompts.append(line)
    
    logger.info(f"Loaded {len(prompts)} prompts")
    
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}
    
    for concurrency in concurrency_levels:
        for qps in qps_levels:
            config_name = f"concurrency_{concurrency}_qps_{qps}"
            logger.info(f"Running: {config_name}")
            
            qps_limiter = QPSLimiter(qps)
            concurrency_limiter = ConcurrencyLimiter(concurrency)
            
            tasks = []
            start_time = time.time()
            
            request_id = 0
            for prompt in tqdm(prompts, desc=config_name):
                task = _process_request_standalone(
                    request_id=request_id,
                    prompt=prompt,
                    model_manager=model_manager,
                    qps_limiter=qps_limiter,
                    concurrency_limiter=concurrency_limiter,
                    output_dir=os.path.join(output_dir, config_name),
                )
                tasks.append(task)
                request_id += 1
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Aggregate results
            successful = [r for r in results if isinstance(r, PerformanceStats) and r.success]
            failed = [r for r in results if isinstance(r, PerformanceStats) and not r.success]
            
            latencies = [s.latency for s in successful]
            
            if successful:
                result = {
                    "concurrency": concurrency,
                    "qps": qps,
                    "total_requests": len(results),
                    "successful_requests": len(successful),
                    "failed_requests": len(failed),
                    "min_latency": min(latencies),
                    "max_latency": max(latencies),
                    "mean_latency": statistics.mean(latencies),
                    "median_latency": statistics.median(latencies),
                    "p95_latency": _percentile(latencies, 95),
                    "p99_latency": _percentile(latencies, 99),
                    "throughput": len(successful) / total_duration,
                    "total_duration": total_duration,
                    "success_rate": len(successful) / len(results) if results else 0,
                }
            else:
                result = {
                    "concurrency": concurrency,
                    "qps": qps,
                    "total_requests": len(results),
                    "successful_requests": 0,
                    "failed_requests": len(failed),
                    "success_rate": 0,
                    "total_duration": total_duration,
                }
            
            all_results[config_name] = result
            
            # Log and print summary
            logger.info(
                f"Config: concurrency={concurrency}, qps={qps} | "
                f"Requests: {result.get('successful_requests', 0)}/{result['total_requests']} | "
                f"Latency: {result.get('mean_latency', 0):.3f}s | "
                f"Throughput: {result.get('throughput', 0):.2f} req/s"
            )
    
    # Save all results
    results_file = os.path.join(output_dir, "benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"{'Config':<30} {'Requests':<15} {'Success Rate':<15} {'Mean Latency':<15} {'Throughput':<15}")
    print("-" * 100)
    
    for config_name, result in sorted(all_results.items()):
        success_rate = result.get('success_rate', 0) * 100
        mean_latency = result.get('mean_latency', 0)
        throughput = result.get('throughput', 0)
        requests = f"{result.get('successful_requests', 0)}/{result['total_requests']}"
        
        print(f"{config_name:<30} {requests:<15} {success_rate:>6.2f}%{'':<8} {mean_latency:>10.3f}s{'':<3} {throughput:>10.2f}{'':<3}")
    
    print("=" * 100 + "\n")
    
    # Unload model
    model_manager.unload_model()


async def _process_request_standalone(
    request_id: int,
    prompt: str,
    model_manager: ModelManager,
    qps_limiter: QPSLimiter,
    concurrency_limiter: ConcurrencyLimiter,
    output_dir: str,
) -> PerformanceStats:
    """Process a single request in standalone mode"""
    
    await qps_limiter.wait()
    await concurrency_limiter.acquire()
    start_time = time.time()
    
    try:
        success, output_text, perf_stats, audio_path = await model_manager.generate_tts(
            prompt=prompt,
            output_dir=output_dir,
        )
        
        end_time = time.time()
        
        return PerformanceStats(
            request_id=request_id,
            prompt=prompt[:100],
            start_time=start_time,
            end_time=end_time,
            success=success,
            error_message=None if success else output_text,
            thinker_tokens=int(perf_stats.get('thinker_tokens', 0)),
            talker_tokens=int(perf_stats.get('talker_tokens', 0)),
            code2wav_tokens=int(perf_stats.get('code2wav_tokens', 0)),
            thinker_time_s=float(perf_stats.get('thinker_time_s', 0)),
            talker_time_s=float(perf_stats.get('talker_time_s', 0)),
            code2wav_time_s=float(perf_stats.get('code2wav_time_s', 0)),
        )
    
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {e}")
        end_time = time.time()
        
        return PerformanceStats(
            request_id=request_id,
            prompt=prompt[:100],
            start_time=start_time,
            end_time=end_time,
            success=False,
            error_message=str(e),
        )
    
    finally:
        concurrency_limiter.release()


def _percentile(data: List[float], percentile: int) -> float:
    """Calculate percentile"""
    if len(data) <= 1:
        return data[0] if data else 0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-Omni TTS API Server with Benchmarking"
    )
    parser.add_argument(
        "--mode",
        choices=["server", "benchmark"],
        default="server",
        help="Run mode: 'server' for API server, 'benchmark' for standalone benchmark"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Path to the model"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="../../build_dataset/top100.txt",
        help="Path to prompts file (default: ../../build_dataset/top100.txt)"
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=100,
        help="Number of prompts to process (default: 100)"
    )
    parser.add_argument(
        "--concurrency_levels",
        type=int,
        nargs='+',
        default=[1, 4, 8],
        help="Concurrency levels for benchmark"
    )
    parser.add_argument(
        "--qps_levels",
        type=float,
        nargs='+',
        default=[0.1, 0.2, 0.3, 0.4],
        help="QPS levels for benchmark"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Output directory for benchmark results"
    )
    
    args = parser.parse_args()
    
    if args.mode == "server":
        # Run as API server
        server = QwenOmniServer(model_path=args.model_path, host=args.host, port=args.port)
        server.run()
    
    elif args.mode == "benchmark":
        # Run as standalone benchmark
        if not args.prompts_file:
            parser.error("--prompts_file is required for benchmark mode")
        
        asyncio.run(run_standalone_benchmark(
            model_path=args.model_path,
            prompts_file=args.prompts_file,
            concurrency_levels=args.concurrency_levels,
            qps_levels=args.qps_levels,
            output_dir=args.output_dir,
            num_prompts=args.num_prompts,
        ))


if __name__ == "__main__":
    main()
