#!/usr/bin/env python3
"""
Fine-grained attention divergence analysis.

Compares cross-attention input (txt_modulated) and output (txt_attn_output)
to identify exactly where computation diverges within attention.

Usage:
    python scripts/debug_attn_divergence.py <log_file> 
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AttentionStats:
    """Container for attention computation stats."""
    
    branch: str
    block_idx: int
    stage: str  # "input" or "output"
    mean: float = 0.0
    std: float = 0.0
    abs_mean: float = 0.0
    abs_max: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"block_{self.block_idx:02d} {self.stage:6} | "
            f"mean={self.mean:12.6f} std={self.std:12.6f} "
            f"abs_mean={self.abs_mean:12.6f} abs_max={self.abs_max:12.6f}"
        )


def parse_attn_log_line(line: str) -> Optional[AttentionStats]:
    """Parse attention debug log line."""
    if "[cfg-debug:" not in line or "txt_modulated" not in line and "txt_attn_output" not in line:
        return None
    
    # Extract branch
    branch_match = re.search(r"\[cfg-debug:(\w+)\]", line)
    if not branch_match:
        return None
    branch = branch_match.group(1)
    
    # Extract stage
    if "txt_modulated (encoder→attn input)" in line:
        stage = "INPUT"
    elif "txt_attn_output (attn→encoder output)" in line:
        stage = "OUTPUT"
    else:
        return None
    
    # Extract block_idx
    block_match = re.search(r"block_(\d+)", line)
    if not block_match:
        return None
    block_idx = int(block_match.group(1))
    
    # Extract stats
    mean_match = re.search(r"mean=([-\d.eE]+)", line)
    std_match = re.search(r"std=([-\d.eE]+)", line)
    abs_mean_match = re.search(r"abs_mean=([-\d.eE]+)", line)
    abs_max_match = re.search(r"abs_max=([-\d.eE]+)", line)
    
    mean = float(mean_match.group(1)) if mean_match else 0.0
    std = float(std_match.group(1)) if std_match else 0.0
    abs_mean = float(abs_mean_match.group(1)) if abs_mean_match else 0.0
    abs_max = float(abs_max_match.group(1)) if abs_max_match else 0.0
    
    return AttentionStats(
        branch=branch,
        block_idx=block_idx,
        stage=stage,
        mean=mean,
        std=std,
        abs_mean=abs_mean,
        abs_max=abs_max,
    )


def analyze_attention_divergence(log_file: Path) -> None:
    """Analyze where attention computation diverges between branches."""
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return
    
    with open(log_file) as f:
        lines = f.readlines()
    
    # Parse all attention stats
    stats_by_branch_and_stage = {}  # {(branch, block_idx, stage): AttentionStats}
    for line in lines:
        stats = parse_attn_log_line(line)
        if stats:
            key = (stats.branch, stats.block_idx, stats.stage)
            stats_by_branch_and_stage[key] = stats
    
    if not stats_by_branch_and_stage:
        print("❌ No attention debug logs found in file")
        return
    
    print("\n" + "=" * 140)
    print("ATTENTION COMPUTATION DIVERGENCE ANALYSIS")
    print("=" * 140)
    
    # Group by block and stage
    blocks_with_data = set()
    for branch, block_idx, stage in stats_by_branch_and_stage.keys():
        blocks_with_data.add((block_idx, stage))
    
    for block_idx, stage in sorted(blocks_with_data):
        original_key = ("original", block_idx, stage)
        parallel_key = ("parallel", block_idx, stage)
        
        original_stats = stats_by_branch_and_stage.get(original_key)
        parallel_stats = stats_by_branch_and_stage.get(parallel_key)
        
        if original_stats is None or parallel_stats is None:
            continue
        
        print(f"\n🔍 Block {block_idx:02d} {stage}:")
        print(f"  Original: {original_stats}")
        print(f"  Parallel: {parallel_stats}")
        
        # Compute differences
        diff_mean = abs(original_stats.mean - parallel_stats.mean)
        diff_std = abs(original_stats.std - parallel_stats.std)
        diff_abs_mean = abs(original_stats.abs_mean - parallel_stats.abs_mean)
        diff_abs_max = abs(original_stats.abs_max - parallel_stats.abs_max)
        
        print(f"  Δ mean     = {diff_mean:12.6e}")
        print(f"  Δ std      = {diff_std:12.6e}")
        print(f"  Δ abs_mean = {diff_abs_mean:12.6e}")
        print(f"  Δ abs_max  = {diff_abs_max:12.6e}")
        
        # Check if input and output both diverge
        if stage == "INPUT" and diff_abs_mean > 1e-6:
            print(f"  🚨 INPUT DIVERGENCE (should be identical!)")
        elif stage == "OUTPUT" and (diff_abs_mean > 1e-4 or diff_abs_max > 1e-4):
            print(f"  🚨 OUTPUT DIVERGENCE (attention computation affected)")
    
    print("\n" + "=" * 140)
    print("📊 SUMMARY:")
    print("=" * 140)
    
    # Check INPUT consistency
    input_diverged = False
    for block_idx in range(20, 25):
        original_key = ("original", block_idx, "INPUT")
        parallel_key = ("parallel", block_idx, "INPUT")
        
        if original_key in stats_by_branch_and_stage and parallel_key in stats_by_branch_and_stage:
            o = stats_by_branch_and_stage[original_key]
            p = stats_by_branch_and_stage[parallel_key]
            if abs(o.abs_mean - p.abs_mean) > 1e-6 or abs(o.abs_max - p.abs_max) > 1e-6:
                print(f"❌ Block {block_idx}: txt_modulated INPUT differs!")
                input_diverged = True
                break
    
    if not input_diverged:
        print("✅ txt_modulated (attention input) is IDENTICAL for all blocks")
        print("   → Problem is NOT in encoder or encoder projection")
        print("   → Problem IS inside attention computation")
    
    # Check OUTPUT consistency
    print("\n📈 txt_attn_output (attention output) divergence profile:")
    for block_idx in range(20, 25):
        output_key = ("original", block_idx, "OUTPUT")
        if output_key in stats_by_branch_and_stage:
            o_stats = stats_by_branch_and_stage.get(("original", block_idx, "OUTPUT"))
            p_stats = stats_by_branch_and_stage.get(("parallel", block_idx, "OUTPUT"))
            if o_stats and p_stats:
                diff = abs(o_stats.abs_mean - p_stats.abs_mean)
                if diff > 1e-4:
                    print(f"  Block {block_idx}: Δabs_mean = {diff:.6e} 🚨")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze attention computation divergence"
    )
    parser.add_argument("log_file", type=Path, help="Log file with attention debug output")
    args = parser.parse_args()
    analyze_attention_divergence(args.log_file)


if __name__ == "__main__":
    main()
