#!/usr/bin/env python3
"""
Encoder debug analysis script.

Parses encoder debug logs from both CFG branches and identifies where encoder_hidden_states
first diverge, then analyzes the progression of divergence through blocks 20-24.

Usage:
    python scripts/debug_encoder_analysis.py <log_file> [--verbose]

Expected log format (from modified cfg_parallel.py and qwen_image_transformer.py):
    [cfg-debug:parallel] encoder_hidden_states input: shape=..., dtype=..., mean=..., std=..., abs_mean=..., abs_max=...
    [cfg-debug:parallel] encoder_hidden_states AFTER txt_norm: ...
    [cfg-debug:parallel] encoder_hidden_states AFTER txt_in: ...
    [cfg-debug:parallel] block_20 encoder_hidden_states AFTER block: ...
    ... (similar for blocks 21-24)
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class EncoderStats:
    """Container for encoder hidden states statistics."""

    branch: str
    stage: str  # "input", "AFTER txt_norm", "AFTER txt_in", "block_N"
    block_idx: Optional[int] = None
    shape: Optional[str] = None
    dtype: Optional[str] = None
    mean: float = 0.0
    std: float = 0.0
    abs_mean: float = 0.0
    abs_max: float = 0.0

    def __repr__(self) -> str:
        block_str = f"block_{self.block_idx:02d}" if self.block_idx is not None else self.stage
        return (
            f"[{self.branch:8}] {block_str:20} → "
            f"mean={self.mean:10.6f}, std={self.std:10.6f}, "
            f"abs_mean={self.abs_mean:10.6f}, abs_max={self.abs_max:10.6f}"
        )

    def diff_from(self, other: "EncoderStats") -> float:
        """Compute aggregated difference from another EncoderStats."""
        if other is None:
            return 0.0
        return (
            abs(self.mean - other.mean)
            + abs(self.std - other.std)
            + abs(self.abs_mean - other.abs_mean)
            + abs(self.abs_max - other.abs_max)
        )


def parse_log_line(line: str) -> Optional[EncoderStats]:
    """Parse a single encoder debug log line."""
    if "[cfg-debug:" not in line:
        return None

    # Extract branch
    branch_match = re.search(r"\[cfg-debug:(\w+)\]", line)
    if not branch_match:
        return None
    branch = branch_match.group(1)

    # Extract stage and block_idx
    stage = None
    block_idx = None

    if "encoder_hidden_states input:" in line:
        stage = "input"
    elif "AFTER txt_norm:" in line:
        stage = "AFTER txt_norm"
    elif "AFTER txt_in:" in line:
        stage = "AFTER txt_in"
    elif "block_" in line and "AFTER block:" in line:
        block_match = re.search(r"block_(\d+)", line)
        if block_match:
            block_idx = int(block_match.group(1))
            stage = "AFTER block"
    else:
        return None

    # Extract statistics
    shape_match = re.search(r"shape=([^,]+)", line)
    dtype_match = re.search(r"dtype=([^,]+)", line)
    mean_match = re.search(r"mean=([-\d.eE]+)", line)
    std_match = re.search(r"std=([-\d.eE]+)", line)
    abs_mean_match = re.search(r"abs_mean=([-\d.eE]+)", line)
    abs_max_match = re.search(r"abs_max=([-\d.eE]+)", line)

    shape = shape_match.group(1) if shape_match else None
    dtype = dtype_match.group(1) if dtype_match else None
    mean = float(mean_match.group(1)) if mean_match else 0.0
    std = float(std_match.group(1)) if std_match else 0.0
    abs_mean = float(abs_mean_match.group(1)) if abs_mean_match else 0.0
    abs_max = float(abs_max_match.group(1)) if abs_max_match else 0.0

    if stage is None:
        return None

    return EncoderStats(
        branch=branch,
        stage=stage,
        block_idx=block_idx,
        shape=shape,
        dtype=dtype,
        mean=mean,
        std=std,
        abs_mean=abs_mean,
        abs_max=abs_max,
    )


def analyze_logs(log_file: Path, verbose: bool = False) -> None:
    """Analyze encoder debug logs from CFG parallel and original branches."""
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return

    with open(log_file) as f:
        lines = f.readlines()

    # Parse all encoder stats
    stats_by_branch = {}
    for line in lines:
        stats = parse_log_line(line)
        if stats:
            if stats.branch not in stats_by_branch:
                stats_by_branch[stats.branch] = []
            stats_by_branch[stats.branch].append(stats)

    if not stats_by_branch:
        print("❌ No encoder debug logs found in file")
        return

    print("\n" + "=" * 120)
    print("ENCODER DEBUG ANALYSIS")
    print("=" * 120)

    # Display raw stats if verbose
    if verbose:
        print("\n📊 Raw Encoder Statistics:")
        for branch in sorted(stats_by_branch.keys()):
            print(f"\n  Branch: {branch}")
            for stat in stats_by_branch[branch]:
                print(f"    {stat}")

    # Analyze divergence
    print("\n🔍 Divergence Analysis:")

    stages = ["input", "AFTER txt_norm", "AFTER txt_in"]
    for stage in stages:
        print(f"\n  Stage: {stage}")
        branch_stats = {}
        for branch, stats_list in stats_by_branch.items():
            for stat in stats_list:
                if stat.stage == stage and stat.block_idx is None:
                    branch_stats[branch] = stat
                    break

        if len(branch_stats) < 2:
            print("    ⚠️  Only one branch has data for this stage")
            continue

        branches = sorted(branch_stats.keys())
        stat_a = branch_stats[branches[0]]
        stat_b = branch_stats[branches[1]] if len(branches) > 1 else None

        if stat_b is None:
            print("    (only one branch present)")
            continue

        # Compute differences
        diff_mean = abs(stat_a.mean - stat_b.mean)
        diff_std = abs(stat_a.std - stat_b.std)
        diff_abs_mean = abs(stat_a.abs_mean - stat_b.abs_mean)
        diff_abs_max = abs(stat_a.abs_max - stat_b.abs_max)

        print(f"    {branches[0]:10} vs {branches[1]:10}:")
        print(f"      Δ mean     = {diff_mean:.6e}")
        print(f"      Δ std      = {diff_std:.6e}")
        print(f"      Δ abs_mean = {diff_abs_mean:.6e}")
        print(f"      Δ abs_max  = {diff_abs_max:.6e}")

        if diff_abs_mean > 1e-4 or diff_abs_max > 1e-4:
            print(f"    🚨 DIVERGENCE DETECTED at {stage}")

    # Analyze blocks 20-24
    print("\n📈 Block-by-block Progression (blocks 20-24):")
    for branch in sorted(stats_by_branch.keys()):
        print(f"\n  Branch: {branch}")
        block_stats = {}
        for stat in stats_by_branch[branch]:
            if stat.block_idx is not None and 20 <= stat.block_idx <= 24:
                block_stats[stat.block_idx] = stat

        for block_idx in sorted(block_stats.keys()):
            stat = block_stats[block_idx]
            print(f"    {stat}")

    print("\n" + "=" * 120)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze encoder debug logs from CFG parallel branches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze logs from stdout/stderr redirected to file
  python scripts/debug_encoder_analysis.py logs.txt

  # Show verbose output
  python scripts/debug_encoder_analysis.py logs.txt --verbose
        """,
    )
    parser.add_argument("log_file", type=Path, help="Log file containing encoder debug output")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed statistics for all logs",
    )

    args = parser.parse_args()
    analyze_logs(args.log_file, verbose=args.verbose)


if __name__ == "__main__":
    main()
