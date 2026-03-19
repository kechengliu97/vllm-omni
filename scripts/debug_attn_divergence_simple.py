#!/usr/bin/env python3
"""
Comprehensive CFG precision divergence analysis.

Parses three types of log lines:
  [cfg-debug:BRANCH] step_N LABEL: abs_mean=X, abs_max=Y   (step-level)
  [cfg-debug:BRANCH] step_N block_M_out_hs: abs_mean=X, abs_max=Y  (block output at step 0)
  [cfg-debug:BRANCH] block_N LABEL: abs_mean=X, abs_max=Y   (attention detail)
"""

import argparse
import re
from pathlib import Path

THRESH = 1e-6  # difference threshold for "OK"


def parse_line(line: str):
    """Parse a cfg-debug log line into a dict."""
    if "[cfg-debug:" not in line:
        return None

    branch_m = re.search(r"\[cfg-debug:(\w+)\]", line)
    if not branch_m:
        return None
    branch = branch_m.group(1)

    abs_mean_m = re.search(r"abs_mean=([-\d.eE+]+)", line)
    abs_max_m = re.search(r"abs_max=([-\d.eE+]+)", line)
    if not abs_mean_m or not abs_max_m:
        return None
    abs_mean = float(abs_mean_m.group(1))
    abs_max = float(abs_max_m.group(1))

    # Step-level line: [cfg-debug:X] step_N label: ...
    step_m = re.search(r"step_(\d+)\s+(.+?):\s*abs_mean", line)
    if step_m:
        return {
            "kind": "step",
            "branch": branch,
            "step": int(step_m.group(1)),
            "label": step_m.group(2).strip(),
            "abs_mean": abs_mean,
            "abs_max": abs_max,
        }

    # Block-level line: [cfg-debug:X] block_N label: ...
    block_m = re.search(r"block_(\d+)\s+(.+?):\s*abs_mean", line)
    if block_m:
        return {
            "kind": "block",
            "branch": branch,
            "block": int(block_m.group(1)),
            "label": block_m.group(2).strip(),
            "abs_mean": abs_mean,
            "abs_max": abs_max,
        }

    return None


def _diff_row(label, o, p):
    d_mean = abs(o["abs_mean"] - p["abs_mean"])
    d_max = abs(o["abs_max"] - p["abs_max"])
    ok = d_mean < THRESH and d_max < THRESH
    tag = "OK" if ok else "DIFF"
    return tag, d_mean, d_max


def analyze(log_file: Path):
    if not log_file.exists():
        print(f"ERROR: log file not found: {log_file}")
        return

    with open(log_file, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # ---- collect ----
    step_data = {}   # (branch, step, label) -> record
    block_data = {}  # (branch, block, label) -> record

    for line in lines:
        r = parse_line(line)
        if not r:
            continue
        if r["kind"] == "step":
            step_data[(r["branch"], r["step"], r["label"])] = r
        else:
            block_data[(r["branch"], r["block"], r["label"])] = r

    if not step_data and not block_data:
        print("ERROR: no cfg-debug logs found in the file.")
        return

    # ---- 1. Step-level: transformer input / output across all steps ----
    step_labels = sorted({k[2] for k in step_data if "block_" not in k[2]})
    max_step = max((k[1] for k in step_data), default=-1)

    if step_labels and max_step >= 0:
        print("\n" + "=" * 90)
        print("STEP-LEVEL: TRANSFORMER INPUT / OUTPUT  (every diffusion step)")
        print("=" * 90)
        for label in step_labels:
            print(f"\n  --- {label} ---")
            first_diff_step = None
            for s in range(max_step + 1):
                o = step_data.get(("original", s, label))
                p = step_data.get(("parallel", s, label))
                if o and p:
                    tag, dm, dx = _diff_row(label, o, p)
                    if tag == "DIFF" and first_diff_step is None:
                        first_diff_step = s
                    print(f"    [{tag}] step {s:3d}: d_mean={dm:.6e}  d_max={dx:.6e}"
                          f"  (orig_mean={o['abs_mean']:.6e}, par_mean={p['abs_mean']:.6e})")
            if first_diff_step is not None:
                print(f"  >>> DIVERGENCE first appears at step {first_diff_step}")
            else:
                print(f"  >>> All {max_step + 1} steps identical.")

    # ---- 2. Block-level: per-block output at step 0 ----
    block_out_labels = sorted(
        {k[2] for k in step_data if "block_" in k[2]},
        key=lambda x: int(re.search(r"block_(\d+)", x).group(1)) if re.search(r"block_(\d+)", x) else 0,
    )

    if block_out_labels:
        print("\n" + "=" * 90)
        print("BLOCK-LEVEL: per-block output hidden_states at step 0")
        print("=" * 90)
        first_diff_block = None
        for label in block_out_labels:
            o = step_data.get(("original", 0, label))
            p = step_data.get(("parallel", 0, label))
            if o and p:
                tag, dm, dx = _diff_row(label, o, p)
                block_num = re.search(r"block_(\d+)", label)
                bname = f"block {block_num.group(1):>2s}" if block_num else label
                if tag == "DIFF" and first_diff_block is None:
                    first_diff_block = bname
                print(f"  [{tag}] {bname}: d_mean={dm:.6e}  d_max={dx:.6e}")
        if first_diff_block is not None:
            print(f"\n  >>> First block divergence: {first_diff_block}")
        else:
            print(f"\n  >>> All blocks identical at step 0.")

    # ---- 3. Block detail (sub-operation level, blocks 41-43) ----
    detail_blocks = sorted({k[1] for k in block_data})
    if detail_blocks:
        print("\n" + "=" * 90)
        print("BLOCK DETAIL: sub-operation divergence (blocks 41-43)")
        print("=" * 90)
        for b in detail_blocks:
            print(f"\n  --- block {b} ---")
            labels_for_block = sorted(
                {k[2] for k in block_data if k[1] == b},
            )
            for label in labels_for_block:
                o = block_data.get(("original", b, label))
                p = block_data.get(("parallel", b, label))
                if o and p:
                    tag, dm, dx = _diff_row(label, o, p)
                    print(f"    [{tag}] {label}: d_mean={dm:.6e}  d_max={dx:.6e}"
                          f"  (orig={o['abs_mean']:.6e}, par={p['abs_mean']:.6e})")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CFG precision divergence analysis")
    parser.add_argument("log_file", type=Path)
    args = parser.parse_args()
    analyze(args.log_file)
