#!/usr/bin/env python3
"""Compare CFG debug block statistics between parallel and original runs.

Default inputs:
  /home/l30053556/cfg-fix/negative_kwargs/parallel/block_stats.json
  /home/l30053556/cfg-fix/negative_kwargs/original/block_stats.json

The script reports:
1. First divergent block index (based on configurable thresholds)
2. Top-N blocks with largest differences
3. Optional fail-fast exit code for CI usage
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PARALLEL = Path("/home/l30053556/cfg-fix/negative_kwargs/parallel/block_stats.json")
DEFAULT_ORIGINAL = Path("/home/l30053556/cfg-fix/negative_kwargs/original/block_stats.json")


@dataclass
class BlockDiff:
    block_idx: int
    hidden_abs_mean_diff: float
    hidden_abs_max_diff: float
    hidden_l2_diff: float
    enc_abs_mean_diff: float
    enc_abs_max_diff: float
    enc_l2_diff: float

    @property
    def score(self) -> float:
        # A simple aggregated score for ranking blocks.
        return (
            self.hidden_abs_mean_diff
            + self.hidden_abs_max_diff
            + self.hidden_l2_diff
            + self.enc_abs_mean_diff
            + self.enc_abs_max_diff
            + self.enc_l2_diff
        )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_rows(data: dict[str, Any], name: str) -> dict[int, dict[str, Any]]:
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid format in {name}: missing list field 'rows'")

    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = row.get("block_idx")
        if isinstance(idx, int):
            out[idx] = row
    return out


def _stat(row: dict[str, Any], group: str, key: str) -> float:
    group_obj = row.get(group, {})
    val = group_obj.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    return float("nan")


def _compute_diffs(parallel_rows: dict[int, dict[str, Any]], original_rows: dict[int, dict[str, Any]]) -> list[BlockDiff]:
    common = sorted(set(parallel_rows) & set(original_rows))
    diffs: list[BlockDiff] = []

    for idx in common:
        pr = parallel_rows[idx]
        orow = original_rows[idx]

        diff = BlockDiff(
            block_idx=idx,
            hidden_abs_mean_diff=abs(_stat(pr, "hidden_states", "abs_mean") - _stat(orow, "hidden_states", "abs_mean")),
            hidden_abs_max_diff=abs(_stat(pr, "hidden_states", "abs_max") - _stat(orow, "hidden_states", "abs_max")),
            hidden_l2_diff=abs(_stat(pr, "hidden_states", "l2") - _stat(orow, "hidden_states", "l2")),
            enc_abs_mean_diff=abs(
                _stat(pr, "encoder_hidden_states", "abs_mean") - _stat(orow, "encoder_hidden_states", "abs_mean")
            ),
            enc_abs_max_diff=abs(
                _stat(pr, "encoder_hidden_states", "abs_max") - _stat(orow, "encoder_hidden_states", "abs_max")
            ),
            enc_l2_diff=abs(_stat(pr, "encoder_hidden_states", "l2") - _stat(orow, "encoder_hidden_states", "l2")),
        )
        diffs.append(diff)

    return diffs


def _is_divergent(d: BlockDiff, abs_mean_thr: float, abs_max_thr: float, l2_thr: float) -> bool:
    return (
        d.hidden_abs_mean_diff > abs_mean_thr
        or d.enc_abs_mean_diff > abs_mean_thr
        or d.hidden_abs_max_diff > abs_max_thr
        or d.enc_abs_max_diff > abs_max_thr
        or d.hidden_l2_diff > l2_thr
        or d.enc_l2_diff > l2_thr
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare parallel/original CFG block stats and find first divergence.")
    parser.add_argument("--parallel", type=Path, default=DEFAULT_PARALLEL, help="Path to parallel block_stats.json")
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL, help="Path to original block_stats.json")
    parser.add_argument("--abs-mean-thr", type=float, default=1e-3, help="Threshold for abs_mean differences")
    parser.add_argument("--abs-max-thr", type=float, default=1e-2, help="Threshold for abs_max differences")
    parser.add_argument("--l2-thr", type=float, default=5e-2, help="Threshold for l2 differences")
    parser.add_argument("--topn", type=int, default=10, help="Number of top divergent blocks to print")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when divergence is detected",
    )
    args = parser.parse_args()

    parallel_data = _load_json(args.parallel)
    original_data = _load_json(args.original)

    parallel_rows = _to_rows(parallel_data, str(args.parallel))
    original_rows = _to_rows(original_data, str(args.original))

    if not parallel_rows:
        raise ValueError(f"No valid rows in {args.parallel}")
    if not original_rows:
        raise ValueError(f"No valid rows in {args.original}")

    if set(parallel_rows) != set(original_rows):
        only_p = sorted(set(parallel_rows) - set(original_rows))
        only_o = sorted(set(original_rows) - set(parallel_rows))
        print("[warn] block index mismatch detected")
        if only_p:
            print(f"  only in parallel: {only_p[:20]}")
        if only_o:
            print(f"  only in original: {only_o[:20]}")

    diffs = _compute_diffs(parallel_rows, original_rows)
    if not diffs:
        raise ValueError("No overlapping block indices found between files")

    first_div = None
    for d in sorted(diffs, key=lambda x: x.block_idx):
        if _is_divergent(d, args.abs_mean_thr, args.abs_max_thr, args.l2_thr):
            first_div = d
            break

    print("=== CFG Block Stats Comparison ===")
    print(f"parallel: {args.parallel}")
    print(f"original: {args.original}")
    print(
        "thresholds: "
        f"abs_mean>{args.abs_mean_thr}, "
        f"abs_max>{args.abs_max_thr}, "
        f"l2>{args.l2_thr}"
    )

    if first_div is None:
        print("first divergent block: none (under thresholds)")
    else:
        print(f"first divergent block: {first_div.block_idx}")
        print(
            "  diffs: "
            f"hidden(abs_mean={first_div.hidden_abs_mean_diff:.6g}, "
            f"abs_max={first_div.hidden_abs_max_diff:.6g}, l2={first_div.hidden_l2_diff:.6g}), "
            f"encoder(abs_mean={first_div.enc_abs_mean_diff:.6g}, "
            f"abs_max={first_div.enc_abs_max_diff:.6g}, l2={first_div.enc_l2_diff:.6g})"
        )

    print("\nTop divergent blocks by aggregated score:")
    ranked = sorted(diffs, key=lambda x: x.score, reverse=True)
    topn = max(1, args.topn)
    for d in ranked[:topn]:
        print(
            f"  block={d.block_idx:03d} "
            f"score={d.score:.6g} "
            f"h(abs_mean={d.hidden_abs_mean_diff:.6g}, abs_max={d.hidden_abs_max_diff:.6g}, l2={d.hidden_l2_diff:.6g}) "
            f"e(abs_mean={d.enc_abs_mean_diff:.6g}, abs_max={d.enc_abs_max_diff:.6g}, l2={d.enc_l2_diff:.6g})"
        )

    if args.strict and first_div is not None:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
