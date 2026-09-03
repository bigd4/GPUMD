#!/usr/bin/env python3
"""Compare selected numeric outputs from TorchScript and AOTI runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("torchscript_dir", type=Path)
    parser.add_argument("aoti_dir", type=Path)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("reproduction.json"))
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    args = parser.parse_args()
    comparisons = []
    passed = True
    for name in args.files:
        left_path = args.torchscript_dir / name
        right_path = args.aoti_dir / name
        left = np.loadtxt(left_path)
        right = np.loadtxt(right_path)
        same_shape = left.shape == right.shape
        close = bool(same_shape and np.allclose(left, right, rtol=args.rtol, atol=args.atol))
        passed = passed and close
        delta = np.abs(left - right) if same_shape else np.asarray([np.inf])
        comparisons.append(
            {
                "file": name,
                "shape": list(left.shape),
                "same_shape": same_shape,
                "max_abs_error": float(np.max(delta)) if delta.size else 0.0,
                "torchscript_sha256": digest(left_path),
                "aoti_sha256": digest(right_path),
                "passed": close,
            }
        )
    report = {
        "schema": "pygas-reproduction-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "torchscript_dir": str(args.torchscript_dir.resolve()),
        "aoti_dir": str(args.aoti_dir.resolve()),
        "rtol": args.rtol,
        "atol": args.atol,
        "passed": passed,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = args.output.with_suffix(".md")
    rows = ["# Backend reproduction", "", f"Overall: **{'PASS' if passed else 'FAIL'}**", "", "| File | Shape | Max abs error | Result |", "| --- | --- | ---: | --- |"]
    rows.extend(
        f"| `{item['file']}` | `{item['shape']}` | {item['max_abs_error']:.6e} | {'PASS' if item['passed'] else 'FAIL'} |"
        for item in comparisons
    )
    markdown.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
