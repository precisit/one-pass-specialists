"""Compare the fp32 and int8 ONNX files position by position, and write a receipt.

For every position in the frozen eval set, both files score the legal columns (fed exactly as the
browser feeds them, `export_c4.compact_inputs`). The receipt records how often they choose the same
column, how close the fp32 scores were where they differ, and whether a different choice changes the
game-theoretic value of the move. Integer kernels can break near-ties differently on different CPUs,
so the receipt names the machine and the ONNX Runtime version.

  python compare_int8.py --fp32 onepass-c4-v2.onnx --int8 onepass-c4-v2-int8.onnx \
      --evalset $C4_WORK/corpus/evalset --out results/int8-vs-fp32.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort

from export_c4 import compact_inputs


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_name() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # not macOS
        return platform.processor() or platform.machine()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--evalset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    boards = np.load(args.evalset / "board.npy")
    scores = np.load(args.evalset / "scores.npy").astype(int)
    sessions = {name: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                for name, path in (("fp32", args.fp32), ("int8", args.int8))}
    preserving = {"fp32": 0, "int8": 0}
    exact = {"fp32": 0, "int8": 0}
    margins, value_changes, max_diff = [], 0, []
    for board, row in zip(boards, scores):
        ctx, options, mask, legal = compact_inputs(board)
        logits = {name: s.run(None, {"context_ids": ctx, "option_ids": options, "option_mask": mask})[0][0][: len(legal)]
                  for name, s in sessions.items()}
        max_diff.append(float(np.abs(logits["fp32"] - logits["int8"]).max()))
        best = max(v for v in row if v != -128)
        keeps = {}
        for name, value in logits.items():
            col = legal[int(np.argmax(value))]
            keeps[name] = np.sign(row[col]) == np.sign(best)
            preserving[name] += keeps[name]
            exact[name] += row[col] == best
        if int(np.argmax(logits["fp32"])) != int(np.argmax(logits["int8"])):
            top = np.sort(logits["fp32"])[::-1]
            margins.append(float(top[0] - top[1]))
            value_changes += keeps["fp32"] != keeps["int8"]
    n = len(boards)
    margins = np.array(margins)
    report = {
        "what": "fp32 vs int8 ONNX, position by position, on the frozen eval set",
        "positions": n,
        "same_column": round(1 - len(margins) / n, 4),
        "different_column": int(len(margins)),
        "fp32_gap_between_top_two_scores_where_different": {
            "median": round(float(np.median(margins)), 4) if len(margins) else None,
            "max": round(float(margins.max()), 4) if len(margins) else None,
        },
        "different_choices_that_change_value_preservation": int(value_changes),
        "value_preserving_rate": {k: round(v / n, 4) for k, v in preserving.items()},
        "exact_top1": {k: round(v / n, 4) for k, v in exact.items()},
        "max_abs_score_difference": {"median": round(float(np.median(max_diff)), 4),
                                     "p99": round(float(np.percentile(max_diff, 99)), 4),
                                     "max": round(float(np.max(max_diff)), 4)},
        "files": {"fp32": {"name": args.fp32.name, "sha256": sha256(args.fp32)},
                  "int8": {"name": args.int8.name, "sha256": sha256(args.int8)}},
        "machine": {"cpu": cpu_name(), "onnxruntime": ort.__version__, "provider": "CPUExecutionProvider"},
        "note": "Integer kernels can break near-ties differently on other CPUs; expect small differences in the counts there.",
    }
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
