"""Time the exact solver's move choice, and write a receipt.

Choosing a move with the solver means scoring every legal column of the position exactly, which is
what `c4label` does for one input line (connect-four-ai, with its embedded depth-8 opening book).
The time depends heavily on the position, so this script times public reference sets:

  * Pascal Pons' six test sets (end, middle and beginning of the game, easy to hard), 1 000 positions
    each (they ship with connect-four-ai under crates/test-data; point $PONS_TESTS there);
  * a seeded sample of the UCI Connect-4 8-ply positions (just past the opening book's depth).

Each position is solved single-threaded and cold (the transposition table is cleared first, outside
the timed part, so earlier positions cannot help); `--threads` only runs several positions side by side, so
use an otherwise idle machine and fewer threads than cores. Native Rust, not WebAssembly: a browser
build of the same solver will be slower.

  python time_solver.py --threads 8 --uci-sample 200 --out results/solver-timing.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np

import solver
from verify_teacher import PONS, PONS_SETS, WORK, uci_board


def cpu_name() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return platform.processor() or platform.machine()


def summary(micros: list[int]) -> dict:
    ms = np.array(micros, dtype=np.float64) / 1000
    return {"positions": int(len(ms)), "mean_ms": round(float(ms.mean()), 3), "median_ms": round(float(np.median(ms)), 3),
            "p95_ms": round(float(np.percentile(ms, 95)), 3), "max_ms": round(float(ms.max()), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="positions per Pons set (0 = all)")
    parser.add_argument("--uci-sample", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    report = {"what": "time for the exact solver (connect-four-ai via c4label) to score every legal column of one position",
              "sets": {}}
    for name in PONS_SETS:
        rows = [line.split() for line in (PONS / name).read_text().splitlines() if line.strip()]
        rows = rows[: args.limit] if args.limit else rows
        out = solver.label_batch([moves for moves, _ in rows], threads=args.threads, fresh=True)
        report["sets"][f"pons/{name}"] = summary([r[3] for r in out])
    lines = (WORK / "data" / "uci" / "connect-4.data").read_text().splitlines()
    sample = random.Random(2026).sample(lines, min(args.uci_sample, len(lines)))
    boards = [solver.board_line(uci_board(line.split(",")[:-1])) for line in sample]
    out = solver.label_batch(boards, threads=args.threads, fresh=True)
    report["sets"]["uci-8-ply-sample"] = summary([r[3] for r in out])
    report["machine"] = {"cpu": cpu_name(), "cores": os.cpu_count(), "positions_in_parallel": args.threads,
                         "solver": "connect-four-ai @ 28a112a (MIT), embedded depth-8 opening book, native build",
                         "transposition_table": "cleared before every position (cold)"}
    report["wall_s"] = round(time.time() - started, 1)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
