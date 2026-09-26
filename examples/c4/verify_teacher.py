"""Gate G0 / G0b: is the teacher right, and does the downloaded corpus agree with it?

Subcommands (each prints one JSON line):
  pons   - the six public test sets shipped with connect-four-ai (Pons' protocol): position score
           = max over child scores must equal the expected score, for every position.
  uci    - UCI Connect-4 (Tromp, CC BY 4.0): all 67 557 8-ply positions, W/L/D for the player to
           move (x); the sign of our position score must match.
  ucibook - all 67 557 UCI positions answered by the embedded depth-8 opening book (W/L/D sign).
  book   - the embedded depth-8 opening book against book-free search on early positions.
  tonyc  - TonyCWang/ConnectFour rows re-solved: all seven child scores must match exactly.

Data root: $C4_WORK (default ./work). Pons' test sets: $PONS_TESTS (they ship in the connect-four-ai
repository under crates/test-data; used here only to verify the labeller, not redistributed).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path

import solver

WORK = Path(os.environ.get("C4_WORK", Path(__file__).resolve().parent / "work"))
PONS = Path(os.environ.get("PONS_TESTS", WORK / "connect-four-ai" / "crates" / "test-data"))
PONS_SETS = ["end-easy", "middle-easy", "middle-medium", "begin-easy", "begin-medium", "begin-hard"]


def best(scores) -> int:
    return max(s for s in scores if s is not None)


def sign(v: int) -> int:
    return (v > 0) - (v < 0)


def cmd_pons(args) -> dict:
    report = {"gate": "pons"}
    for name in PONS_SETS:
        rows = [l.split() for l in (PONS / name).read_text().splitlines() if l.strip()]
        if args.limit:
            rows = rows[: args.limit]
        started = time.time()
        out = solver.label_batch([m for m, _ in rows], threads=args.threads)
        wrong = [(m, e, best(s)) for (m, e), (_, _, s, _) in zip(rows, out) if best(s) != int(e)]
        report[name] = {"n": len(rows), "wrong": len(wrong), "examples": wrong[:3],
                        "wall_s": round(time.time() - started, 2),
                        "mean_label_ms": round(sum(r[3] for r in out) / len(out) / 1000, 3)}
    return report


def uci_board(cells: list[str]) -> str:
    # attributes are a1..a6, b1..b6, ... (column-major, row 1 = bottom); x is to move at 8 ply
    grid = [["."] * 7 for _ in range(6)]  # grid[row_from_top][col]
    for index, value in enumerate(cells):
        col, row = divmod(index, 6)
        grid[5 - row][col] = {"x": "x", "o": "o", "b": "."}[value]
    return "".join("".join(r) for r in grid)


def cmd_uci(args) -> dict:
    lines = (WORK / "data" / "uci" / "connect-4.data").read_text().splitlines()
    rng = random.Random(2026)
    if args.limit and args.limit < len(lines):
        lines = rng.sample(lines, args.limit)
    boards, truth = [], []
    for line in lines:
        *cells, cls = line.split(",")
        boards.append(solver.board_line(uci_board(cells)))
        truth.append({"win": 1, "draw": 0, "loss": -1}[cls])
    started = time.time()
    out = solver.label_batch(boards, threads=args.threads)
    wrong = [(b, t, best(s)) for b, t, (_, _, s, _) in zip(boards, truth, out) if sign(best(s)) != t]
    micros = sorted(r[3] for r in out)
    return {"gate": "uci", "n": len(boards), "wrong": len(wrong), "examples": wrong[:3],
            "wall_s": round(time.time() - started, 1),
            "label_ms_p50": micros[len(micros) // 2] / 1000, "label_ms_p99": micros[int(len(micros) * .99)] / 1000,
            "class_counts": dict(collections.Counter(truth))}


def cmd_ucibook(args) -> dict:
    """All 67 557 UCI 8-ply positions are inside the depth-8 opening book: check the book's value of
    each one (`c4label --value`, which answers from the book) against Tromp's W/L/D label."""
    import subprocess

    lines = (WORK / "data" / "uci" / "connect-4.data").read_text().splitlines()
    boards, truth = [], []
    for line in lines:
        *cells, cls = line.split(",")
        boards.append(solver.board_line(uci_board(cells)))
        truth.append({"win": 1, "draw": 0, "loss": -1}[cls])
    done = subprocess.run([str(solver.BINARY), "--value"], input="\n".join(boards) + "\n",
                          capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(done) == len(boards)
    wrong, slow = [], 0
    for b, t, line in zip(boards, truth, done):
        key, ply, value, micros = line.split("\t")
        slow += int(micros) > 1000  # a book hit is microseconds; a search would be ms-s
        if sign(int(value)) != t:
            wrong.append((b, t, value))
    return {"gate": "ucibook", "n": len(boards), "wrong": len(wrong), "examples": wrong[:3],
            "answered_slower_than_1ms": slow}


def cmd_book(args) -> dict:
    rng = random.Random(11)
    lines = set()
    while len(lines) < args.limit:
        moves, heights = [], [0] * 7
        for _ in range(rng.randint(args.min_ply, args.max_ply)):
            col = rng.choice([c for c in range(7) if heights[c] < 6])
            moves.append(col)
            heights[col] += 1
        lines.add(solver.moves_line(moves))
    lines = sorted(lines)
    with_book = solver.label_batch(lines, threads=args.threads)
    usable = [l for l, r in zip(lines, with_book) if r[1] >= 0]
    started = time.time()
    no_book = solver.label_batch(usable, threads=args.threads, book=False)
    by_key = {r[0]: r for r in with_book}
    wrong = [(r[0], by_key[r[0]][2], r[2]) for r in no_book if by_key[r[0]][2] != r[2]]
    return {"gate": "book", "n": len(usable), "wrong": len(wrong), "examples": wrong[:3],
            "no_book_wall_s": round(time.time() - started, 1), "plies": [args.min_ply, args.max_ply]}


def tonyc_rows(path: Path):
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    obs = np.stack([np.asarray(o, dtype=np.uint8) for o in table.column("obs").to_pylist()])
    target = np.asarray(table.column("target").to_pylist(), dtype=np.int16)
    return obs, target


def tonyc_board(obs) -> str:
    # obs[0] = player to move, obs[1] = opponent; row 0 is the TOP row (verified 2026-09-26)
    return "".join("x" if obs[0][r][c] else ("o" if obs[1][r][c] else ".") for r in range(6) for c in range(7))


def cmd_tonyc(args) -> dict:
    import numpy as np

    obs, target = tonyc_rows(Path(args.parquet))
    plies = (obs > 0).reshape(len(obs), -1).sum(1)
    rng = np.random.default_rng(2026)
    # stratify: an equal share per ply
    pick = []
    for ply in range(42):
        idx = np.flatnonzero(plies == ply)
        if len(idx):
            pick.extend(rng.choice(idx, size=min(len(idx), args.limit // 42 + 1), replace=False).tolist())
    boards = [solver.board_line(tonyc_board(obs[i])) for i in pick]
    out = solver.label_batch(boards, threads=args.threads)
    wrong, statuses = [], collections.Counter()
    for i, board, (_, ply, s, _) in zip(pick, boards, out):
        if ply < 0:
            statuses[s] += 1
            continue
        theirs = [None if v == -1000 else int(v) for v in target[i]]
        if theirs != s:
            wrong.append((board, theirs, s))
    return {"gate": "tonyc", "file": Path(args.parquet).name, "n": len(pick), "wrong": len(wrong),
            "examples": wrong[:3], "skipped": dict(statuses),
            "ply_hist_file": np.bincount(plies, minlength=43).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gate", choices=["pons", "uci", "ucibook", "book", "tonyc"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--min-ply", type=int, default=4)
    parser.add_argument("--max-ply", type=int, default=7)
    parser.add_argument("--parquet", default=str(WORK / "data" / "tonyc" / "CHUNK_0" / "test-00000-of-00020.parquet"))
    args = parser.parse_args()
    report = globals()[f"cmd_{args.gate}"](args)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
