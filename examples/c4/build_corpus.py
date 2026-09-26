"""Build the v2 training corpus: deduplicated, exactly labelled positions as flat numpy arrays.

Sources (tag in `src.npy`):
  0 tonyc  - TonyCWang/ConnectFour CHUNK_0 (MIT; solver self-play with temperature; every column
             scored by Pons' solver - verified against our teacher in gate G0b).
  1 gen    - our own mixed-strength games from `c4label --gen` (weak and strong players, so
             positions after blunders are covered).
  2 dagger - positions reached by the student, labelled by c4label (added in later rounds).

Output directory:
  board.npy  uint8 [N, 42]   (encode_c4 convention: bottom row first, 1 = to move, 2 = opponent)
  scores.npy int8  [N, 7]    (connect-four-ai scores, ILLEGAL = -128)
  src.npy    uint8 [N]
  key.npy    uint64 [N]      symmetry-reduced key (dedup, split hygiene)
  manifest.json              sources, counts per ply, sha256 of every array, arguments

Dedup is on the canonical key *within* the output; `--exclude` drops every key that appears in
another corpus (e.g. the eval set's keys are excluded from training above ply `--exclude-min-ply`).
Positions with fewer than two legal moves carry no decision and are dropped.

Usage:
  python build_corpus.py tonyc --split train --out $W/corpus/tonyc-train
  python build_corpus.py gen   --lines $W/gen/gen-s1.tsv --out $W/corpus/gen-s1
  python build_corpus.py merge --inputs A B --out C [--exclude EVALDIR --exclude-min-ply 12]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import encode_c4 as E

WORK = Path(os.environ.get("C4_WORK", Path(__file__).resolve().parent / "work"))
SOURCES = {"tonyc": 0, "gen": 1, "dagger": 2}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def dedup(board, scores, src, key):
    _, first = np.unique(key, return_index=True)
    first.sort()
    return board[first], scores[first], src[first], key[first]


def keep_decisions(board, scores, src, key):
    legal = (scores != E.ILLEGAL).sum(1)
    ok = legal >= 2
    return board[ok], scores[ok], src[ok], key[ok]


def write(out: Path, board, scores, src, key, extra: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"board": board, "scores": scores, "src": src, "key": key}
    manifest = dict(extra)
    manifest["rows"] = int(len(board))
    manifest["per_ply"] = np.bincount(E.plies(board), minlength=43).tolist()
    manifest["per_src"] = {name: int((src == code).sum()) for name, code in SOURCES.items()}
    manifest["sha256"] = {}
    for name, array in arrays.items():
        path = out / f"{name}.npy"
        np.save(path, array)
        manifest["sha256"][name] = sha(path)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "per_src")}), flush=True)


def read_tonyc(path: str):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    obs = table.column("obs").combine_chunks().flatten().flatten().flatten()
    obs = np.asarray(obs.to_numpy(zero_copy_only=False), dtype=np.uint8).reshape(-1, 2, 6, 7)
    target = table.column("target").combine_chunks().flatten()
    target = np.asarray(target.to_numpy(zero_copy_only=False), dtype=np.int16).reshape(-1, 7)
    board = E.boards_from_tonyc(obs)
    scores = np.where(target == -1000, E.ILLEGAL, target).astype(np.int8)
    return board, scores


def cmd_tonyc(args) -> None:
    files = sorted(glob.glob(str(WORK / "data" / "tonyc" / "CHUNK_0" / f"{args.split}-*.parquet")))
    if args.max_files:
        files = files[: args.max_files]
    parts = []
    raw = 0
    started = time.time()
    for i, path in enumerate(files):
        board, scores = read_tonyc(path)
        raw += len(board)
        src = np.full(len(board), SOURCES["tonyc"], dtype=np.uint8)
        key = E.canonical_keys(board)
        parts.append(dedup(board, scores, src, key))
        print(f"{i + 1}/{len(files)} {Path(path).name}: {len(board)} rows -> {len(parts[-1][0])} unique "
              f"({time.time() - started:.0f}s)", flush=True)
    merged = [np.concatenate(x) for x in zip(*parts)]
    merged = keep_decisions(*dedup(*merged))
    write(Path(args.out), *merged, {"source": "TonyCWang/ConnectFour@92b1f84953ba73d6d16343571f39a0f9376c069b",
                                    "config": "CHUNK_0", "split": args.split, "files": len(files),
                                    "raw_rows": raw})


def cmd_gen(args) -> None:
    boards, scores = [], []
    for path in args.lines:
        for line in open(path, encoding="ascii"):
            moves, ply, cells, _micros, *_tag = line.rstrip("\n").split("\t")
            if int(ply) < 0:
                continue
            boards.append(E.board_from_moves([int(c) - 1 for c in moves] if moves != "-" else []))
            scores.append([E.ILLEGAL if c == "x" else int(c) for c in cells.split(",")])
    board = np.stack(boards)
    score = np.asarray(scores, dtype=np.int8)
    src = np.full(len(board), SOURCES[args.tag], dtype=np.uint8)
    merged = keep_decisions(*dedup(board, score, src, E.canonical_keys(board)))
    write(Path(args.out), *merged, {"source": args.tag, "lines": [Path(p).name for p in args.lines],
                                    "raw_rows": len(board)})


def load(directory: Path):
    return [np.load(directory / f"{name}.npy") for name in ("board", "scores", "src", "key")]


def cmd_merge(args) -> None:
    parts = [load(Path(p)) for p in args.inputs]
    merged = [np.concatenate(x) for x in zip(*parts)]
    merged = dedup(*merged)
    dropped = 0
    if args.exclude:
        board, scores, src, key = merged
        banned = np.concatenate([np.load(Path(p) / "key.npy") for p in args.exclude])
        hit = np.isin(key, banned) & (E.plies(board) >= args.exclude_min_ply)
        dropped = int(hit.sum())
        merged = [board[~hit], scores[~hit], src[~hit], key[~hit]]
    write(Path(args.out), *merged, {"inputs": [Path(p).name for p in args.inputs],
                                    "excluded": [Path(p).name for p in (args.exclude or [])],
                                    "exclude_min_ply": args.exclude_min_ply, "dropped_by_exclude": dropped})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tonyc")
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("gen")
    p.add_argument("--lines", nargs="+", required=True)
    p.add_argument("--tag", default="gen", choices=list(SOURCES))
    p.add_argument("--out", required=True)
    p = sub.add_parser("merge")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--exclude", nargs="*")
    p.add_argument("--exclude-min-ply", type=int, default=12)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
