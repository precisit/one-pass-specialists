"""Exact Connect Four labels via `c4label` (our Rust CLI over the MIT `connect-four-ai` crate).

Score convention (connect-four-ai / Pons): for the player to move, positive = win (1 = on the last
possible move, larger = sooner), 0 = draw, negative = loss. `None` = full column.

Two entry points:
  * `label_batch(lines)` - many positions, parallel, order preserved (corpus building, DAgger).
  * `Oracle` - a persistent `--serve` process for interactive players (protocol games).
Positions are passed either as 1-indexed move strings or as `b:` + 42-char boards (top row first,
x = player to move, o = opponent), see c4label/src/main.rs.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
BINARY = Path(os.environ.get("C4LABEL", HERE / "c4label" / "target" / "release" / "c4label"))

Scores = list  # list[int | None], length 7


def parse_line(line: str) -> tuple[str, int, Scores | str, int]:
    """(input, ply, scores or status string, micros)."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 4:
        raise ValueError(f"malformed c4label line: {line!r}")
    key, ply, cells, micros = parts
    if int(ply) < 0:
        return key, -1, cells, 0
    scores = [None if c == "x" else int(c) for c in cells.split(",")]
    return key, int(ply), scores, int(micros)


def label_batch(lines: list[str], threads: int | None = None, book: bool = True, fresh: bool = False) -> list[tuple]:
    args = [str(BINARY)] + (["--fresh"] if fresh else [])
    if threads:
        args += ["--threads", str(threads)]
    if not book:
        args.append("--no-book")
    payload = "".join(line.strip() + "\n" for line in lines)
    done = subprocess.run(args, input=payload, capture_output=True, text=True, check=True)
    out = [parse_line(l) for l in done.stdout.splitlines()]
    if len(out) != len(lines):
        raise RuntimeError(f"c4label returned {len(out)} rows for {len(lines)} inputs")
    for (key, *_), line in zip(out, lines):
        if key != line.strip():
            raise RuntimeError(f"c4label misaligned: {key!r} vs {line.strip()!r}")
    return out


class Oracle:
    """One persistent solver process; its transposition table stays warm across a game."""

    def __init__(self, book: bool = True) -> None:
        args = [str(BINARY), "--serve"] + ([] if book else ["--no-book"])
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def scores(self, position_line: str) -> Scores:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(position_line.strip() + "\n")
        self.proc.stdin.flush()
        key, ply, scores, _ = parse_line(self.proc.stdout.readline())
        if ply < 0:
            raise ValueError(f"oracle: {position_line!r} -> {scores}")
        return scores

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)


def board_line(cells_top_first: str) -> str:
    """`b:` input from a 42-char board, top row first, x = to move, o = opponent, . = empty."""
    assert len(cells_top_first) == 42, len(cells_top_first)
    return "b:" + cells_top_first


def moves_line(moves: list[int] | tuple[int, ...]) -> str:
    """c4label input from 0-indexed columns."""
    return "".join(str(c + 1) for c in moves) or "-"
