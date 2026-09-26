"""DAgger round: the student plays a pool of opponents; every position reached is labelled exactly.

The student's mistakes put it in positions the corpus never showed it; the opponents' blunders put
it in positions strong play never reaches (humans in the demo will produce many of those). Both are
labelled by c4label and appended as a `dagger` corpus.

Pool (per game, uniformly over the list, bot:4 listed twice): random, safe (takes a win / blocks a
loss, else random), bot:2, bot:4, bot:4, bot:6, solver-eps:0.1, solver-eps:0.3, the student itself
sampling at T=1 (`self`). The protocol's bots use a different seed stream (games are seeded by
round seed x worker), and every position is labelled by the solver, never by the bot. The student plays the
other side, argmax with probability 0.8, sampled at T=1 otherwise (explores its own near-misses).
Random opening of 0-6 plies. Output: c4label-format TSV (moves, ply, scores, micros, tag).

  python dagger.py --checkpoint RUN/model --games 20000 --out $W/gen/dagger-r1.tsv --workers 10
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np

import protocol as P

POOL = ["random", "safe", "bot:2", "bot:4", "bot:4", "bot:6", "solver-eps:0.1", "solver-eps:0.3", "self"]


class Safe:
    name = "safe"

    def move(self, position, rng):
        legal = position.legal_columns()
        wins = [c for c in legal if position.winning_move(c)]
        if wins:
            return rng.choice(wins)
        threat = P.c4.Position(position.opponent, position.mask)
        blocks = [c for c in legal if threat.winning_move(c)]
        if blocks:
            return rng.choice(blocks)
        return rng.choice(legal)


class Student:
    def __init__(self, scorer, argmax_p: float, name: str = "student") -> None:
        self.scorer, self.argmax_p, self.name = scorer, argmax_p, name

    def move(self, position, rng):
        logits = self.scorer.logits(P.E.board_from_moves(position.moves).reshape(1, 42))[0]
        legal = position.legal_columns()
        if rng.random() < self.argmax_p:
            best = max(logits[c] for c in legal)
            return rng.choice([c for c in legal if logits[c] >= best - 1e-6])
        z = np.array([logits[c] for c in legal], dtype=np.float64)
        p = np.exp(z - z.max())
        p /= p.sum()
        return legal[int(np.searchsorted(np.cumsum(p), rng.random() * p.sum()))]


def worker(job) -> list[str]:
    import torch

    torch.set_num_threads(1)
    from model_c4 import Scorer

    checkpoint, games, seed = job
    scorer = Scorer(checkpoint, "cpu")
    student = Student(scorer, 0.8)
    opponents = {"random": P.Random(), "safe": Safe(), "bot:2": P.Bot(2), "bot:4": P.Bot(4), "bot:6": P.Bot(6),
                 "solver-eps:0.1": P.Perfect(0.1), "solver-eps:0.3": P.Perfect(0.3),
                 "self": Student(scorer, 0.0, "self")}
    rng = random.Random(seed)
    prefixes: set[str] = set()
    for game in games:
        rng.seed(seed * 1_000_003 + game)
        opponent = opponents[rng.choice(POOL)]
        position = P.c4.Position(0, 0, ())
        for _ in range(rng.randint(0, 6)):
            col = rng.choice(position.legal_columns())
            if position.winning_move(col):
                break
            position = position.play(col)
        players = (student, opponent) if game % 2 == 0 else (opponent, student)
        turn = len(position.moves) % 2
        while not position.is_full() and not position.has_won():
            if len(position.legal_columns()) >= 2:
                prefixes.add(P.S.moves_line(position.moves))
            col = players[turn].move(position, rng)
            if position.winning_move(col):
                break
            position = position.play(col)
            turn = 1 - turn
    for opponent in opponents.values():
        if isinstance(opponent, P.Perfect):
            opponent.oracle.close()
    return sorted(prefixes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label-threads", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label-only", action="store_true", help="skip the games, label <out>.positions")
    args = parser.parse_args()
    started = time.time()
    out = Path(args.out)
    positions = out.with_suffix(".positions")
    if args.label_only:
        lines = positions.read_text().split()
    else:
        games = list(range(args.games))
        jobs = [(args.checkpoint, games[i::args.workers], args.seed * 100 + i) for i in range(args.workers)]
        with mp.get_context("spawn").Pool(args.workers) as pool:
            parts = pool.map(worker, jobs)
        lines = sorted(set(l for part in parts for l in part))
        # saved before labelling, so a slow labelling pass can be restarted with more threads
        positions.write_text("\n".join(lines) + "\n")
        print(f"played {args.games} games in {time.time() - started:.0f}s -> {len(lines)} distinct positions", flush=True)
    with out.open("w") as handle:
        for i in range(0, len(lines), 50_000):
            for key, ply, scores, micros in P.S.label_batch(lines[i:i + 50_000], threads=args.label_threads):
                if ply < 0:
                    continue
                cells = ",".join("x" if s is None else str(s) for s in scores)
                handle.write(f"{key}\t{ply}\t{cells}\t{micros}\tdagger\n")
            print(f"labelled {min(i + 50_000, len(lines))}/{len(lines)} ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
