"""The frozen evaluation protocol (see results/PROTOCOL.md). One script, pinned seeds.

Design (kept from the first version's protocol, where each rule was learned from a measurement error):
  * colours alternate within a cell and A/B are tallied, never first/second;
  * epsilon-noise (default 0.05) on BOTH players, so N games are N games;
  * Wilson intervals on every rate;
  * the hand-written bots are `engine.bot_move` (depth-limited alpha-beta + heuristic leaf), the
    same bots the first version was measured against, so "bot:4" means the same thing in both.
New:
  * `solver` (perfect: uniformly random among the best-scoring moves) and `solver-eps:E`;
  * `model:<checkpoint>` is a pure policy - one forward pass, argmax over legal columns;
  * `--opening K`: K uniformly random plies before the players take over (0 = empty board);
  * every decision a model makes is labelled afterwards by the exact solver, giving the
    on-policy value-preserving rate and the ply of the first value-losing move per game;
  * games run in parallel worker processes, each with its own solver process.

Usage:
  python protocol.py --a model:runs/m/model --b bot:4 --games 200
  python protocol.py --a solver --b bot:6 --games 200          # the ceiling probe
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import engine as c4  # noqa: E402  (board + the hand-written bots)
import encode_c4 as E  # noqa: E402
import solver as S  # noqa: E402


def render_v1(position) -> str:
    """v1's context: whose turn, the move history, the board (top row first), free columns."""
    to_move = "X" if position.stones % 2 == 0 else "O"
    drag = " ".join(str(column + 1) for column in position.moves) if position.moves else "-"
    return f"SPELARE {to_move}\nDRAG {drag}\n{position.render()}\nLEDIGA {len(position.legal_columns())}"


def wilson(k: float, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


class Random:
    def __init__(self) -> None:
        self.name = "random"

    def move(self, position, rng):
        return rng.choice(position.legal_columns())


class Bot:
    def __init__(self, depth: int) -> None:
        self.name = f"bot:{depth}"
        self.depth = depth

    def move(self, position, rng):
        return c4.bot_move(position, self.depth)


class Perfect:
    def __init__(self, eps: float = 0.0) -> None:
        self.name = "solver" if not eps else f"solver-eps:{eps}"
        self.eps = eps
        self.oracle = S.Oracle()

    def move(self, position, rng):
        legal = position.legal_columns()
        if self.eps and rng.random() < self.eps:
            return rng.choice(legal)
        scores = self.oracle.scores(S.moves_line(position.moves))
        best = max(s for s in scores if s is not None)
        return rng.choice([c for c in legal if scores[c] == best])


class Model:
    def __init__(self, checkpoint: str) -> None:
        import torch

        torch.set_num_threads(1)
        from model_c4 import Scorer

        self.scorer = Scorer(checkpoint, "cpu")
        self.name = f"model:{Path(checkpoint).parent.name}"
        self.decisions: list[tuple[str, int]] = []

    def move(self, position, rng):
        board = E.board_from_moves(position.moves)
        logits = self.scorer.logits(board.reshape(1, 42))[0]
        best = logits.max()
        tied = [c for c in range(7) if logits[c] >= best - 1e-6]
        choice = rng.choice(tied)
        # (position, the column *this model* chose). Recording only the position and looking the
        # move up afterwards would merge the opponent's moves at the same prefix into the model's
        # decisions - the model plays X in half the games (a real bug in an earlier version).
        self.decisions.append((S.moves_line(position.moves), choice))
        return choice


class OnnxV1:
    """The previous generation (v1, `onepass-c4-8x24.onnx` from HF precisit/onepass-c4), played
    exactly as the demo plays it: v1's text rendering (SPELARE/DRAG/grid/LEDIGA, 224 bytes),
    options "spela kolumn N" in 8 x 24 slots, argmax. Lets v2 meet v1 in Python, not only in the
    browser arena."""

    def __init__(self, path: str) -> None:
        import onnxruntime as ort

        self.render = render_v1
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        self.name = f"onnx-v1:{Path(path).name}"

    def move(self, position, rng):
        import numpy as np

        legal = position.legal_columns()
        ctx = np.zeros((1, 224), dtype=np.int32)
        raw = self.render(position).encode("utf-8")[:224]
        ctx[0, : len(raw)] = np.frombuffer(raw, dtype=np.uint8) + 1
        opt = np.zeros((1, 8, 24), dtype=np.int32)
        mask = np.zeros((1, 8), dtype=np.int32)
        for i, col in enumerate(legal):
            text = f"spela kolumn {col + 1}".encode("utf-8")
            opt[0, i, : len(text)] = np.frombuffer(text, dtype=np.uint8) + 1
            mask[0, i] = 1
        logits = self.session.run(None, {"context_ids": ctx, "option_ids": opt, "option_mask": mask})[0][0][: len(legal)]
        best = logits.max()
        return rng.choice([c for c, v in zip(legal, logits) if v >= best - 1e-6])


class OnnxV2:
    """An exported v2 ONNX file (fp32 or int8), fed exactly as the browser feeds it (legal options
    packed from slot 0, `export_c4.compact_inputs`). Used to check that the file that ships plays
    like the checkpoint it came from."""

    def __init__(self, path: str) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        self.name = f"onnx-v2:{Path(path).name}"
        self.decisions: list[tuple[str, int]] = []

    def move(self, position, rng):
        from export_c4 import compact_inputs

        ctx, opt, mask, legal = compact_inputs(E.board_from_moves(position.moves))
        logits = self.session.run(None, {"context_ids": ctx, "option_ids": opt, "option_mask": mask})[0][0][: len(legal)]
        best = logits.max()
        choice = rng.choice([c for c, v in zip(legal, logits) if v >= best - 1e-6])
        self.decisions.append((S.moves_line(position.moves), choice))
        return choice


def make(spec: str):
    if spec == "random":
        return Random()
    if spec.startswith("bot:"):
        return Bot(int(spec.split(":")[1]))
    if spec == "solver":
        return Perfect()
    if spec.startswith("solver-eps:"):
        return Perfect(float(spec.split(":")[1]))
    if spec.startswith("model:"):
        return Model(spec.split(":", 1)[1])
    if spec.startswith("onnx-v2:"):
        return OnnxV2(spec.split(":", 1)[1])
    if spec.startswith("onnx-v1:"):
        return OnnxV1(spec.split(":", 1)[1])
    raise SystemExit(f"unknown player {spec}")


class Noisy:
    def __init__(self, inner, eps: float) -> None:
        self.inner, self.eps, self.name = inner, eps, inner.name
        self.noise_moves: set[str] = set()

    def move(self, position, rng):
        if self.eps and rng.random() < self.eps:
            self.noise_moves.add(S.moves_line(position.moves))
            return rng.choice(position.legal_columns())
        return self.inner.move(position, rng)


def play(first, second, opening: int, rng: random.Random) -> tuple[str, str]:
    position = c4.Position(0, 0, ())
    while len(position.moves) < opening and not position.is_full():
        col = rng.choice(position.legal_columns())
        if position.winning_move(col):  # the random opening never ends the game by itself
            continue
        position = position.play(col)
    to_move = 0
    players = (first, second)
    while not position.is_full() and not position.has_won():
        position = position.play(players[to_move].move(position, rng))
        to_move = 1 - to_move
    if position.has_won():
        return ("second" if to_move == 0 else "first"), S.moves_line(position.moves)
    return "draw", S.moves_line(position.moves)


def worker(job) -> dict:
    spec_a, spec_b, games, opening, noise, seed = job
    a, b = Noisy(make(spec_a), noise), Noisy(make(spec_b), noise)
    rng = random.Random(seed)
    tally = {"a": 0, "b": 0, "draw": 0, "first": 0}
    lines = []
    for game, a_first in games:
        rng.seed(seed * 1_000_003 + game)
        result, line = play(a, b, opening, rng) if a_first else play(b, a, opening, rng)
        lines.append(line)
        tally["first"] += result == "first"
        if result == "draw":
            tally["draw"] += 1
        elif (result == "first") == a_first:
            tally["a"] += 1
        else:
            tally["b"] += 1
    decisions = {}
    for side, player in (("a", a), ("b", b)):
        inner = player.inner
        if isinstance(inner, (Model, OnnxV2)):
            decisions[side] = list(inner.decisions)  # noise moves never reach inner.move
    return {"tally": tally, "lines": lines, "decisions": decisions}


def on_policy(decisions: list[tuple[str, int]]) -> dict:
    """Label each model decision exactly: did the column it chose keep the game-theoretic value?"""
    if not decisions:
        return {}
    labelled = S.label_batch([d[0] for d in decisions])
    kept = total = nontrivial_kept = nontrivial = 0
    by_band = {}
    for (key, ply, scores, _), (_, chosen) in zip(labelled, decisions):
        if ply < 0:
            continue
        cols = {chosen}
        legal = [s for s in scores if s is not None]
        best = max(legal)
        signs = {(s > 0) - (s < 0) for s in legal}
        for col in cols:
            ok = ((scores[col] > 0) - (scores[col] < 0)) == ((best > 0) - (best < 0))
            total += 1
            kept += ok
            band = f"{(ply // 8) * 8}-{(ply // 8) * 8 + 7}"
            by_band.setdefault(band, [0, 0])
            by_band[band][0] += ok
            by_band[band][1] += 1
            if len(signs) > 1:
                nontrivial += 1
                nontrivial_kept += ok
    return {"decisions": total, "value_preserving": round(kept / total, 4) if total else None,
            "value_preserving_nontrivial": round(nontrivial_kept / nontrivial, 4) if nontrivial else None,
            "bands": {k: round(v[0] / v[1], 4) for k, v in sorted(by_band.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--opening", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", help="append the JSON report to this file")
    args = parser.parse_args()

    started = time.time()
    games = [(g, g % 2 == 0) for g in range(args.games)]
    chunks = [games[i::args.workers] for i in range(args.workers)]
    jobs = [(args.a, args.b, chunk, args.opening, args.noise, args.seed) for chunk in chunks if chunk]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(worker, jobs)
    tally = {k: sum(r["tally"][k] for r in results) for k in ("a", "b", "draw", "first")}
    lines = [l for r in results for l in r["lines"]]
    n = args.games
    score = (tally["a"] + 0.5 * tally["draw"]) / n
    report = {"a": args.a, "b": args.b, "games": n, "opening": args.opening, "noise": args.noise, "seed": args.seed,
              "a_wins": tally["a"], "b_wins": tally["b"], "draws": tally["draw"],
              "a_win_rate": round(tally["a"] / n, 4), "a_win_rate_ci": wilson(tally["a"], n),
              "a_score": round(score, 4), "a_score_ci": wilson(score * n, n),
              "first_player_win_rate": round(tally["first"] / n, 4),
              "distinct_games": len(set(lines)), "wall_s": round(time.time() - started, 1)}
    for side in ("a", "b"):
        decisions = [d for r in results for d in r["decisions"].get(side, [])]
        if decisions:
            report[f"{side}_on_policy"] = on_policy(decisions)
    text = json.dumps(report, sort_keys=True)
    print(text, flush=True)
    if args.out:
        with open(args.out, "a") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
