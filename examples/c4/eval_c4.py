"""Move-level evaluation and controls (results/PROTOCOL.md).

  python eval_c4.py build-evalset --tonyc-test DIR --gen DIR --out DIR   # the frozen eval set
  python eval_c4.py score --checkpoint RUN/model --evalset DIR [--controls]

`score` prints one JSON line: value-preserving (VP) overall / non-trivial / exact top-1, per ply
band, and with --controls:
  * option-order invariance: option slots permuted at random (logits un-permuted) - choice must match;
  * mirror consistency: mirrored board's choice == mirror of the choice, counted where the choice
    is unambiguous (the chosen logit beats the runner-up by > 1e-4);
  * owner shuffle: every stone's owner reassigned at random (heights kept); VP on non-trivial
    positions must drop by >= 10 points (the model must read the board).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import encode_c4 as E
from train_c4 import move_metrics

PER_PLY = 480


def cmd_build(args) -> None:
    rng = np.random.default_rng(2026)
    parts = []
    for d in (args.tonyc_test, args.gen):
        board = np.load(Path(d) / "board.npy", mmap_mode="r")
        scores = np.load(Path(d) / "scores.npy", mmap_mode="r")
        key = np.load(Path(d) / "key.npy", mmap_mode="r")
        ply = E.plies(np.asarray(board))
        pick = []
        for p in range(42):
            idx = np.flatnonzero(ply == p)
            take = min(len(idx), PER_PLY // 2)
            if take:
                pick.extend(rng.choice(idx, size=take, replace=False).tolist())
        pick = np.sort(np.asarray(pick))
        parts.append((np.asarray(board[pick]), np.asarray(scores[pick]), np.asarray(key[pick])))
    board = np.concatenate([p[0] for p in parts])
    scores = np.concatenate([p[1] for p in parts])
    key = np.concatenate([p[2] for p in parts])
    src = np.concatenate([np.full(len(parts[0][0]), 0, np.uint8), np.full(len(parts[1][0]), 1, np.uint8)])
    _, first = np.unique(key, return_index=True)
    first.sort()
    from build_corpus import write
    write(Path(args.out), board[first], scores[first], src[first], key[first],
          {"evalset": "v2", "per_ply_target": PER_PLY, "from": [Path(args.tonyc_test).name, Path(args.gen).name]})


def owner_shuffle(boards: np.ndarray, rng) -> np.ndarray:
    out = boards.copy()
    occupied = out > 0
    out[occupied] = rng.integers(1, 3, size=int(occupied.sum()), dtype=np.uint8)
    return out


def cmd_score(args) -> None:
    from model_c4 import Scorer

    device = args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
    scorer = Scorer(args.checkpoint, device)
    boards = np.load(Path(args.evalset) / "board.npy", mmap_mode="r")
    scores = np.load(Path(args.evalset) / "scores.npy", mmap_mode="r")
    if args.limit and args.limit < len(boards):
        pick = np.sort(np.random.default_rng(5).permutation(len(boards))[: args.limit])
        boards, scores = boards[pick], scores[pick]
    boards, scores = np.asarray(boards), np.asarray(scores)
    logits = scorer.logits(boards)
    report = {"checkpoint": str(Path(args.checkpoint).parent.name), **move_metrics(logits, boards, scores)}
    if args.controls:
        rng = np.random.default_rng(11)
        # option-order invariance: run the raw model with permuted slots
        model, batcher = scorer.model, scorer.batcher
        agree = total = 0
        with torch.no_grad():
            for i in range(0, len(boards), 4096):
                b = batcher(torch.from_numpy(boards[i:i + 4096]))
                perm = torch.stack([torch.randperm(7) for _ in range(b["option_ids"].shape[0])]).to(b["option_ids"].device)
                pb = dict(b)
                pb["option_ids"] = torch.gather(b["option_ids"], 1, perm[:, :, None].expand(-1, -1, E.OPTION_BYTES))
                pb["option_token_mask"] = pb["option_ids"].ne(0)
                pb["option_mask"] = torch.gather(b["option_mask"], 1, perm)
                pl = model(pb).float()
                unperm = torch.empty_like(pl).scatter_(1, perm, pl)
                base = model(b).float()
                legal = b["option_mask"]
                c1 = torch.where(legal, base, torch.full_like(base, -1e9)).argmax(1)
                c2 = torch.where(legal, unperm, torch.full_like(base, -1e9)).argmax(1)
                agree += int((c1 == c2).sum())
                total += len(c1)
        report["control_option_order"] = round(agree / total, 5)
        mirrored = E.mirror_boards(boards)
        ml = scorer.logits(mirrored)
        choice = logits.argmax(1)
        mchoice = ml.argmax(1)
        srt = np.sort(np.where(np.isfinite(logits), logits, -1e9), 1)
        clear = (srt[:, -1] - srt[:, -2]) > 1e-4
        report["control_mirror"] = round(float((mchoice[clear] == 6 - choice[clear]).mean()), 5)
        shuffled = owner_shuffle(boards, rng)
        sl = scorer.logits(shuffled)
        shuffled_report = move_metrics(sl, boards, scores)  # judged against the TRUE labels
        report["control_owner_shuffle_vp_nt"] = shuffled_report["value_preserving_nontrivial"]
        report["control_owner_shuffle_drop_nt"] = round(report["value_preserving_nontrivial"] - shuffled_report["value_preserving_nontrivial"], 4)
    print(json.dumps(report, sort_keys=True), flush=True)
    if args.out:
        with open(args.out, "a") as handle:
            handle.write(json.dumps(report, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build-evalset")
    p.add_argument("--tonyc-test", required=True)
    p.add_argument("--gen", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--evalset", required=True)
    p.add_argument("--controls", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0, help="score a random subset (not for the frozen eval set)")
    p.add_argument("--out")
    args = parser.parse_args()
    {"build-evalset": cmd_build, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
