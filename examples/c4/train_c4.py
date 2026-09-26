"""Train the one-pass scorer on exactly labelled Connect Four positions.

Loss (per position, over its legal columns), from connect-four-ai scores s_i:
  --loss soft : cross-entropy to softmax(u), u_i = M*sign(s_i) + s_i/tau.  The W/D/L class
                dominates (M = 12) and, within a class, a faster win / slower loss is preferred.
  --loss set  : cross-entropy to the uniform distribution over the value-preserving set
                (every move with the best sign) - arXiv:2607.08984's optimal-set target.
Both train *all* options at once (listwise), unlike the first version's single hard label.

Sampling is ply-balanced: each ply gets probability proportional to clip(count, --ply-min-weight,
--ply-floor), then a uniform position within the ply, so openings are not drowned by the midgame and the midgame is
not drowned by duplicates. Horizontal mirroring is applied at random (the game is symmetric).

Validation reports the value-preserving rate (argmax keeps the game-theoretic result), the
distance-exact top-1, and both per ply band. The best checkpoint by value-preserving rate is kept.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import encode_c4 as E
from model_c4 import Batcher, config_for, make_system, parameter_count, save_checkpoint

BANDS = [(0, 7), (8, 15), (16, 23), (24, 31), (32, 41)]
MARGIN = 12.0


def load_corpus(dirs: list[str]):
    boards, scores = [], []
    for d in dirs:
        boards.append(np.load(Path(d) / "board.npy", mmap_mode="r"))
        scores.append(np.load(Path(d) / "scores.npy", mmap_mode="r"))
    return np.concatenate(boards), np.concatenate(scores)


class PlySampler:
    def __init__(self, boards: np.ndarray, floor: int, seed: int, min_weight: int = 0) -> None:
        ply = E.plies(boards)
        self.order = np.argsort(ply, kind="stable")
        counts = np.bincount(ply, minlength=43)
        self.offsets = np.concatenate([[0], np.cumsum(counts)])
        # Openings have few distinct positions but every game passes through them: a ply with any
        # positions gets at least `min_weight` (so the empty board is sampled as often as a ply
        # with `min_weight` positions would be), capped at `floor` like every other ply.
        weights = np.where(counts > 0, np.clip(counts, min_weight, floor), 0).astype(np.float64)
        self.p = weights / weights.sum()
        self.counts = counts
        self.rng = np.random.default_rng(seed)

    def __call__(self, n: int) -> np.ndarray:
        plies = self.rng.choice(len(self.p), size=n, p=self.p)
        within = (self.rng.random(n) * self.counts[plies]).astype(np.int64)
        return self.order[self.offsets[plies] + within]


def targets(scores: torch.Tensor, loss: str, tau: float) -> torch.Tensor:
    legal = scores != E.ILLEGAL
    s = scores.float()
    if loss == "soft":
        u = MARGIN * torch.sign(s) + s / tau
    else:
        best = torch.where(legal, torch.sign(s), torch.full_like(s, -9)).max(1, keepdim=True).values
        u = torch.where(torch.sign(s) == best, torch.zeros_like(s), torch.full_like(s, -1e4))
    u = torch.where(legal, u, torch.full_like(u, -1e4))
    return u.softmax(1)


def mirror(board: torch.Tensor, scores: torch.Tensor, flip: torch.Tensor):
    b = board.view(-1, 6, 7)
    board = torch.where(flip[:, None, None], b.flip(2), b).reshape(-1, 42)
    scores = torch.where(flip[:, None], scores.flip(1), scores)
    return board, scores


@torch.no_grad()
def evaluate(model, batcher, boards: np.ndarray, scores: np.ndarray, device, batch: int = 4096) -> dict:
    model.eval()
    logits = []
    for i in range(0, len(boards), batch):
        b = batcher(torch.from_numpy(np.ascontiguousarray(boards[i:i + batch])))
        logits.append(model(b).float().cpu().numpy())
    model.train()
    return move_metrics(np.concatenate(logits), boards, scores)


def move_metrics(logits: np.ndarray, boards: np.ndarray, scores: np.ndarray) -> dict:
    scores = scores.astype(np.int32)
    legal = scores != E.ILLEGAL
    logits = np.where(legal, logits, -np.inf)
    choice = logits.argmax(1)
    chosen = scores[np.arange(len(scores)), choice]
    best = np.where(legal, scores, -999).max(1)
    preserving = np.sign(chosen) == np.sign(best)
    exact = chosen == best
    signs = np.where(legal, np.sign(scores), 9)
    nontrivial = (np.where(legal, signs, -9).max(1) != np.where(legal, signs, 9).min(1))
    ply = E.plies(boards)
    report = {"n": int(len(scores)), "value_preserving": float(preserving.mean()),
              "value_preserving_nontrivial": float(preserving[nontrivial].mean()) if nontrivial.any() else None,
              "exact_top1": float(exact.mean()), "bands": {}}
    for lo, hi in BANDS:
        m = (ply >= lo) & (ply <= hi)
        if m.any():
            report["bands"][f"{lo}-{hi}"] = {"n": int(m.sum()), "vp": round(float(preserving[m].mean()), 4),
                                            "vp_nt": round(float(preserving[m & nontrivial].mean()), 4) if (m & nontrivial).any() else None,
                                            "exact": round(float(exact[m].mean()), 4)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--val", required=True, help="corpus dir used for validation (held out)")
    parser.add_argument("--val-limit", type=int, default=40_000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", choices=["S", "M", "L"], default="M")
    parser.add_argument("--loss", choices=["soft", "set"], default="soft")
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--ply-floor", type=int, default=20_000)
    parser.add_argument("--ply-min-weight", type=int, default=2_000)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init", help="checkpoint dir to start from (fine-tuning / DAgger)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    boards, scores = load_corpus(args.train)
    vb = np.load(Path(args.val) / "board.npy", mmap_mode="r")
    vs = np.load(Path(args.val) / "scores.npy", mmap_mode="r")
    pick = np.random.default_rng(1).permutation(len(vb))[: args.val_limit]
    pick.sort()
    vb, vs = np.asarray(vb[pick]), np.asarray(vs[pick])

    if args.init:
        from model_c4 import load_checkpoint
        model, _, config = load_checkpoint(Path(args.init), device)
        model.train()
    else:
        config = config_for(args.size)
        model, _ = make_system(config, device)
    batcher = Batcher(device)
    params = parameter_count(model)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embedding" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "embedding" not in n)]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
    schedule = lambda step: min(1.0, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    sampler = PlySampler(boards, args.ply_floor, args.seed, args.ply_min_weight)
    gen = torch.Generator().manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "train-log.jsonl").open("a")
    meta = {"args": vars(args), "params": params, "train_rows": int(len(boards)), "config": config}
    print(json.dumps(meta), flush=True)
    log.write(json.dumps(meta) + "\n")
    best_vp, started, running = -1.0, time.time(), 0.0
    for step in range(args.steps):
        idx = np.sort(sampler(args.batch))
        b = torch.from_numpy(np.asarray(boards[idx]))
        s = torch.from_numpy(np.asarray(scores[idx]))
        flip = torch.rand(len(idx), generator=gen) < 0.5
        b, s = mirror(b, s, flip)
        batch = batcher(b)
        logits = model(batch).float()
        target = targets(s.to(device), args.loss, args.tau)
        legal = batch["option_mask"]
        logp = torch.where(legal, logits.log_softmax(1), torch.zeros_like(logits))
        loss = -(target * logp).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        value = float(loss.detach())
        running = 0.98 * running + 0.02 * value if step else value
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            report = evaluate(model, batcher, vb, vs, device)
            report.update({"step": step + 1, "loss": round(running, 4), "lr": scheduler.get_last_lr()[0],
                           "elapsed_s": round(time.time() - started, 1)})
            print(json.dumps(report), flush=True)
            log.write(json.dumps(report) + "\n")
            log.flush()
            if report["value_preserving"] > best_vp:
                best_vp = report["value_preserving"]
                save_checkpoint(out / "model", model, config,
                                {"step": step + 1, "value_preserving": best_vp, "loss": args.loss,
                                 "train": [Path(t).name for t in args.train]})
            save_checkpoint(out / "last", model, config, {"step": step + 1})
    print(json.dumps({"done": True, "best_value_preserving": best_vp}), flush=True)


if __name__ == "__main__":
    main()
