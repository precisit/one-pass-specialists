"""train_c4.py on MLX: same data, sampling, targets, schedule and metrics; ~3x faster on Apple GPUs.

The model is the toolkit's `TinyTransformerScorer`, initialised by the toolkit's own constructor,
trained through its MLX twin (`mlx_scorer.py`, logit parity ~1e-6), and saved as a toolkit
checkpoint (safetensors + config) - so everything downstream (eval, games, ONNX export) uses the
toolkit's PyTorch class. Every saved checkpoint is re-checked for parity against that class.

  python train_mlx.py --train DIR [DIR ...] --val DIR --out RUN --size M --steps 30000
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import torch

import encode_c4 as E
import mlx_scorer as X
from model_c4 import config_for, load_checkpoint, make_system, parameter_count, save_checkpoint
from train_c4 import MARGIN, PlySampler, load_corpus, move_metrics


def targets(scores: mx.array, loss: str, tau: float) -> mx.array:
    legal = scores != E.ILLEGAL
    s = scores.astype(mx.float32)
    if loss == "soft":
        u = MARGIN * mx.sign(s) + s / tau
    else:
        best = mx.where(legal, mx.sign(s), -9.0).max(axis=1, keepdims=True)
        u = mx.where(mx.sign(s) == best, 0.0, -1e4)
    u = mx.where(legal, u, -1e4)
    return mx.softmax(u, axis=1)


def decays(name: str, value) -> bool:
    return value.ndim >= 2 and "embedding" not in name and "position" not in name


def save(out: Path, params: dict, config: dict, meta: dict, check_boards: np.ndarray) -> float:
    model, _ = make_system(config, "cpu")
    state = {k: torch.from_numpy(np.array(v)) for k, v in params.items()}
    model.load_state_dict(state, strict=True)
    save_checkpoint(out, model, config, meta)
    return X.parity(model, params, config, check_boards)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--weights", nargs="*", type=float, help="per-corpus sampling share (default: pooled)")
    parser.add_argument("--val", required=True)
    parser.add_argument("--val-limit", type=int, default=40_000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", choices=["S", "M", "L"], default="M")
    parser.add_argument("--loss", choices=["soft", "set"], default="soft")
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr-frac", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--ply-floor", type=int, default=20_000)
    parser.add_argument("--ply-min-weight", type=int, default=2_000)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init", help="toolkit checkpoint dir to start from")
    args = parser.parse_args()

    mx.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    corpora = [load_corpus([d]) for d in args.train]
    samplers = [PlySampler(b, args.ply_floor, args.seed + i, args.ply_min_weight) for i, (b, _) in enumerate(corpora)]
    sizes = np.array([len(b) for b, _ in corpora], dtype=np.float64)
    share = np.array(args.weights, dtype=np.float64) if args.weights else sizes
    share = share / share.sum()

    vb = np.load(Path(args.val) / "board.npy", mmap_mode="r")
    vs = np.load(Path(args.val) / "scores.npy", mmap_mode="r")
    pick = np.sort(np.random.default_rng(1).permutation(len(vb))[: args.val_limit])
    vb, vs = np.asarray(vb[pick]), np.asarray(vs[pick])

    if args.init:
        torch_model, _, config = load_checkpoint(Path(args.init), "cpu")
    else:
        config = config_for(args.size)
        torch_model, _ = make_system(config, "cpu")
    params = X.from_torch(torch_model.state_dict())
    n_params = parameter_count(torch_model)
    option_ids = mx.array(E.option_ids_table().astype(np.int32))

    def loss_fn(p, ctx, legal, scores):
        logits = X.forward(p, config, ctx, option_ids, legal)
        t = targets(scores, args.loss, args.tau)
        logp = logits - mx.logsumexp(logits, axis=1, keepdims=True)
        return -(t * mx.where(legal, logp, 0.0)).sum(1).mean()

    def lr_at(step: int) -> float:
        warm = min(1.0, (step + 1) / args.warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))
        return args.lr * warm * (args.min_lr_frac + (1 - args.min_lr_frac) * cos)

    optimizer = optim.Adam(learning_rate=args.lr, betas=[0.9, 0.95])
    grad_fn = mx.value_and_grad(loss_fn)
    decay_mask = {k: decays(k, v) for k, v in params.items()}

    # Only the loss/gradient graph is compiled; the learning rate stays outside it (a Python float
    # passed into a compiled function is a constant, and changing it every step recompiles every
    # step - measured at 51 s/step before this was fixed, 0.36 s/step after, M model, M1 Max).
    step_fn = mx.compile(grad_fn)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "train-log.jsonl").open("a")
    meta = {"trainer": "mlx", "args": vars(args), "params": n_params, "config": config,
            "train_rows": [int(s) for s in sizes], "share": share.tolist()}
    print(json.dumps(meta), flush=True)
    log.write(json.dumps(meta) + "\n")
    log.flush()

    def evaluate(p) -> dict:
        logits = []
        for i in range(0, len(vb), 8192):
            ctx, _, legal = X.inputs(vb[i:i + 8192])
            logits.append(np.array(X.forward(p, config, ctx, option_ids, legal)))
        return move_metrics(np.concatenate(logits), vb, vs)

    best_vp, running, started = -1.0, None, time.time()
    for step in range(args.steps):
        which = rng.choice(len(corpora), size=args.batch, p=share)
        boards, scores = [], []
        for c in range(len(corpora)):
            n = int((which == c).sum())
            if n:
                idx = np.sort(samplers[c](n))
                boards.append(np.asarray(corpora[c][0][idx]))
                scores.append(np.asarray(corpora[c][1][idx]))
        b = np.concatenate(boards)
        s = np.concatenate(scores)
        flip = rng.random(len(b)) < 0.5
        b[flip] = E.mirror_boards(b[flip])
        s[flip] = s[flip][:, ::-1]
        ctx, _, legal = X.inputs(b)
        lr = lr_at(step)
        optimizer.learning_rate = lr
        loss, grads = step_fn(params, ctx, legal, mx.array(s))
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        # decoupled weight decay on matrices only (not embeddings, norms, biases) - as train_c4.py
        params = {k: (v * (1 - lr * args.wd) if decay_mask[k] else v) for k, v in params.items()}
        params = optimizer.apply_gradients(grads, params)
        mx.eval(params, optimizer.state, loss)
        value = loss.item()
        running = value if running is None else 0.98 * running + 0.02 * value
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            report = evaluate(params)
            report.update({"step": step + 1, "loss": round(running, 4), "lr": lr, "elapsed_s": round(time.time() - started, 1)})
            if report["value_preserving"] > best_vp:
                best_vp = report["value_preserving"]
                report["parity"] = save(out / "model", params, config,
                                        {"step": step + 1, "value_preserving": best_vp, "trainer": "mlx",
                                         "train": [Path(t).name for t in args.train]}, vb[:512])
            save(out / "last", params, config, {"step": step + 1, "trainer": "mlx"}, vb[:64])
            print(json.dumps(report), flush=True)
            log.write(json.dumps(report) + "\n")
            log.flush()
    print(json.dumps({"done": True, "best_value_preserving": best_vp}), flush=True)


if __name__ == "__main__":
    main()
