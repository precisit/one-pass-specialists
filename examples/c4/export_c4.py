"""Export a v2 checkpoint to ONNX at the v2 shape (context 44, 7 options x 8 bytes) and gate it.

Inputs (int32): context_ids [1, 44], option_ids [1, 7, 8], option_mask [1, 7]; output: logits
[1, 7]. The same input/output contract as the first version, so the browser runtime is unchanged.

Gates (printed as JSON; the export is refused unless both hold):
  * argmax parity PyTorch vs ONNX Runtime (CPU) on N eval positions == 1.0;
  * cross-shape: the same positions with the options compacted to the legal columns only (what the
    browser sends) give the same choices as the fixed column slots used in training.
Also writes `selftest-v2.json`: reference positions with sha256 prefixes of the encoded buffers,
for the browser's encoder self-test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import encode_c4 as E
from model_c4 import Scorer


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, context_ids, option_ids, option_mask):
        batch = {"context_ids": context_ids, "context_mask": context_ids.ne(0), "option_ids": option_ids,
                 "option_token_mask": option_ids.ne(0), "option_mask": option_mask.ne(0)}
        return self.model(batch).float()


def compact_inputs(board: np.ndarray):
    """What the browser sends: context bytes + the legal options packed from slot 0."""
    ctx = (E.contexts(board.reshape(1, 42)).astype(np.int32) + 1)
    legal = [c for c in range(7) if board.reshape(6, 7)[5, c] == 0]
    table = E.option_ids_table().astype(np.int32)
    options = np.zeros((1, 7, E.OPTION_BYTES), dtype=np.int32)
    mask = np.zeros((1, 7), dtype=np.int32)
    for slot, col in enumerate(legal):
        options[0, slot] = table[col]
        mask[0, slot] = 1
    return ctx, options, mask, legal


def sha32(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:32]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evalset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()

    scorer = Scorer(args.checkpoint, "cpu")
    wrapper = ExportWrapper(scorer.model).eval()
    example = (torch.zeros((1, E.CTX_BYTES), dtype=torch.int32),
               torch.zeros((1, E.MAX_OPTIONS, E.OPTION_BYTES), dtype=torch.int32),
               torch.zeros((1, E.MAX_OPTIONS), dtype=torch.int32))
    out = Path(args.out)
    torch.onnx.export(wrapper, example, str(out), input_names=["context_ids", "option_ids", "option_mask"],
                      output_names=["logits"], opset_version=17, dynamic_axes=None, dynamo=False)

    import onnxruntime as ort

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    boards = np.load(Path(args.evalset) / "board.npy")
    pick = np.random.default_rng(3).permutation(len(boards))[: args.n]
    boards = boards[pick]
    torch_choice = scorer.logits(boards).argmax(1)
    agree = cross = 0
    max_delta = 0.0
    fixed = scorer.logits(boards)
    for i, board in enumerate(boards):
        ctx, options, mask, legal = compact_inputs(board)
        logits = session.run(None, {"context_ids": ctx, "option_ids": options, "option_mask": mask})[0][0]
        choice = legal[int(np.argmax(logits[: len(legal)]))]
        agree += choice == torch_choice[i]
        max_delta = max(max_delta, float(np.max(np.abs(logits[: len(legal)] - fixed[i][legal]))))
        cross += choice == int(np.argmax(fixed[i]))
    report = {"onnx": out.name, "bytes": out.stat().st_size, "n": len(boards),
              "argmax_parity": agree / len(boards), "cross_shape_parity": cross / len(boards),
              "max_logit_delta": round(max_delta, 6)}
    print(json.dumps(report), flush=True)
    if report["argmax_parity"] < 1.0:
        out.unlink()
        raise SystemExit("parity gate failed - export removed")

    # browser self-test reference rows: a few positions across the game
    rows = []
    for i in np.linspace(0, len(boards) - 1, 8).astype(int):
        board = boards[i]
        ctx, options, mask, legal = compact_inputs(board)
        rows.append({"name": f"eval-{int(pick[i])}", "board": board.tolist(), "context": E.context_text(board),
                     "legal": legal, "hashes": {"context": sha32(ctx), "options": sha32(options), "mask": sha32(mask)}})
    (out.parent / "selftest-v2.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
