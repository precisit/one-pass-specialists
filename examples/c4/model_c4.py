"""The toolkit's one-pass scorer, fed with the v2 encoding. Shared by training, eval and protocol.

The architecture is `TinyTransformerScorer` from this repository's vendored model
(vendor/cua_s1/model.py), imported unchanged: only its config (width, layers, heads, token limits)
is chosen here. That is the point of the example - the one-pass contract itself, not a Connect Four
network wearing its name.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

import encode_c4 as E

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from onepass._vendor import ensure_vendor  # noqa: E402

ensure_vendor()

warnings.filterwarnings("ignore", message="enable_nested_tensor")
from cua_s1.model import load_checkpoint, make_system, parameter_count, save_checkpoint  # noqa: E402

SIZES = {
    "S": {"width": 128, "rank": 128, "layers": 2, "heads": 4},
    "M": {"width": 192, "rank": 192, "layers": 6, "heads": 6},
    "L": {"width": 256, "rank": 256, "layers": 8, "heads": 8},
}


def config_for(size: str, dropout: float = 0.0) -> dict:
    return {"encoder": "tinyx", **SIZES[size], "dropout": dropout,
            "context_tokens": E.CTX_BYTES, "option_tokens": E.OPTION_BYTES}


class Batcher:
    """Boards -> the model's batch dict, on device. Every option slot is 'column k' at slot k-1;
    full columns are masked (the scorer has no cross-option interaction, so slot order is inert -
    the option-order invariance control checks this)."""

    def __init__(self, device) -> None:
        self.device = device
        lut = torch.zeros(3, dtype=torch.long)
        lut[:] = torch.tensor([ord("."), ord("m"), ord("t")])
        self.cell_lut = (lut + 1).to(device)
        self.options = torch.from_numpy(E.option_ids_table()).to(device)

    def __call__(self, board: torch.Tensor) -> dict:
        board = board.to(self.device, non_blocking=True).long()
        n = board.shape[0]
        first = (board > 0).sum(1) % 2 == 0
        ctx = torch.empty((n, E.CTX_BYTES), dtype=torch.long, device=self.device)
        ctx[:, 0] = torch.where(first, ord("1") + 1, ord("2") + 1)
        ctx[:, 1] = ord(":") + 1
        ctx[:, 2:] = self.cell_lut[board]
        option_ids = self.options.unsqueeze(0).expand(n, -1, -1)
        legal = board.view(n, 6, 7)[:, 5, :] == 0
        return {"context_ids": ctx, "context_mask": ctx.ne(0), "option_ids": option_ids,
                "option_token_mask": option_ids.ne(0), "option_mask": legal}


class Scorer:
    """A checkpoint as a function boards[N,42] -> logits[N,7] (-inf on full columns)."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        self.model, _, self.config = load_checkpoint(Path(checkpoint), device)
        self.model.eval()
        self.batcher = Batcher(torch.device(device))
        self.device = device

    @torch.no_grad()
    def logits(self, boards: np.ndarray, batch: int = 4096) -> np.ndarray:
        out = []
        boards = np.asarray(boards, dtype=np.uint8).reshape(-1, 42)
        for i in range(0, len(boards), batch):
            b = self.batcher(torch.from_numpy(boards[i:i + batch]))
            out.append(self.model(b).float().cpu().numpy())
        logits = np.concatenate(out) if out else np.zeros((0, 7), dtype=np.float32)
        return np.where(E.legal_mask(boards), logits, -np.inf)
