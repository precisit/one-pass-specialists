"""Numerical check: the export-friendly forward pass (no in-place bool assignment,
required by coremltools) must reproduce the vendored model's logits exactly on real rows.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


from ._vendor import ensure_vendor

ensure_vendor()  # puts the vendored MIT upstream code on sys.path (see THIRD_PARTY_NOTICES.md)

from cua_s1.model import ChoiceExample, TinyTransformerScorer, load_checkpoint  # noqa: E402


def export_forward(self, batch, shuffle_context: bool = False):
    """Upstream math with the two in-place mask assignments replaced by an explicit OR
    with column 0 (`torch.logical_or`, since the coremltools torch frontend does not
    implement `__or__` on bool tensors)."""
    context_mask = batch["context_mask"]
    first_column = torch.arange(context_mask.shape[1], device=context_mask.device).unsqueeze(0) == 0
    safe_context_mask = torch.logical_or(context_mask, first_column)
    context = self.encoder(
        self._embed(batch["context_ids"]), src_key_padding_mask=torch.logical_not(safe_context_mask)
    )

    option_ids = batch["option_ids"]
    batch_size, option_count, token_count = option_ids.shape
    flat_ids = option_ids.reshape(batch_size * option_count, token_count)
    flat_mask = batch["option_token_mask"].reshape(batch_size * option_count, token_count)
    first_token = torch.arange(token_count, device=flat_mask.device).unsqueeze(0) == 0
    safe_mask = torch.logical_or(flat_mask, first_token)
    hidden = self.option_encoder(self._embed(flat_ids), src_key_padding_mask=torch.logical_not(safe_mask))
    weights = flat_mask.unsqueeze(-1).float()
    denominator = weights.sum(1).clamp_min(torch.tensor(1.0, dtype=torch.float32))
    pooled = (hidden * weights).sum(1) / denominator
    options = pooled.reshape(batch_size, option_count, -1)
    return self.head(context, context_mask, options, batch["option_mask"], shuffle_context)


def main() -> None:
    rows = [
        json.loads(line)
        for line in Path(os.environ.get("ONEPASS_TEST_ROWS", "data/sv/test.jsonl")).open(encoding="utf-8")
        if line.strip()
    ][:256]
    model, collator, _ = load_checkpoint(Path("runs/sv-tinyx/model"), "cpu")
    examples = [ChoiceExample(context=r["context"], options=tuple(r["options"]), label=r["label"]) for r in rows]
    batch = collator(examples)

    model.eval()
    with torch.no_grad():
        original = model(batch)

    TinyTransformerScorer.forward = export_forward
    with torch.no_grad():
        patched = model(batch)

    difference = (original - patched).abs().max().item()
    agreement = float((original.argmax(-1) == patched.argmax(-1)).float().mean())
    print(
        json.dumps(
            {
                "rows": len(rows),
                "max_abs_logit_difference": difference,
                "argmax_agreement": agreement,
                "identical": difference == 0.0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
