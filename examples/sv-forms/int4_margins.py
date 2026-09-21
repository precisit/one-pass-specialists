"""Why does 4-bit palettisation destroy one checkpoint and not the other?

Hypothesis under test: the decision is an argmax over near-identical options, so what matters is
the *logit margin* (best minus second best). If a model's margins are small relative to the
palettisation error, 4 bits flip decisions even though "accuracy" looked fine.

Measured here on the same 2 000 held-out rows for three models:
  * our v2 fp16 checkpoint (PyTorch)
  * our v2 int4 Core ML package
  * the released English checkpoint of the same family (PyTorch)

Outputs: margin distribution, the fp16 margin of the rows int4 flips, and the accuracy of each.
"""

from __future__ import annotations

import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

SPIKE = Path("/Users/hermes-tool/builder-work/one-pass-specialists-67k19_tm")
TEST = SPIKE / "data" / "sv-10000" / "test.jsonl"
OUR_FP16 = SPIKE / "runs" / "sv-tinyx-10000" / "model"
OUR_INT4 = SPIKE / "sv-coreml-10000" / "one_pass_forms_int4_options40.mlpackage"
ENGLISH = Path("/tmp/cua1ckpt/cua-s1-forms.safetensors")
MAX_OPTIONS = 40
SAMPLE = 2000

import sys

sys.path.insert(0, str(SPIKE))
from onepass._vendor import ensure_vendor  # noqa: E402
from onepass.encode import InputLimits, byte_ids, choose  # noqa: E402

ensure_vendor()
from cua_s1.model import ChoiceExample, load_checkpoint  # noqa: E402


def load_rows(limit: int) -> list[dict]:
    rows = []
    with TEST.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index % 15 == 0:  # every 15th row, so the sample spans the whole split
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def torch_logits(checkpoint: Path, rows: list[dict]) -> list[np.ndarray]:
    """Per-row logits over that row's real options only (padding excluded)."""
    model, collator, _ = load_checkpoint(checkpoint, "cpu")
    model.eval()
    with torch.no_grad():
        logits = model(collator([ChoiceExample(context=r["context"], options=tuple(r["options"]), label=r["label"]) for r in rows]))
    matrix = logits.float().numpy()
    return [matrix[index, : len(row["options"])] for index, row in enumerate(rows)]


def coreml_logits(package: Path, rows: list[dict]) -> list[np.ndarray]:
    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    limits = InputLimits(context_bytes=224, option_bytes=96, max_options=MAX_OPTIONS)
    out: list[np.ndarray] = []
    for row in rows:
        logits = model.predict(encode_row(row, limits))["logits"][0][: len(row["options"])]
        out.append(np.asarray(logits, dtype=np.float32))
    return out


def margins(logits_rows: list[np.ndarray]) -> np.ndarray:
    """Best minus second best, per row, over that row's real options."""
    return np.array(
        [float(np.sort(row)[-1] - np.sort(row)[-2]) for row in logits_rows],
        dtype=np.float64,
    )


def summarise(name: str, logits: list[np.ndarray], rows: list[dict]) -> dict:
    gold = np.array([row["label"] for row in rows])
    predicted = np.array([int(np.asarray(row).argmax()) for row in logits])
    margin = margins(logits)
    correct = predicted == gold
    summary = {
        "model": name,
        "rows": len(rows),
        "top1": round(float(correct.mean()), 4),
        "margin_median": round(float(np.median(margin)), 3),
        "margin_p10": round(float(np.percentile(margin, 10)), 3),
        "share_margin_below_1_logit": round(float((margin < 1.0).mean()), 4),
    }
    print(json.dumps(summary), flush=True)
    return {"summary": summary, "predicted": predicted, "margin": margin, "gold": gold}


def encode_row(row: dict, limits: InputLimits) -> dict:
    context = np.zeros((1, limits.context_bytes), dtype=np.int32)
    ids = np.asarray(byte_ids(row["context"], limits.context_bytes), dtype=np.int32)
    context[0, : ids.shape[0]] = ids
    options = np.zeros((1, limits.max_options, limits.option_bytes), dtype=np.int32)
    for index, option in enumerate(row["options"][: limits.max_options]):
        option_ids = np.asarray(byte_ids(option, limits.option_bytes), dtype=np.int32)
        options[0, index, : option_ids.shape[0]] = option_ids
    mask = np.zeros((1, limits.max_options), dtype=np.int32)
    mask[0, : len(row["options"][: limits.max_options])] = 1
    return {"context_ids": context, "option_ids": options, "option_mask": mask}


def main() -> None:
    rows = load_rows(SAMPLE)
    print(f"sample: {len(rows)} rows, options per row median "
          f"{int(np.median([len(r['options']) for r in rows]))}", flush=True)

    our = summarise("ours-fp16", torch_logits(OUR_FP16, rows), rows)
    english = summarise("english-fp16", torch_logits(ENGLISH, rows), rows)
    quantised = summarise("ours-int4-coreml", coreml_logits(OUR_INT4, rows), rows)

    flips = quantised["predicted"] != our["predicted"]
    report = {
        "sample_rows": len(rows),
        "ours_fp16": our["summary"],
        "english_fp16": english["summary"],
        "ours_int4": quantised["summary"],
        "flips_vs_fp16": int(flips.sum()),
        "flip_rate": round(float(flips.mean()), 4),
        "fp16_margin_of_flipped_rows_median": round(float(np.median(our["margin"][flips])), 3) if flips.any() else None,
        "fp16_margin_of_kept_rows_median": round(float(np.median(our["margin"][~flips])), 3),
        "share_of_flips_with_margin_below_1_logit": round(float((our["margin"][flips] < 1.0).mean()), 4) if flips.any() else None,
        "accuracy_cost_of_flips": round(float((our["predicted"][~flips] == our["gold"][~flips]).mean()), 4),
    }
    out = SPIKE / "RESULTS-int4-margins.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
