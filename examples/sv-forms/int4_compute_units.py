"""Is the int4 collapse a property of the quantisation, or of the Neural Engine path?

Same rows, same encoding, same masked argmax; only the compute unit changes. If int4 is fine on
CPU and broken on CPU+ANE, then the finding is about the ANE compilation of the palettised graph,
not about 4-bit weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

SPIKE = Path("/Users/hermes-tool/builder-work/one-pass-specialists-67k19_tm")
TEST = SPIKE / "data" / "sv-10000" / "test.jsonl"
CHECKPOINT = SPIKE / "runs" / "sv-tinyx-10000" / "model"
PACKAGES = {
    "fp16": SPIKE / "sv-coreml-10000" / "one_pass_forms_fp16_options40.mlpackage",
    "int8": SPIKE / "sv-coreml-10000" / "one_pass_forms_int8_options40.mlpackage",
    "int4": SPIKE / "sv-coreml-10000" / "one_pass_forms_int4_options40.mlpackage",
}
MAX_OPTIONS = 40
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 6

sys.path.insert(0, str(SPIKE))
from onepass._vendor import ensure_vendor  # noqa: E402
from onepass.encode import InputLimits, prepare_inputs  # noqa: E402

ensure_vendor()
from cua_s1.model import ChoiceExample, load_checkpoint  # noqa: E402


def rows_sample() -> list[dict]:
    rows = []
    with TEST.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index % STRIDE == 0:
                rows.append(json.loads(line))
    return rows


def torch_predictions(rows: list[dict]) -> list[int]:
    model, collator, _ = load_checkpoint(CHECKPOINT, "cpu")
    model.eval()
    out: list[int] = []
    for start in range(0, len(rows), 512):
        chunk = rows[start : start + 512]
        batch = collator([ChoiceExample(context=r["context"], options=tuple(r["options"]), label=r["label"]) for r in chunk])
        with torch.no_grad():
            logits = model(batch).float().numpy()
        out.extend(int(logits[i, : len(row["options"])].argmax()) for i, row in enumerate(chunk))
    return out


def coreml_predictions(package: Path, rows: list[dict], compute: ct.ComputeUnit, limits: InputLimits) -> list[int]:
    model = ct.models.MLModel(str(package), compute_units=compute)
    out: list[int] = []
    for row in rows:
        logits = np.asarray(model.predict(prepare_inputs(row["context"], tuple(row["options"]), limits))["logits"]).reshape(-1)
        out.append(int(logits[: len(row["options"])].argmax()))
    return out


def main() -> None:
    rows = rows_sample()
    gold = np.array([row["label"] for row in rows])
    limits = InputLimits(context_bytes=224, option_bytes=96, max_options=MAX_OPTIONS)
    baseline = torch_predictions(rows)
    print(f"sample: {len(rows)} rows | pytorch top-1 {float((np.array(baseline) == gold).mean()):.4f}", flush=True)

    table: dict[str, dict] = {}
    for name, package in PACKAGES.items():
        for label, compute in (("cpu", ct.ComputeUnit.CPU_ONLY), ("ane", ct.ComputeUnit.CPU_AND_NE)):
            predictions = np.array(coreml_predictions(package, rows, compute, limits))
            entry = {
                "top1": round(float((predictions == gold).mean()), 4),
                "parity_with_torch": round(float((predictions == np.array(baseline)).mean()), 6),
                "mismatches": int((predictions != np.array(baseline)).sum()),
            }
            table[f"{name}__{label}"] = entry
            print(f"{name:>5} {label:>4}: top1={entry['top1']:.4f} parity={entry['parity_with_torch']:.6f} "
                  f"mismatches={entry['mismatches']}", flush=True)

    out = SPIKE / "RESULTS-int4-compute-units.json"
    out.write_text(json.dumps({"rows": len(rows), "pytorch_top1": round(float((np.array(baseline) == gold).mean()), 4), "table": table}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
