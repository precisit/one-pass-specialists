"""Measure exported Core ML packages against the PyTorch checkpoint they came from.

Metrics: top-1, per-action accuracy, **silent skips** (a required fill answered `hoppa över`),
argmax parity with PyTorch, median/p95 latency with and without the Neural Engine, and package
size. Same rows, same gold, same grader as `onepass.evaluate`, so the numbers are comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from ._vendor import ensure_vendor

ensure_vendor()  # puts the vendored MIT upstream code on sys.path (see THIRD_PARTY_NOTICES.md)

from cua_s1.model import ChoiceExample, load_checkpoint  # noqa: E402

from .encode import InputLimits, prepare_inputs  # noqa: E402

SKIP_OPTION = "hoppa över"


def variant_paths(output_dir: Path, name: str, max_options: int) -> dict[str, Path]:
    return {
        variant: output_dir / f"{name}_{variant}_options{max_options}.mlpackage"
        for variant in ("fp16", "int8", "int4")
    }


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def package_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def torch_predictions(checkpoint: Path, rows: list[dict], batch_size: int = 512) -> list[int]:
    """Predictions for the whole split, in chunks.

    Chunking is not an optimisation: materialising a 30 000-row split as a single batch builds
    ~1 GB of int64 tensors before the model even runs, which is enough to have the process killed
    on a 24 GB laptop. The held-out splits grow with the corpus, so this has to scale.
    """
    model, collator, _ = load_checkpoint(checkpoint, "cpu")
    model.eval()
    predictions: list[int] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        examples = [ChoiceExample(context=r["context"], options=tuple(r["options"]), label=r["label"]) for r in chunk]
        batch = collator(examples)
        with torch.no_grad():
            predictions.extend(model(batch).argmax(-1).tolist())
    return predictions


def score(predictions: list[int], rows: list[dict], baseline: list[int] | None = None) -> dict:
    correct = sum(1 for row, prediction in zip(rows, predictions) if prediction == row["label"])
    per_action: dict[str, list[int]] = {}
    silent_skips = 0
    for row, prediction in zip(rows, predictions):
        action = row["meta"].get("action", "fill")
        per_action.setdefault(action, [0, 0])
        per_action[action][0] += int(prediction == row["label"])
        per_action[action][1] += 1
        if action == "fill" and prediction != row["label"] and row["options"][prediction] == SKIP_OPTION:
            silent_skips += 1
    result = {
        "examples": len(rows),
        "top1": correct / len(rows),
        "skipped_when_fill_expected": silent_skips,
        "silent_skip_rate_of_fills": silent_skips / per_action.get("fill", [0, 1])[1],
        "per_action": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(per_action.items())},
    }
    if baseline is not None:
        result["parity_with_torch"] = sum(1 for a, b in zip(predictions, baseline) if a == b) / len(predictions)
        result["mismatches_vs_torch"] = sum(1 for a, b in zip(predictions, baseline) if a != b)
    return result


def run_variant(
    package: Path,
    limits: InputLimits,
    rows: list[dict],
    baseline: list[int],
    compute: ct.ComputeUnit,
    warmup: int = 3,
) -> dict:
    model = ct.models.MLModel(str(package), compute_units=compute)
    predictions: list[int] = []
    latencies: list[float] = []
    for index, row in enumerate(rows):
        inputs = prepare_inputs(row["context"], tuple(row["options"]), limits)
        if index < warmup:
            model.predict(inputs)
            continue
        start = time.perf_counter()
        output = model.predict(inputs)
        latencies.append((time.perf_counter() - start) * 1000.0)
        logits = np.asarray(output["logits"]).reshape(-1)
        predictions.append(int(logits[: len(row["options"])].argmax()))
    scored = score(predictions, rows[warmup:], baseline[warmup:])
    scored.update(
        {
            "compute_units": str(compute),
            "latency_ms_median": statistics.median(latencies),
            "latency_ms_p95": sorted(latencies)[int(0.95 * len(latencies))],
        }
    )
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="the PyTorch checkpoint the packages came from")
    parser.add_argument("--packages", type=Path, default=Path("sv-coreml"), help="directory holding the .mlpackage variants")
    parser.add_argument("--data", type=Path, default=Path("data/sv"), help="directory holding test.jsonl / demo-handwritten.jsonl")
    parser.add_argument("--name", default="one_pass_forms", help="package name prefix used at export time")
    parser.add_argument("--max-options", type=int, default=40, help="option ceiling the packages were exported with")
    parser.add_argument("--out", type=Path, default=Path("RESULTS-coreml.json"))
    args = parser.parse_args()

    limits = InputLimits(context_bytes=224, option_bytes=96, max_options=args.max_options)
    variants = variant_paths(args.packages, args.name, args.max_options)
    missing = [str(path) for path in variants.values() if not path.exists()]
    if missing:
        raise SystemExit(f"missing exported package(s): {', '.join(missing)} — run onepass.export_coreml first")
    datasets = {
        "synthetic_test": args.data / "test.jsonl",
        "handwritten_demo": args.data / "demo-handwritten.jsonl",
    }

    test_rows = read_rows(datasets["synthetic_test"])
    baseline = torch_predictions(args.checkpoint, test_rows)
    report: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "limits": {"context_bytes": 224, "option_bytes": 96, "max_options": args.max_options},
        "torch_baseline": score(baseline, test_rows),
        "variants": {},
    }
    for variant, package in variants.items():
        entry: dict[str, object] = {
            "package_bytes": package_bytes(package),
            "synthetic_test__ane": run_variant(package, limits, test_rows, baseline, ct.ComputeUnit.CPU_AND_NE),
        }
        if variant == "fp16":
            entry["synthetic_test__cpu"] = run_variant(package, limits, test_rows, baseline, ct.ComputeUnit.CPU_ONLY)
        if datasets["handwritten_demo"].exists():
            demo_rows = read_rows(datasets["handwritten_demo"])
            demo_baseline = torch_predictions(args.checkpoint, demo_rows)
            entry["handwritten_demo__ane"] = run_variant(
                package, limits, demo_rows, demo_baseline, ct.ComputeUnit.CPU_AND_NE
            )
        report["variants"][variant] = entry  # type: ignore[index]
        print(f"{variant}: {json.dumps(entry, sort_keys=True)[:300]}", flush=True)

    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
