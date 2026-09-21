"""Evaluate the Swedish specialist and the English reference on the same rows.

Reports aggregate top-1, per-action accuracy, ECE, the shuffled-context control
(an option-statistics / majority-class floor), how often a field that should be
filled was silently skipped, and the hand-authored out-of-distribution demo set.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ._vendor import ensure_vendor

ensure_vendor()  # puts the vendored MIT upstream code on sys.path (see THIRD_PARTY_NOTICES.md)

import torch  # noqa: E402
from cua_s1.model import ChoiceExample, load_checkpoint, parameter_count, select_device  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

DATASETS = {
    "synthetic_test": Path("data/sv/test.jsonl"),
    "handwritten_demo": Path("data/sv/demo-handwritten.jsonl"),
}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@torch.no_grad()
def score(model, collator, rows: list[dict], device, batch_size: int = 128, shuffle_context: bool = False) -> dict:
    model.eval()
    examples = [
        ChoiceExample(context=row["context"], options=tuple(row["options"]), label=row["label"]) for row in rows
    ]
    loader = DataLoader(examples, batch_size=batch_size, collate_fn=collator)
    per_action: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confidences: list[torch.Tensor] = []
    corrects: list[torch.Tensor] = []
    silent_skip = 0
    offset = 0
    for host_batch in loader:
        batch = {key: value.to(device) for key, value in host_batch.items()}
        logits = model(batch, shuffle_context=shuffle_context)
        probabilities = logits.softmax(-1)
        confidence, prediction = probabilities.max(-1)
        correct = prediction.eq(batch["labels"])
        for row_index in range(correct.shape[0]):
            row = rows[offset + row_index]
            action = row["meta"].get("action", "fill")
            per_action[action][0] += int(correct[row_index])
            per_action[action][1] += 1
            if action == "fill" and not correct[row_index]:
                if row["options"][int(prediction[row_index])] in ("hoppa över", "skip"):
                    silent_skip += 1
        offset += correct.shape[0]
        confidences.append(confidence.cpu())
        corrects.append(correct.cpu())
    confidence = torch.cat(confidences)
    correct = torch.cat(corrects).float()
    ece = 0.0
    for lower in torch.linspace(0, 0.9, 10):
        selected = (confidence >= lower) & (confidence < lower + 0.1)
        if selected.any():
            ece += float(selected.float().mean() * (correct[selected].mean() - confidence[selected].mean()).abs())
    actions = Counter(row["meta"].get("action", "fill") for row in rows)
    return {
        "examples": len(rows),
        "top1": round(float(correct.mean()), 5),
        "ece": round(ece, 6),
        "majority_baseline": round(max(actions.values()) / len(rows), 5),
        "action_mix": dict(actions),
        "skipped_when_fill_expected": silent_skip,
        "per_action": {
            action: {"acc": round(hits / total, 5), "n": total} for action, (hits, total) in sorted(per_action.items())
        },
    }


def resolve_checkpoint(prefix: Path) -> Path:
    """Accept a directory, a <name>.safetensors file, or a bare <name> prefix."""
    if prefix.is_dir():
        return prefix
    if prefix.suffix == ".safetensors":
        return prefix
    candidate = Path(f"{prefix}.safetensors")
    return candidate if candidate.exists() else prefix


def evaluate_checkpoint(prefix: Path, label: str, device, datasets: dict[str, Path]) -> dict:
    resolved = resolve_checkpoint(prefix)
    model, collator, config = load_checkpoint(resolved, device)
    weights = resolved if resolved.is_file() else resolved / "model.safetensors"
    report: dict[str, object] = {
        "label": label,
        "checkpoint": str(weights),
        "checkpoint_bytes": weights.stat().st_size if weights.exists() else None,
        "config": config,
        "trainable_params": parameter_count(model),
    }
    for name, path in datasets.items():
        rows = read_rows(path)
        report[name] = score(model, collator, rows, device)
        report[f"{name}__shuffled_context"] = score(model, collator, rows, device, shuffle_context=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="checkpoint directory, <name>.safetensors file, or a bare <name> prefix",
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=None,
        help="optional second checkpoint scored on exactly the same rows (e.g. another language)",
    )
    parser.add_argument("--label", default="specialist")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data", type=Path, default=Path("data/sv"), help="directory holding test.jsonl / demo-handwritten.jsonl")
    parser.add_argument("--out", type=Path, default=Path("RESULTS-eval.json"))
    args = parser.parse_args()

    datasets = {
        "synthetic_test": args.data / "test.jsonl",
        "handwritten_demo": args.data / "demo-handwritten.jsonl",
    }
    missing = [str(path) for path in datasets.values() if not path.exists()]
    if missing:
        raise SystemExit(f"missing dataset(s): {', '.join(missing)} — generate a corpus first (python -m onepass.synth)")

    device = select_device(args.device)
    report: dict[str, object] = {
        "device": str(device),
        "datasets": {name: str(path) for name, path in datasets.items()},
        "specialist": evaluate_checkpoint(args.checkpoint, args.label, device, datasets),
    }
    if args.reference_checkpoint is not None:
        report["reference"] = evaluate_checkpoint(
            args.reference_checkpoint, f"reference:{args.reference_checkpoint.name}", device, datasets
        )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
