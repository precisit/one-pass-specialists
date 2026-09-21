"""Small, deterministic Core ML accuracy probe, not a latency benchmark.

Run with the toolkit's Python environment on macOS. See coreml-follow-up.md for
artifact revisions and invocation. All inputs are public synthetic test rows.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=512)
    parser.add_argument("--decompress", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.toolkit.resolve()))

    import coremltools as ct
    import numpy as np
    import torch
    from onepass._vendor import ensure_vendor
    from onepass.encode import InputLimits, prepare_inputs

    ensure_vendor()
    from cua_s1.model import ChoiceExample, load_checkpoint

    torch.set_num_threads(2)
    raw = gzip.decompress(args.test.read_bytes()) if args.test.suffix == ".gz" else args.test.read_bytes()
    lines = raw.splitlines()
    if not 0 < args.sample <= len(lines):
        raise ValueError("sample must be positive and no larger than the corpus")
    indices = np.linspace(0, len(lines) - 1, num=args.sample, dtype=int).tolist()
    rows = [json.loads(lines[i]) for i in indices]
    gold = np.array([row["label"] for row in rows])
    model, collator, _ = load_checkpoint(args.checkpoint, "cpu")
    model.eval()
    baseline_logits = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            chunk = rows[start:start + 32]
            batch = collator([ChoiceExample(context=r["context"], options=tuple(r["options"]), label=r["label"]) for r in chunk])
            scores = model(batch).float().numpy()
            baseline_logits.extend(scores[i, :len(row["options"])].copy() for i, row in enumerate(chunk))
    if not all(np.isfinite(score).all() for score in baseline_logits):
        raise ValueError("nonfinite PyTorch logits")
    baseline = np.array([int(score.argmax()) for score in baseline_logits])
    del model
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {"chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
                     "macos": platform.mac_ver()[0], "build": command("sw_vers", "-buildVersion"),
                     "architecture": platform.machine(), "python": platform.python_version(),
                     "coremltools": ct.__version__, "torch": torch.__version__, "numpy": np.__version__},
        "toolkit_revision": command("git", "-C", str(args.toolkit), "rev-parse", "HEAD"),
        "probe_sha256": sha256(Path(__file__).read_bytes()),
        "test_plaintext_sha256": sha256(raw), "corpus_rows": len(lines),
        "checkpoint_sha256": sha256(args.checkpoint.read_bytes()),
        "sample_method": "numpy.linspace(0, corpus_rows - 1, num=sample, dtype=int)",
        "sample_indices": indices, "sample_rows": len(rows),
        "notes": ["Accuracy probe only; no timing claims.",
                  "No warm-up rows excluded. All real-option logits checked for finiteness.",
                  "CPU_AND_NE permits CPU and Neural Engine; actual placement not measured.",
                  "Decompression restores dense constants from rounded weights, not original weights."],
        "baseline": {"correct": int((baseline == gold).sum()), "accuracy": float((baseline == gold).mean()),
                     "predictions": baseline.tolist()},
        "packages": {}, "results": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def describe(package):
        unloaded = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
        spec = unloaded.get_spec()
        counts = Counter()
        for function in spec.mlProgram.functions.values():
            for block in function.block_specializations.values():
                counts.update(op.type for op in block.operations)
        files = {str(p.relative_to(package)): sha256(p.read_bytes()) for p in sorted(package.rglob("*")) if p.is_file()}
        return {"files_sha256": files, "file_bytes": sum(p.stat().st_size for p in package.rglob("*") if p.is_file()),
                "specification_version": spec.specificationVersion, "operation_counts": dict(sorted(counts.items()))}

    print(f"PyTorch: {result['baseline']['correct']}/{len(rows)}", flush=True)
    packages = [(name, args.packages / f"sv0_forms_{name}_options40.mlpackage") for name in ("fp16", "int8", "int4")]
    if args.decompress:
        print("Decompressing the published int4 package without retraining or re-quantizing", flush=True)
        compressed = ct.models.MLModel(str(packages[-1][1]), compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
        dense = ct.optimize.coreml.decompress_weights(compressed)
        dense_path = args.output.parent / "int4-decompressed.mlpackage"
        dense.save(str(dense_path))
        del dense, compressed
        packages.append(("int4_decompressed", dense_path))
    limits = InputLimits(context_bytes=224, option_bytes=96, max_options=40)
    inputs = [prepare_inputs(row["context"], row["options"], limits) for row in rows]
    for name, package in packages:
        result["packages"][name] = describe(package)
        cpu_logits = None
        cpu_predictions = None
        for unit in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_NE):
            key = f"{name}__{unit.name}"
            print(f"Loading {key}", flush=True)
            model = ct.models.MLModel(str(package), compute_units=unit)
            logits = [np.asarray(model.predict(encoded)["logits"]).reshape(-1)[:len(row["options"])].copy()
                      for row, encoded in zip(rows, inputs, strict=True)]
            if not all(np.isfinite(score).all() for score in logits):
                raise ValueError(f"nonfinite real-option logits in {key}")
            predictions = np.array([int(score.argmax()) for score in logits])
            changed = predictions != baseline
            entry = {"correct": int((predictions == gold).sum()), "accuracy": float((predictions == gold).mean()),
                     "mismatches_vs_torch": int(changed.sum()),
                     "changed_corpus_indices": [indices[i] for i in np.flatnonzero(changed)],
                     "predictions": predictions.tolist(), "all_real_option_logits_finite": True,
                     "max_abs_logit_difference_vs_torch": float(max(np.abs(a - b).max() for a, b in zip(logits, baseline_logits, strict=True)))}
            if cpu_predictions is not None:
                entry["mismatches_vs_cpu_only"] = int((predictions != cpu_predictions).sum())
                entry["max_abs_logit_difference_vs_cpu_only"] = float(max(np.abs(a - b).max() for a, b in zip(logits, cpu_logits, strict=True)))
            else:
                cpu_logits, cpu_predictions = logits, predictions
            result["results"][key] = entry
            save()
            print(f"{key}: correct={entry['correct']}/{len(rows)}, torch mismatches={entry['mismatches_vs_torch']}", flush=True)
            del model


if __name__ == "__main__":
    main()
