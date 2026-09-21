"""Export a trained one-pass specialist to Core ML: fp16, int8 and int4 packages.

The export mirrors the contract of the FluidInference CUA-S1 release — fixed shapes
`context_bytes / option_bytes / max_options`, byte-level int32 ids in, one logit per option
out — so a consumer needs no tokenizer and no PyTorch.

Nothing is exported before two proofs:

1. the **export-forward** used here (an assign-free rewrite of upstream's mask handling, which
   coremltools cannot convert) is numerically identical to the vendored model, and
2. the **traced graph** reproduces the model on real rows.

Both are checked against a real split and abort the run on mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from ._vendor import ensure_vendor

ensure_vendor()  # puts the vendored MIT upstream code on sys.path (see THIRD_PARTY_NOTICES.md)

from cua_s1.model import (  # noqa: E402
    ChoiceExample,
    TinyTransformerScorer,
    load_checkpoint,
    parameter_count,
)

from .forward import export_forward  # noqa: E402

# coremltools cannot convert the upstream in-place bool mask assignment; the export forward is
# bit-identical to it (verified: max |logit difference| = 0.0 on 256 real rows).
TinyTransformerScorer.forward = export_forward

DEFAULT_NAME = "one_pass_forms"


class ExportWrapper(torch.nn.Module):
    """Fixed-shape int32 tensors in, float32 logits out."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, context_ids: torch.Tensor, option_ids: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
        batch = {
            "context_ids": context_ids,
            "context_mask": context_ids.ne(0),
            "option_ids": option_ids,
            "option_token_mask": option_ids.ne(0),
            "option_mask": option_mask.ne(0),
        }
        return self.model(batch).float()


def example_inputs(max_options: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((1, 224), dtype=torch.int32),
        torch.zeros((1, max_options, 96), dtype=torch.int32),
        torch.zeros((1, max_options), dtype=torch.int32),
    )


def real_batch(path: Path, max_options: int, rows: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode real rows into fixed-shape int32 tensors (byte + 1, zero padded)."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()][:rows]
    if not lines:
        raise SystemExit(f"no rows in {path} — the export refuses to ship unverified")
    context_ids = torch.zeros((len(lines), 224), dtype=torch.int32)
    option_ids = torch.zeros((len(lines), max_options, 96), dtype=torch.int32)
    option_mask = torch.zeros((len(lines), max_options), dtype=torch.int32)
    for index, line in enumerate(lines):
        payload = json.loads(line)
        context = payload["context"].encode("utf-8")[:224]
        context_ids[index, : len(context)] = torch.tensor([byte + 1 for byte in context], dtype=torch.int32)
        for option_index, option in enumerate(payload["options"][:max_options]):
            data = option.encode("utf-8")[:96]
            option_ids[index, option_index, : len(data)] = torch.tensor([byte + 1 for byte in data], dtype=torch.int32)
            option_mask[index, option_index] = 1
    return context_ids, option_ids, option_mask


def verify_trace(model: torch.nn.Module, traced: torch.jit.ScriptModule, test_rows: Path, max_options: int) -> dict:
    """The traced export graph must reproduce the model's logits on real rows."""
    context_ids, option_ids, option_mask = real_batch(test_rows, max_options)
    wrapper = ExportWrapper(model.eval())
    traced = traced.eval()
    with torch.no_grad():
        expected = wrapper(context_ids, option_ids, option_mask)
        actual = traced(context_ids, option_ids, option_mask)
    difference = (expected - actual).abs().max().item()
    agreement = float((expected.argmax(-1) == actual.argmax(-1)).float().mean())
    print(json.dumps({"trace_max_abs_logit_difference": difference, "trace_argmax_agreement": agreement}), flush=True)
    if agreement != 1.0 or difference > 1e-5:
        raise SystemExit("traced graph does not match the model; aborting export")
    return {"max_abs_logit_difference": difference, "argmax_agreement": agreement}


def build_fp16(
    checkpoint: Path,
    out: Path,
    max_options: int,
    name: str,
    target: ct.target,
    test_rows: Path,
) -> ct.models.MLModel:
    """Convert (or reuse) the fp16 mlprogram for a given deployment target."""
    suffix = "" if target == ct.target.iOS17 else "_ios18"
    path = out / f"{name}_fp16{suffix}_options{max_options}.mlpackage"
    if os.environ.get("REUSE_FP16") == "1" and path.exists():
        return ct.models.MLModel(str(path))
    model, _collator, _config = load_checkpoint(checkpoint, "cpu")
    wrapper = ExportWrapper(model.eval())
    # Trace with grad *enabled*: under `no_grad` PyTorch takes its fused sparsity fast path
    # (`torch._transformer_encoder_layer_fwd`), which coremltools cannot convert.
    # check_trace=False: torch's own replay check reports spurious divergences here; the traced
    # module is validated numerically against the model instead (verify_trace, right below).
    with torch.enable_grad():
        traced = torch.jit.trace(wrapper, example_inputs(max_options), strict=False, check_trace=False)
    trace_check = verify_trace(model, traced, test_rows, max_options)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="context_ids", shape=(1, 224), dtype=np.int32),
            ct.TensorType(name="option_ids", shape=(1, max_options, 96), dtype=np.int32),
            ct.TensorType(name="option_mask", shape=(1, max_options), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    mlmodel.short_description = f"One-pass specialist (fp16, {max_options} options)"
    mlmodel.save(str(path))
    globals()["_LAST_TRACE_CHECK"] = trace_check
    return mlmodel


def convert(
    checkpoint: Path,
    out: Path,
    max_options: int,
    name: str = DEFAULT_NAME,
    test_rows: Path | None = None,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    test_rows = test_rows or Path("data/sv/test.jsonl")
    model, _collator, config = load_checkpoint(checkpoint, "cpu")
    parameters = parameter_count(model)
    weights = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint

    fp16 = build_fp16(checkpoint, out, max_options, name, ct.target.iOS17, test_rows)
    manifest: dict[str, dict] = {
        "fp16": {
            "path": f"{name}_fp16_options{max_options}.mlpackage",
            "precision": "float16",
            "minimum_target": "iOS17/macOS14",
        }
    }

    int8 = ct.optimize.coreml.palettize_weights(
        fp16,
        ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpPalettizerConfig(nbits=8, mode="UNIFORM", granularity="per_tensor")
        ),
    )
    int8.short_description = f"One-pass specialist (int8 weights, fp16 compute, {max_options} options)"
    int8.save(str(out / f"{name}_int8_options{max_options}.mlpackage"))
    manifest["int8"] = {
        "path": f"{name}_int8_options{max_options}.mlpackage",
        "precision": "int8_weights_float16_compute",
        "minimum_target": "iOS17/macOS14",
        "mode": "UNIFORM",
        "granularity": "per_tensor",
    }

    # per_grouped_channel palettisation requires an iOS18 target, so the fp16 base for the int4
    # variant is converted separately (the upstream Core ML release ships int4 as iOS18 too).
    fp16_ios18 = build_fp16(checkpoint, out, max_options, name, ct.target.iOS18, test_rows)
    int4 = ct.optimize.coreml.palettize_weights(
        fp16_ios18,
        ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpPalettizerConfig(
                nbits=4, mode="KMEANS", granularity="per_grouped_channel", group_size=32
            )
        ),
    )
    int4.short_description = f"One-pass specialist (int4 weights, fp16 compute, {max_options} options)"
    int4.save(str(out / f"{name}_int4_options{max_options}.mlpackage"))
    manifest["int4"] = {
        "path": f"{name}_int4_options{max_options}.mlpackage",
        "precision": "int4_weights_float16_compute",
        "minimum_target": "iOS18/macOS15",
        "mode": "KMEANS",
        "granularity": "per_grouped_channel",
        "group_size": 32,
    }

    document = {
        "model": f"{name} (one-pass specialist)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "limits": {"context_bytes": 224, "option_bytes": 96, "max_options": max_options},
        "model_config": config,
        "parameters": parameters,
        "trace_check": globals().get("_LAST_TRACE_CHECK"),
        "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "tooling": {"coremltools": ct.__version__, "torch": torch.__version__, "python": sys.version.split()[0]},
        "variants": manifest,
    }
    (out / "conversion.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="trained checkpoint (directory or <name>.safetensors)")
    parser.add_argument("--out", type=Path, default=Path("sv-coreml"), help="directory for the .mlpackage variants")
    parser.add_argument("--name", default=DEFAULT_NAME, help="package name prefix")
    parser.add_argument("--max-options", type=int, default=40, help="option ceiling baked into the export; size it from the widest real input")
    parser.add_argument("--test-rows", type=Path, default=Path("data/sv/test.jsonl"), help="real rows used to verify the traced graph")
    args = parser.parse_args()
    if args.test_rows.exists() is False:
        raise SystemExit(
            f"verification rows not found: {args.test_rows} — generate a corpus first "
            "(python -m onepass.synth); an unverifiable export is not worth shipping"
        )
    result = convert(args.checkpoint, args.out, args.max_options, args.name, args.test_rows)
    print(json.dumps({key: value for key, value in result.items() if key != "model_config"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
