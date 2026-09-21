# RESULTS-SV0-COREML — the Swedish SV0 specialist exported to Core ML

**Verdict: PARTIAL — fp16 and int8 export cleanly and run on the Neural Engine at the same
accuracy as PyTorch; int4 is _not_ usable for this checkpoint (83 % → 51 % top-1) and does not
compile for the ANE.** Recommended deployment artifact: **int8, 787 KB, ~1.3 ms per decision.**

## What was exported

`export_sv_coreml.py` in this directory converts `runs/sv-tinyx/model` (the SV0 checkpoint,
706 048 parameters) with coremltools 9.0, mirroring the fixed-shape contract of the
FluidInference CUA-S1 release: `context_ids (1, 224)` / `option_ids (1, 40, 96)` /
`option_mask (1, 40)`, all int32 byte-ids (byte + 1, 0 = pad), output `logits (1, 40)`.
`max_options` is **40** here rather than their 32 because our corpus tops out at 37 options —
the ceiling is baked into the export, so it must be sized from the widest real input.

Variants: fp16 (iOS17), int8 (UNIFORM, per-tensor, iOS17), int4 (KMEANS, per-grouped-channel,
group 32, iOS18). `conversion.json` records hashes, config and tooling versions.

## Export-time checks (all passed before any measurement)

| Check | Result |
| --- | --- |
| Export forward vs vendored model, 256 real rows, max abs logit difference | **0.0** (bit-identical) |
| Traced graph vs model, 64 real rows | max abs Δ 7.2 × 10⁻⁶, argmax agreement **1.0** |

## Measurements (Apple M4, macOS 26.6.2; 2 312 scored rows after 3 warm-up rows; same grader as the spike)

| Variant | package | test top-1 | demo top-1 | silent skips (of 1 008 fills) | argmax parity vs PyTorch | mismatches | median | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch (MPS), reference | 2 828 784 B | **83.02 %** | 86 % | 76 | — | — | — | — |
| **fp16, CPU + ANE** | 1 507 481 B (1.44 MB) | 83.00 % | 85.11 % | 76 | 0.99870 | 3 / 2 312 | 1.307 ms | 1.460 ms |
| fp16, CPU only | 1 507 481 B | 83.04 % | — | 76 | 0.99913 | 2 / 2 312 | 1.734 ms | 1.917 ms |
| **int8, CPU + ANE** | **806 132 B (787 KB)** | **83.09 %** | 85.11 % | 76 | 0.99611 | 9 / 2 312 | 1.316 ms | 1.402 ms |
| int4, CPU + ANE | 492 584 B (481 KB) | **50.87 %** | 27.66 % | 954 | 0.48702 | 1 186 / 2 312 | 1.968 ms | 2.099 ms |

Fills per-action on fp16: `fill` 75.40 %, `skip` 88.26 %, `check` 96.55 %, `click` 93.67 % —
within a few rows of the PyTorch run, i.e. the export is behaviour-preserving.

## What the numbers say

1. **The export works, and the ANE helps here too**: fp16 median 1.307 ms with the ANE vs
   1.734 ms CPU-only, p95 1.460 ms vs 1.917 ms. Smaller absolute win than on the English model
   (whose weights are better conditioned), but the same direction.
2. **int8 is free and is the artifact to ship**: 787 KB (52 % of fp16), top-1 83.09 % — one
   decision *better* than PyTorch on this split — 9 argmax deviations out of 2 312, identical
   silent-skip count, and the same ANE behaviour.
3. **int4 does not transfer to an undertrained model.** On Cua's English checkpoint the same
   kind of palettisation cost 14 decisions in 24 367 (0.06 pp); here it costs 1 186 in 2 312
   (51 pp) and raises silent skips from 76 to 954. Two mechanisms are plausible and both point
   the same way: a model that has not converged stores information in fine weight differences
   that a 4-bit palette cannot represent, and the per-grouped-channel kmeans graph also fails
   to compile for the ANE (`ANECCompile() FAILED`, seen repeatedly during save and load), which
   is why its latency is worst despite being the smallest artifact. **Quantisation headroom is
   a property of the training run, not of the architecture** — that is the transferable finding.
4. **fp16 parity is 99.87 %, not 100 %**, and that is expected rather than alarming: three
   rows change their argmax under fp16 rounding on a model whose decision margins are thin
   (83 % top-1). On the well-trained English model the same conversion showed zero deviations.
   Parity must therefore be *measured* per checkpoint, not assumed — exactly what this harness
   does.

## Pitfalls found while exporting (all cost a failed run)

- **`torch.jit.trace` under `torch.no_grad()` takes PyTorch's fused sparsity fast path**
  (`torch._transformer_encoder_layer_fwd`), which coremltools cannot convert. Trace with grad
  *enabled*.
- **The coremltools torch frontend has no `__or__` for bool tensors** → use
  `torch.logical_or` / `torch.logical_not`. (`&`, `~` alone, and in-place `mask[:, 0] = True`
  all fail; probe the op support in a 5-line script before rewriting a model.)
- **`clamp_min(1)` with a Python int trips `assert x.dtype == y.dtype`** in the frontend; pass a
  same-dtype tensor constant.
- **`per_grouped_channel` palettisation requires an iOS18 deployment target**, so the int4
  variant needs its own fp16 base converted at iOS18 (the upstream release does the same).
- **k-means palettisation needs `scikit-learn`**, which is not a coremltools dependency; the
  error arrives mid-run, after the fp16 conversion has already succeeded — hence the
  `REUSE_FP16=1` cache flag in the export script.
- **Palettizer API in coremltools 9.0**: wrap the op config —
  `OptimizationConfig(global_config=OpPalettizerConfig(...))` — and note the mode names are
  upper-case (`UNIFORM`, `KMEANS`) while granularity is `per_tensor` / `per_grouped_channel`
  only.
- **Check that the traced graph is the model** before converting: `torch.jit.trace(..., check_trace=False)`
  suppresses a noisy replay check, so the export script compares logits itself and aborts on
  mismatch.

## Reproduction

```bash
# (in the spike venv with coremltools 9.0, torch 2.7, safetensors, scikit-learn)
python check_export_forward.py     # export forward is bit-identical to the model
REUSE_FP16=1 python export_sv_coreml.py    # fp16 + int8 + int4 mlpackages, conversion.json
python eval_sv_coreml.py           # accuracy, silent skips, parity, latency -> RESULTS-sv0-coreml.json
```

Artifacts (not in Git): `sv-coreml/*.mlpackage`, `RESULTS-sv0-coreml.json`,
`export.log`, `eval-coreml.log`.
