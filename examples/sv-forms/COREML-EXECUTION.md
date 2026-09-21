# Core ML execution configuration: the int4 result

Updated 2026-09-21. The released int4 package retains high accuracy on CPU but
produces substantially different choices with `CPU_AND_NE`. The mechanism is
unresolved. Weight precision alone does not explain these results.

## Full test split on M4

The original [comparison](results-int4-compute-units.json) covers all 29,839
held-out decisions. Reported platform: Apple M4, macOS 26.6, coremltools 9.0.

| Package | CPU-only accuracy | CPU-only changes vs PyTorch | CPU + NE accuracy | CPU + NE changes vs PyTorch |
| --- | ---: | ---: | ---: | ---: |
| Float16 | 99.28% | 14 | 99.29% | 15 |
| Int8 | 99.31% | 12 | 99.29% | 13 |
| Int4 | 99.27% | 32 | 49.92% | 15,017 |

`CPU_AND_NE` permits CPU and Neural Engine execution. It does not measure actual
operation placement. Similar accuracy does not imply identical predictions:
int8 is not established to be identical to float16 under either setting.

The earlier [timing run](results-coreml.json) excludes three warm-up examples
and measures 29,836 rows. Keep its denominator and mismatch counts separate.

## Bounded reproduction on M1 Max

[Raw results](results-coreml-m1-max-512.json), including file hashes, sample
indices and predictions, were produced by [the portable probe](probe_coreml.py).
Platform: Apple M1 Max, macOS 26.5.1 (25F80), Python 3.12.13, PyTorch 2.7.0,
coremltools 9.0, NumPy 2.2.6. The probe ran against toolkit revision
`1618a0a24c83d1c9afa6b5a2e2943d55ecfcd7e4`.

| Package | CPU-only correct | CPU + NE correct |
| --- | ---: | ---: |
| Float16 | 512 / 512 | 511 / 512 |
| Int8 | 512 / 512 | 512 / 512 |
| Int4 | 511 / 512 | 266 / 512 |
| Int4 expanded into dense constants | 511 / 512 | 266 / 512 |

The 512 indices were selected before observing predictions, using
`numpy.linspace(0, 29838, num=512, dtype=int)`. This is a deterministic diagnostic
sample, not a random sample or a replacement for full-set accuracy. PyTorch gets
all 512 correct. All real-option logits from all eight configurations are finite.
No timing claims are made from this run. The chip and OS both differ from the M4
run; this is not an isolated chip comparison.

The control uses Apple's `decompress_weights` on the released int4 package. It
materializes the already rounded weights as dense float constants, without
restoring the original weights or retraining. All 17 `constexpr_lut_to_dense`
operations disappear. The package grows from 492,577 to 1,506,602 bytes and retains
specification version 9. The utility also runs conversion passes, so this does
not establish an otherwise identical graph.

Decompression changes zero predicted choices under either compute setting.
Both int4 representations disagree with their own CPU-only output on 246 sampled
rows under CPU + NE. Removing the palette operations did not repair the result.

The successful process emitted two messages containing `MILCompilerForANE error`
and `ANECCompile() FAILED`. These were buffered at the end of the combined log;
the run does not establish which invocation emitted which message, whether a
fallback happened, or where the predictions were computed.

## Corrections to earlier interpretations

- A failure observed with `CPU_AND_NE` is not an ANE-only measurement. The numbers
  establish an execution-configuration discrepancy, not a specific miscompile.
- [The margin script](int4_margins.py) uses `CPU_ONLY`. Its 1,990-row result has
  two changed choices with median PyTorch margin 0.623. The 12.062 median is over
  all rows. It is not evidence about margins on failing CPU + NE predictions.
- The margin script's `ours_fp16` label refers to its PyTorch baseline without an
  explicit half conversion. Its `accuracy_cost_of_flips` value is accuracy on
  unchanged rows, not a measured cost of changed choices. Retained historical
  result files should be interpreted with these qualifications.
- Logit correlation alone cannot distinguish fallback from miscompilation.
- The English `ane-gather` conversion is a useful lead, not a matched control
  for this export recipe. Do not infer identical formats or measured placement.

## Reproduce the bounded check

Use macOS with Core ML runtime access. In a checkout containing the probe:

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[train,dev]' 'torch==2.7.0' 'coremltools==9.0' 'numpy==2.2.6' huggingface-hub
.venv/bin/hf download precisit/one-pass-sv-forms sv0-forms.safetensors sv0-forms.json --revision 778ff920104b36800a9f028521ad958eedcb8b7a --local-dir runs/released
.venv/bin/hf download precisit/one-pass-sv-forms --include 'coreml/**' --revision 778ff920104b36800a9f028521ad958eedcb8b7a --local-dir runs/released
.venv/bin/hf download precisit/one-pass-sv-forms-synthetic data/test.jsonl.gz --repo-type dataset --revision b6461d95a2c3308c2224216cec51b7ae04d945e6 --local-dir data/probe
.venv/bin/python examples/sv-forms/probe_coreml.py --toolkit . --test data/probe/data/test.jsonl.gz --checkpoint runs/released/sv0-forms.safetensors --packages runs/released/coreml --output /tmp/one-pass-coreml/results.json --sample 512 --decompress
```

The probe was originally run from an editorial checkout with the same script
bytes (its SHA-256 is recorded in the result). The historical full-set script
`int4_compute_units.py` has machine-specific paths; this CLI accepts explicit
paths and keeps its outputs outside the source tree. Retain stdout/stderr.

The test split's decompressed SHA-256 matches the published corpus manifest:
`105b7c0509af8667c11653f77d3866dc528e846a3f439e284dccb62164d2dc3f`.

## Next tests

1. Match the conversion target: float16 and int8 currently target iOS 17
   (specification version 8); int4 targets iOS 18 (version 9). Compare float16
   exports at both targets with the same source graph and converter.
2. Inspect placement and compare the English gather variant. If needed, bisect
   context encoder, option encoder and scoring head. Capture compiler diagnostics
   separately for each configuration, rather than assigning buffered messages.
3. Test any candidate fix on all 29,839 rows, checking decision parity and action
   errors, then verify on both platforms. Measure latency separately.
4. Consider new palette recipes or quantization-aware training only if a remaining
   gap is actually attributable to weight approximation. CPU int4 already keeps
   99.27% in the reported full-set measurement.

Apple references: [compute units](https://apple.github.io/coremltools/source/coremltools.models.html),
[compression and hardware](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html),
[decompression utility](https://apple.github.io/coremltools/docs/source/coremltools.optimize.coreml.post_training_quantization.html).
