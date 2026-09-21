# Worked example: One-Pass SV-Forms (SV0)

A Swedish form specialist, trained from a Swedish synthetic corpus on one laptop in an
afternoon, and published as
[`precisit/one-pass-sv-forms`](https://huggingface.co/precisit/one-pass-sv-forms) (MIT).

Everything in this directory is a receipt: the corpus manifest, both result files, and the
50-decision hand-written set. The point of shipping them is that you can re-run the pipeline
and get the same numbers — or refute them.

## The task

Given the UI element (role, label, current state) plus the entities extracted from a document,
choose one option from the supplied list:

```
fyll <entitet>   fyll i fältet med ett värde från dokumentet
kryssa           sätt kryss i en kryssruta
klicka           klicka på en knapp (t.ex. skicka in)
hoppa över       lämna fältet tomt
```

The model never writes text and never decides what to do next — the option list and the
execution order come from ordinary code around it.

## How it was made

```bash
python -m onepass.synth  --output data/sv --episodes 900 --seed 2026
python -m onepass.train  --data data/sv --output runs/sv-tinyx/model --epochs 4 --batch-size 64
python -m onepass.evaluate --checkpoint runs/sv-tinyx/model --out RESULTS-pytorch.json
```

- **Corpus:** 900 episodes → 21 306 train / 2 855 validation / 2 315 test decisions, split by
  *form signature* (a form's rows never straddle splits), seed 2026, manifest with per-split
  SHA-256: [`corpus-manifest-900-episodes.json`](corpus-manifest-900-episodes.json).
- **Model:** 706 048 parameters (`tinyx`: byte embeddings, 2 encoder layers, width 128, 4 heads,
  option-attention head), 2.83 MB fp16, trained 32 minutes on an Apple M4.
- **Held-out set:** the synthetic test split *plus* [`demo-handwritten.jsonl`](demo-handwritten.jsonl) —
  50 decisions over three forms written by hand, before the model existed, as an
  out-of-distribution check.

## Results

| Run | decisions | top-1 | majority-class baseline | ECE | silently skipped a required fill |
| --- | ---: | ---: | ---: | ---: | ---: |
| Held-out synthetic test (form-disjoint) | 2 315 | **83.02 %** | 50.45 % | 0.017 | **76 (7.5 % of fills)** |
| Hand-written out-of-distribution demo | 50 | **86.00 %** | 64.00 % | 0.092 | 3 (of 32 fills) |
| Shuffled-context control (same test rows) | 2 315 | 34.08 % | 50.45 % | 0.497 | 264 |
| The same rows, *English* checkpoint of the family | 2 315 | 21.4 % | 50.45 % | — | 371 of 1 010 |

Per-action on the test split: `check` 98.3 %, `click` 93.7 %, `skip` 88.0 %, `fill` 75.5 %.
Validation was still improving when the run stopped (48 % → 63 % → 79 % → 83 % top-1 across
four epochs), so 83 % is a floor for this recipe, not a ceiling.

Core ML export of the same checkpoint (macOS, coremltools 9.0):

| Variant | package | top-1 | argmax parity vs PyTorch | median | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16, CPU + ANE | 1.44 MB | 83.00 % | 0.99870 | 1.307 ms | 1.460 ms |
| fp16, CPU only | 1.44 MB | 83.04 % | 0.99913 | 1.734 ms | 1.917 ms |
| **int8, CPU + ANE** | **787 KB** | **83.09 %** | 0.99611 | 1.316 ms | 1.402 ms |
| int4, CPU + ANE | 481 KB | **50.87 %** | 0.48702 | 1.968 ms | 2.099 ms |

**int4 collapses this checkpoint** (and fails ANE compilation) where the same palettisation
cost 0.06 pp on the converged English model. Quantisation headroom is a property of the
training run, not of the architecture: sweep it per checkpoint, and grade it with the metric
that actually hurts (here: silent skips, which went 76 → 954).

Raw output: [`RESULTS-SV0-pytorch.md`](RESULTS-SV0-pytorch.md),
[`RESULTS-SV0-COREML.md`](RESULTS-SV0-COREML.md).

## Known issues, honestly

1. **The corpus is synthetic.** Fictional names, `.invalid` e-mail domains, generated
   identifiers that belong to nobody. No real form has ever been through this model; the only
   non-synthetic evidence is the 50-decision set above.
2. **Silent skips.** 7.5 % of required fills are answered "hoppa över" — a required field that
   quietly stays empty. Any real integration needs an out-of-model check (fail closed, dry run,
   one submit, human review before consequential actions).
3. **The published SV0 checkpoint was trained before two generator bugs were fixed**
   (2026-09-21): `gen_personnummer` emitted nine digits formatted as 8+2 instead of
   `YYMMDD-XXXX`, and `gen_orgnummer` appended the check digit instead of replacing the tenth.
   Values were still synthetic and internally consistent — so the model works — but it learned
   the *wrong surface* for those identifiers, which is a real distribution mismatch on real
   Swedish documents. The fix is in `onepass/catalogue_sv.py`; the next release should be
   retrained on the corrected corpus.
4. **Nothing here is tuned for throughput.** One decision per `predict` call is what the
   latency numbers describe; batching is untested.
