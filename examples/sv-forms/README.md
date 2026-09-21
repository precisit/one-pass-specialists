# Worked example: One-Pass SV-Forms (SV0)

A Swedish form specialist trained on one laptop, published as
[`precisit/one-pass-sv-forms`](https://huggingface.co/precisit/one-pass-sv-forms) (MIT), with its
synthetic corpus at
[`precisit/one-pass-sv-forms-synthetic`](https://huggingface.co/datasets/precisit/one-pass-sv-forms-synthetic).

Everything here is a receipt: the corpus manifest, the result files, and the 50-decision
hand-written set. The point of shipping them is that you can re-run the pipeline and get the same
numbers, or refute them.

## The task

Given the UI element (role, label, current state) plus the entities extracted from a document,
choose one option from the supplied list:

```
fyll <entitet>   fyll i fältet med ett värde från dokumentet
kryssa           sätt kryss i en kryssruta
klicka           klicka på en knapp (t.ex. skicka in)
hoppa över       lämna fältet tomt
```

The model never writes text and never decides what to do next: the option list and the execution
order come from ordinary code around it.

## How it was made

```bash
python -m onepass.synth    --output data/sv --episodes 10000 --seed 2026
python -m onepass.train    --data data/sv --output runs/sv-tinyx/model --epochs 4 --batch-size 64
python -m onepass.evaluate --checkpoint runs/sv-tinyx/model --out RESULTS-pytorch.json
```

- **Model:** 706 048 parameters (`tinyx`: byte embeddings, 2 encoder layers, width 128, 4 heads,
  option-attention head), 2.83 MB fp16, trained 163 minutes on an Apple M4.
- **Corpus:** 10 000 episodes → 234 921 train / 29 649 validation / 29 839 test decisions, split by
  **form signature** so a form's rows never straddle splits; seed 2026; `manifest.json` carries the
  per-split SHA-256.
- **Held-out sets:** the corpus's own test split (29 839 decisions) plus
  [`demo-handwritten.jsonl`](demo-handwritten.jsonl) — 50 decisions over three forms written by
  hand, before the model existed.

## Results

| Run | decisions | top-1 | majority-class baseline | ECE | silently skipped a required fill |
| --- | ---: | ---: | ---: | ---: | ---: |
| Held-out synthetic test (form-disjoint) | 29 839 | **99.29 %** | 47.36 % | 0.0014 | **1** |
| Hand-written out-of-distribution demo | 50 | **100.00 %** | 64.00 % | — | 0 |
| Shuffled-context control (same test rows) | 29 839 | 31.99 % | 47.36 % | — | — |
| The same *earlier* checkpoint on Swedish rows: the released English model | 2 310 | 21.4 % | 48.87 % | — | 371 of 1 044 fills |

The shuffled control is the number to read first: rotating contexts between rows collapses the
model below the majority baseline, so the score comes from reading the element label and the
document, not from option statistics.

## The data-size curve (the useful part)

Same generator, same 4-epoch schedule, same seed; only the number of episodes changes. Each point
is scored on its own corpus's held-out split, so the test set grows with the corpus.

| Episodes | Train decisions | Test decisions | top-1 | silent skips | ECE | Training time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 900 | 21 305 | 2 310 | 77.10 % | 127 | 0.0216 | 32 min |
| 4 000 | 94 405 | 11 556 | 98.94 % | 6 | 0.0014 | 56 min |
| **10 000** | **234 921** | **29 839** | **99.29 %** | **1** | 0.0014 | 163 min |

Read the first two columns together: at 900 episodes this pipeline sits at 77 %, which is a number
someone could easily mistake for a statement about the method. It is a statement about the data.
`RESULTS-sweep.jsonl` holds the raw points.

## Core ML export of the released checkpoint (Apple M4, coremltools 9.0)

| Variant | package | top-1 (29 839 rows) | argmax parity vs PyTorch | median | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16, CPU + ANE | 1.44 MB | 99.29 % | 0.99950 (15) | 1.314 ms | 1.412 ms |
| fp16, CPU only | 1.44 MB | 99.28 % | — | 1.727 ms | 1.929 ms |
| **int8, CPU + ANE** | **787 KB** | **99.29 %** | 0.99956 (13) | 1.314 ms | 1.411 ms |
| int4, CPU + ANE | 481 KB | **49.92 %** | 0.49675 (15 015) | 1.984 ms | 2.115 ms |

**int4 does not work for this checkpoint** (and its graph fails ANE compilation); it is published as
evidence. Note the falsified assumption: we first thought quantisation headroom tracked how well
trained the model was, because Cua's converged English checkpoint tolerated the same palettisation
at a cost of 0.06 pp. The well-trained 10 000-episode checkpoint collapses just like the
undertrained one, so that explanation is wrong and we say so. int8, meanwhile, is free: 787 KB,
same accuracy, same tail latency.

Nothing is exported before two proofs: the export-forward rewrite is bit-identical to the
checkpoint, and the traced graph reproduces it on real rows (argmax agreement 1.0 over 512 rows,
maximum softmax-probability difference 1.0e-6). A conversion that changes a decision aborts the run.

## Changelog

- **v2 (2026-09-21)** — released checkpoint.
  1. **Corpus correctness.** `gen_personnummer` emitted nine digits formatted `DDDDDDDD-DD` and
     `gen_orgnummer` emitted `DDDDD-DDDDD`; both now produce the real Swedish shape
     `YYMMDD-XXXX`. Side effect worth knowing: the bug had made those two concepts identifiable
     from the *shape of the value* alone, so this fixed an accidental giveaway as well as a format.
     `tests/test_catalogue_sv.py` now asserts both shapes and verifies the Luhn checksum.
  2. **Scale.** 900 → 10 000 episodes.
  3. **A corrected explanation.** An intermediate 900-episode run on the corrected corpus scored
     77.10 % where the earlier (buggy-corpus) run scored 83.02 %, and the first reading of that was
     "the fix cost us six points". The 4 000- and 10 000-episode points show the dominant factor was
     data size. Recorded here because the wrong explanation was published for part of a day.
- **v1 (2026-09-21)** — first release: 900 episodes on the buggy corpus; 83.02 % top-1, 7.5 %
  silent skips. Superseded.

## Known issues, honestly

1. **The corpus is synthetic.** Fictional names, `.invalid` addresses, invented organisations,
   generated identifiers that belong to nobody. No real form has ever been through this model. The
   only non-synthetic evidence is the 50-decision set above; a small set built from two of our own
   shipped forms (Kanslist, Pratsam) is the next measurement.
2. **Silent skips are rare, not gone.** 1 of 29 839 decisions still left a required field empty
   while claiming success. Out-of-model verification stays mandatory (fail closed, dry run, one
   submit, human review before consequential actions).
3. **The held-out splits share a generator with the training data.** They are form-disjoint, not
   distribution-disjoint: they test generalisation to unseen *forms*, not to unseen *writers* or
   real documents.
4. **Nothing here is tuned for throughput.** One decision per call is what the latency numbers
   describe; batching is untested.
