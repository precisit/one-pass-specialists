# RESULTS-SV0 v2 — the released Swedish one-pass specialist

Raw artefacts: `results-pytorch.json`, `results-coreml.json`, `size-sweep.jsonl`,
`corpus-manifest-10000-episodes.json`. Model card:
[`precisit/one-pass-sv-forms`](https://huggingface.co/precisit/one-pass-sv-forms).

## The checkpoint

- 706 048 parameters (`tinyx`), 2.83 MB fp16, trained from scratch.
- Corpus: 10 000 synthetic episodes, seed 2026, split by form signature → 234 921 train /
  29 649 validation / 29 839 test decisions.
- Schedule: 4 epochs, batch 64, learning rate 2e-3, seed 7; 163 minutes on an Apple M4 (24 GB).

## Held-out results (the corpus's own test split, and the hand-written set)

| Run | decisions | top-1 | majority-class baseline | ECE | silent skips |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic test (form-disjoint) | 29 839 | 99.29 % | 47.36 % | 0.0014 | 1 |
| Hand-written out-of-distribution demo | 50 | 100.00 % | 64.00 % | — | 0 |
| Shuffled-context control | 29 839 | 31.99 % | 47.36 % | — | — |
| The *earlier* checkpoint on Swedish rows: released English model of the same family | 2 310 | 21.4 % | 48.87 % | — | 371 of 1 044 fills |

The shuffled control (31.99 %, below the 47.36 % majority baseline) is the load-bearing control:
it shows the score comes from reading the element and the document rather than from the option
statistics. Silent skips fell from 127 (900 episodes) to 1.

## Data-size curve (fixed 4-epoch schedule, fixed seed, corrected generator)

| Episodes | Train decisions | Test decisions | top-1 | silent skips | ECE | training time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 900 | 21 305 | 2 310 | 77.10 % | 127 | 0.0216 | 32 min |
| 4 000 | 94 405 | 11 556 | 98.94 % | 6 | 0.0014 | 56 min |
| 10 000 | 234 921 | 29 839 | 99.29 % | 1 | 0.0014 | 163 min |

Each point is scored on its own corpus's held-out split, so the test set grows with the corpus.

## What changed from v1, and what we got wrong

1. **A format bug in the generator, fixed.** `gen_personnummer` produced nine digits formatted
   `DDDDDDDD-DD` and `gen_orgnummer` appended the check digit, producing eleven characters. Both
   now emit the real Swedish shape (`YYMMDD-XXXX`, 6+4, Luhn-valid). Tests assert the shape and the
   checksum.
2. **An accidental giveaway, removed.** The buggy shapes happened to be unique among the 50
   concepts, so those two concepts could be identified from the *shape of the value* alone. After
   the fix both share `DDDDDD-DDDD` with each other only. Per-action `fill` accuracy is therefore
   the honest kind now (`shape_analysis` in the working notes; the shape census is ten lines).
3. **A wrong explanation, corrected.** The first 900-episode run on the corrected corpus scored
   77.10 % against 83.02 % for the buggy-corpus run, and we attributed the six-point drop to the
   removed giveaway. The 4 000- and 10 000-episode points show the dominant factor was data size.
   The claim was wrong for part of a working day and is recorded here rather than quietly dropped.

## Limits

- Synthetic corpus only: form-disjoint held-out splits, not distribution-disjoint ones. No real
  Swedish form has been through the model; the 50-decision set is the only non-generated evidence.
- One silent skip remains: out-of-model verification is still mandatory.
- Latency figures are single-decision calls; batching untested.
