# one-pass specialists

Small models that **score a supplied list of options in one forward pass** — no text
generation, no tokenizer, one probability per candidate decision. This repository is the
toolkit that builds them: generate a synthetic corpus from a concept catalogue, train a
~0.7 M-parameter encoder, measure it honestly, and export it to Core ML so it runs on an
Apple device in about a millisecond.

The worked example is **[One-Pass SV-Forms (SV0)](https://huggingface.co/precisit/one-pass-sv-forms)** —
a Swedish form specialist trained from a Swedish synthetic corpus on one laptop. The recipe,
the numbers and the failures are all in [`examples/sv-forms/`](examples/sv-forms/).

A second worked example, **[onepass-c4](https://huggingface.co/precisit/onepass-c4)**, uses the same
model to play Connect Four in the browser — context = the board, options = the legal columns — trained
on positions labelled by an exact solver: [`examples/c4/`](examples/c4/)
([play it](https://precisit.github.io/onepass-web/demo/c4/)).

## Why this shape of model

A language model that *writes* an answer has to be trusted, budgeted for and served. A model
that only *ranks options you already defined* can be verified field by field, costs
microseconds to milliseconds, and fits in a phone — which makes it a plausible component for
bounded, high-volume interface work (filling a form from an extracted document, routing a
known set of categories, triaging against fixed criteria). The trade is explicit: it cannot
invent a value it was not given, and it cannot decide what to do next. Execution order,
retries and the decision to submit belong in ordinary code around it.

## Pipeline

| Module | What it does |
| --- | --- |
| `onepass.synth` | Generate episodes (context + options + gold label) from a concept catalogue, split by **form signature** so held-out forms are truly unseen; writes a manifest with per-split SHA-256. |
| `onepass.train` | Train the vendored upstream trainer on those splits. |
| `onepass.evaluate` | Score a checkpoint: top-1, per-action accuracy, ECE, a shuffled-context control, and **silent skips** (a required fill answered "skip"). |
| `onepass.export_coreml` | Export fp16 / int8 / int4 Core ML packages, checking the export-friendly eager implementation against its traced graph before conversion. |
| `onepass.coreml_evaluate` | Measure the exported packages against the PyTorch checkpoint: argmax parity, accuracy, and median/p95 latency with and without the Neural Engine. |
| `onepass.encode` | The byte-input contract shared by both paths (byte + 1, zero padding, fail-closed option ceiling). |

```bash
uv venv .venv && VIRTUAL_ENV=.venv uv pip install torch safetensors numpy pytest
python -m onepass.synth    --output data/sv --episodes 900 --seed 2026
python -m onepass.train    --data data/sv --output runs/sv-tinyx/model --epochs 4 --batch-size 64
python -m onepass.evaluate --checkpoint runs/sv-tinyx/model --out RESULTS-pytorch.json
# optional, on macOS: python -m onepass.export_coreml && python -m onepass.coreml_evaluate
```

On a 24 GB Apple laptop the reference run (900 episodes, 21 306 decisions, 4 epochs) trains in
about 32 minutes. Keep it the only heavy job on the machine: a second concurrent run more than
doubles the wall-clock and pushes the machine into swap.

## Writing another language or vertical

The interesting work is the **catalogue**, not the architecture. A catalogue is a list of
concepts, each with the labels it appears under in forms and in documents, plus a value
format — and that is where local knowledge lives (`personnummer`, `bankgiro`, `momsregistreringsnummer`
for Sweden; whatever the equivalent is elsewhere). `onepass/catalogue_sv.py` (50 concepts) is
a worked example to copy:

```python
C("personnummer", ("Personnummer", "Personnr"), ("Personnummer", "Pnr"),
  gen_personnummer, placeholder=("ÅÅÅÅMMDD-XXXX",), group="person"),
```

Then regenerate the corpus, retrain, and re-measure. Nothing above `onepass.train` needs to
know it is Swedish.

## What we measured, and what we did not

`examples/sv-forms/` holds the receipts: the training run, both result files and the corpus
manifest. Headlines from that example:

- The current release scores **99.29%** on 29,839 held-out synthetic decisions;
  the majority-class baseline is 47.36%. It scores 100% on a small 50-decision
  handwritten set. The unchanged English checkpoint scores 20.75% on the same
  Swedish test split, with different labels and action strings from its training.
- It silently skips **1 of 13,844 expected fills**. This excludes wrong-value
  fills, so inspect the action breakdown as well as the overall accuracy.
- The **787 KiB int8** Core ML package scores 99.31% with CPU-only execution and
  99.29% with CPU and Neural Engine allowed. Similar accuracy is not identical
  predictions. The earlier M4 timing run measured a 1.314 ms median prediction
  call under `CPU_AND_NE`, excluding encoding, loading and warm-up.
- The **481 KiB int4** package scores 99.27% on CPU and 49.92% with `CPU_AND_NE`.
  A 512-row M1 Max probe reproduces this configuration-dependent discrepancy.
  Decompressing the rounded weights does not repair it. The exact cause is
  unresolved; do not attribute it simply to 4-bit precision or claim ANE-only
  placement. See [the investigation and reproduction](examples/sv-forms/COREML-EXECUTION.md).

The [fixtures based on our websites](examples/sv-forms/results-real-forms.json)
score 52.8% (Kanslist, 195 decisions) and 77.3% (Pratsam, 75). These are simplified
field definitions with synthetic values, not browser tests. Pratsam's radio
expected-answer logic needs correction; that part of its score is provisional.
Not measured: real submissions, real users, non-Swedish catalogues or end-to-end
form completion. Generated values are synthetic; generated identifiers are not
guaranteed never to coincide with real identifiers.

## Provenance and licence

MIT (see `LICENSE`). The model architecture, training loop and evaluation metrics are
vendored **unmodified** from Cua's MIT [`libs/cua-s1`](https://github.com/trycua/cua/tree/main/libs/cua-s1)
(which in turn credits [`jevlike`](https://github.com/vinnylarouge/jevlike) for the
option-attention head); the catalogue, generator adaptation, metrics wrappers, Core ML tooling
and the SV0 checkpoint are Precisit's. **No TypeSafe AI code, weights, data or API output was
used** — "Jev" and "System One" are their terms for a similar interface, and this project is
independent. Details in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
