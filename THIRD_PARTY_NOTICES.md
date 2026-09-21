# Third-party notices

This repository builds on public, MIT-licensed work. It is an independent implementation of a
public model *interface*; it is not affiliated with, derived from, or endorsed by TypeSafe AI
(the makers of "Jev").

## trycua/cua — `libs/cua-s1` (MIT)

- Source: https://github.com/trycua/cua/tree/main/libs/cua-s1
- Revision used: `9bbfa7d` (the commit that links their published artifacts)
- Licence text: `vendor/CUA-S1-LICENSE` (kept verbatim)
- What is vendored **unmodified** under `vendor/`:
  - `cua_s1/` — the `tinyx` architecture, the byte collator, the checkpoint format and the
    concept/synthesis helpers used as the shape we mirror
  - `training/` — the training loop
  - `metrics.py` — the evaluation metrics
- What is ours, adapted or new: `onepass/catalogue_sv.py` (the Swedish catalogue),
  `onepass/synth.py` (an adaptation of the upstream generator with a fully Swedish surface),
  `onepass/encode.py`, `onepass/evaluate.py`, `onepass/forward.py`, `onepass/export_coreml.py`
  and `onepass/coreml_evaluate.py`.

The published checkpoint was trained **from scratch** on our own synthetic corpus with this
code. It shares the architecture, trainer and data format with Cua's release — not weights,
not data.

## vinnylarouge/jevlike (MIT), via Cua's attribution

- Source: https://github.com/vinnylarouge/jevlike
- The option-attention head design used by `tinyx` originates there; Cua's `model.py` credits
  commit `94f5fd1` (MIT, Copyright 2026 Minimal Labs) and that attribution is preserved in the
  vendored code.

## What was not used

- No TypeSafe AI code, weights, data or API output. "Jev" and "System One" are TypeSafe AI's
  names for a similar interface; this project is independent and does not claim their lineage.
- No real personal data. Every generated value is synthetic: fictional names, `.invalid`
  e-mail domains, invented organisations, and personnummer-/organisationsnummer-shaped
  identifiers that pass their checksum but belong to nobody.
