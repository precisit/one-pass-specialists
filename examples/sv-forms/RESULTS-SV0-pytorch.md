# RESULTS-SV0 — Swedish Jev-shaped form specialist

**Verdict: VALIDATED, with documented limits.**
The Cua-S1 contract trains on our own Swedish synthetic data with the upstream MIT code,
on one laptop, and reaches **83.02 %** top-1 on a held-out, form-disjoint synthetic test
(2 315 decisions) and **86 %** on a hand-written out-of-distribution Swedish demo (50
decisions) — against **21.4 %** / **20 %** for the released English `cua-s1-forms`
checkpoint on the very same rows. The result is undertrained (validation was still
improving at the last epoch), so 83 % is a floor, not a ceiling.

## Protocol as run

| Item | Setting |
| --- | --- |
| Architecture | `tinyx` (byte-level 2-layer Transformer encoder, width 128, 4 heads) + jevlike `AttentionHead`, context 224 / option 96 bytes — identical to upstream |
| Code | `trycua/cua` → `libs/cua-s1` @ `9bbfa7d`, vendored unmodified (`vendor/`, MIT) |
| Corpus | `sv_synth.py` + `concepts_sv.py` (ours): 900 episodes, seed 2026, splits by form field signature |
| Rows | train 21 306 / validation 2 855 / test 2 315 decisions; 725 / 96 / 79 distinct form signatures, **zero overlap** between splits |
| Training | AdamW lr 2e-3, cosine + 5 % warmup, wd 1e-2, 4 epochs, batch 64, seed 7, MPS, 1 927 s |
| Checkpoint rule | best validation NLL (upstream's rule) — epoch 4 |
| Grader | `eval_sv.py`, unchanged for both checkpoints |

## Numbers

| Run | decisions | top-1 | majority baseline | ECE | skipped a required fill | per action |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Swedish specialist — synthetic test (form-disjoint) | 2 315 | **83.02 %** | 50.45 % | 0.01656 | 76 | `check` 98.28 % (n=58), `click` 93.67 % (n=79), `fill` 75.55 % (n=1010), `skip` 88.01 % (n=1168) |
| Swedish specialist — handwritten OOD demo (50) | 50 | **86.00 %** | 64.00 % | 0.09176 | 3 | `check` 100 % (n=3), `click` 100 % (n=3), `fill` 81.25 % (n=32), `skip` 91.67 % (n=12) |
| Swedish specialist — shuffled-context control (test) | 2 315 | 34.08 % | 50.45 % | 0.49735 | 264 | `check` 15.52 %, `click` 0 %, `fill` 1.29 %, `skip` 65.67 % |
| Swedish specialist — shuffled-context control (demo) | 50 | 6.00 % | 64.00 % | 0.78579 | 6 | all actions ≤ 25 % |
| English reference (`cua-s1-forms`) — synthetic test | 2 315 | 21.38 % | 50.45 % | 0.64072 | 371 | `check` 93.10 %, `click` 26.58 %, `fill` 5.54 %, `skip` 31.16 % |
| English reference — handwritten OOD demo | 50 | 20.00 % | 64.00 % | 0.66098 | 16 | `check` 66.67 %, `click` 33.33 %, `fill` 15.62 %, `skip` 16.67 % |
| English reference — shuffled-context control (test) | 2 315 | 17.50 % | 50.45 % | 0.67954 | 363 | `check` 60.35 %, `click` 6.33 %, `fill` 1.78 %, `skip` 29.71 % |

Per-epoch validation: 48.23 % → 62.98 % → 78.63 % → **83.01 %** top-1 (train NLL
2.06 → 0.45; val NLL 1.75 → 0.52). Still improving when the run stopped.

## Reading the numbers

1. **The contract transfers; the language boundary was the real barrier.** The English
   checkpoint on Swedish rows sits at **21.4 %**, barely above its own shuffled-context
   control (17.5 %) and *below* the majority-action baseline (50.5 %), with ECE 0.64 — it
   is not reading the content, it is guessing confidently. On the hand-written demo it
   scores 20 % against a 64 % baseline, i.e. worse than always answering "fill". The
   Swedish specialist, same architecture, same grader, same rows: 83 % / 86 %.
2. **The shuffled control is the floor that matters.** Our specialist drops from 83.02 %
   to 34.08 % when contexts are rotated between rows (below the 50.45 % majority
   baseline), which is the evidence that it uses the element label and the document, not
   option statistics. The English reference barely moves (21.4 % → 17.5 %) — its score was
   near the floor to begin with.
3. **`fill` is the weak action, and the silent-skip failure mode persists.** 75.5 % of
   fills are right; 76 of 1 010 fill decisions (7.5 %) are answered `hoppa över` instead.
   In a real workflow that is the dangerous class: a required field that looks done and is
   not. It is much smaller than the English reference's (371/1 010), but it is not zero,
   and it is the reason a production integration needs a per-item check outside the model.
4. **Calibration is good in-distribution and degrades out-of-distribution** (ECE 0.017 →
   0.092), which is exactly the wrong direction for confidence routing if the threshold is
   tuned on synthetic data.
5. **It is undertrained.** Both the validation curve and the comparison with the reference
   (trained on ~150 k rows vs our 21 k) say the remaining error is largely a data-scale
   effect. The next honest step is more episodes, not a different architecture.

## What this does not establish

- Nothing about real Swedish forms. The demo set is hand-written by the same person who
  wrote the catalogue, so it shares the author's vocabulary; a genuine test needs real
  blanketter (Skolverket/Försäkringskassan/kommunala) or our own live forms.
- Nothing about the runtime: this spike scores decisions, it does not drive a GUI.
  Fill/click execution, retries and fail-closed behaviour were not exercised (upstream's
  `planner.py` refuses to execute without a compatible driver contract).
- Nothing about English transfer in the other direction: we never trained a Swedish model
  on English rows, and the corpus changes *language and label vocabulary at once*, so the
  ablation "language vs vocabulary" is still open.
- Reproducibility is bounded by the environment: torch 2.14 on Apple Silicon MPS, seed 7,
  `deterministic=False` (deterministic algorithms were too slow on MPS), so the exact
  numbers are one draw, not a bit-reproducible receipt.

## Follow-ups worth pre-registering

1. **Scale test**: 4 000–10 000 episodes, 6 epochs, batch 32–64 on a machine with room —
   does top-1 cross ~95 % and does the silent-skip rate fall below 2 % of fills?
2. **Language ✕ vocabulary ablation**: Swedish documents + English labels, and the mirror,
   to attribute the win.
3. **Real-form OOD**: hand-build 3–5 authentic Swedish blanketter (never in the catalogue)
   and score before touching the corpus again.
4. **Abstention outside the model**: a spread/top-probability gate plus a "required field
   still empty" check, measured on the silent-skip class specifically.

## Reproduction

See `README.md` in this directory. Runtime: ~5 min generation, 32 min training, seconds of
scoring on one Apple Silicon laptop. Corpus and checkpoint are not in Git
(`~/agent-data/jevlike-sv-spike/`); `data/sv/manifest.json` carries the SHA-256 of each split
(test `92c0dfec…d272`).
