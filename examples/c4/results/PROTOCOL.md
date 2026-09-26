# PROTOCOL: how the v2 Connect Four model is judged (frozen before the first training run that counts)

The original of this text was hashed (sha256 `732eda1b29f6...`) on 2026-09-26 before the first
training run that counts, and was not changed afterwards; this copy differs only in file names.
Amendments are dated and additive; a bar that fails is reported as failed.

## The claim under test

A Jev-like one-pass option scorer - the toolkit's `TinyTransformerScorer`, unchanged, only its
config scaled - plays Connect Four well as a **pure policy**: one forward pass over the context
bytes and the legal `column k` options, then argmax. No search, no rules, no solver at play time.

## Players (`protocol_v2.py`)

| name | what it is |
| --- | --- |
| `model:<ckpt>` | pure policy: `encode_c4` context, argmax over legal columns (exact ties at random) |
| `onnx-v2:<file>` / `onnx-v1:<file>` | an exported ONNX file, fed exactly as the browser feeds it |
| `bot:d` | the hand-written bot, `engine.bot_move(position, d)`: alpha-beta to depth d with a heuristic leaf - the same bot the first version was measured against |
| `random` | uniform legal column |
| `solver` | perfect play: uniformly random among the best-scoring moves (connect-four-ai, verified in G0) |
| `solver-eps:E` | `solver`, uniformly random move with probability E |

## Game cells (kept unchanged from the first version's frozen protocol)

* Colours alternate within a cell; A and B are tallied (never first/second).
* Epsilon-noise 0.05 on **both** players, every move; games from the **empty board** (`--opening 0`).
* 200 games per cell, seed 2026, Wilson 95 % intervals. Score = (wins + draws/2) / games.
* A ceiling cell (`solver` against the same opponent, same noise) is run for every bar, so a bar is
  never set above what perfect play reaches under this protocol. Measured before freezing: solver vs bot:4 **0.890** [0.839-0.926], solver vs bot:6 **0.925** [0.880-0.954].

## The bars ("Strong", agreed 2026-09-26)

| # | measurement | bar |
| --- | --- | --- |
| B1 | model vs `bot:4`, 200 games, from the empty board | score >= 0.85 |
| B2 | model vs `bot:6`, 200 games, from the empty board | score >= 0.70 |
| B3 | value-preserving rate on the frozen eval set (below), all decisions | >= 0.97 |

A bar is met when the point estimate meets it; intervals are always reported next to it.
"Reliably beats casual humans" is judged in the demo, not scored here.

## The frozen eval set (move level)

`evalset-v2`: **17 325 positions**, up to 480 per ply for plies 0-40 (fewer where a ply has fewer
distinct positions: plies 0-5 and 38-40), 240 per ply from the TonyCWang **test** split (9 031) and
240 per ply from our generator with a seed never used for training (`c4label --gen 3000 --seed
9001`, 8 294), deduplicated by symmetric key. Built by
`eval_c4.py build-evalset --tonyc-test corpus/tonyc-test --gen corpus/gen-eval-s9001`; array
hashes (sha256): board `a4535cfa...63e9`, scores `aafe9fc5...ae4f`, key `56b7722d...8861`,
src `b98d14b4...0215` (full values in the eval set's manifest.json).
Training corpora exclude every eval key at plies >= 12 (below ply 12 the positions are few and
every game passes through them; their overlap with training is reported, not removed).
Labels: exact connect-four-ai scores for every column.

Metrics, all reported per ply band (0-7, 8-15, 16-23, 24-31, 32-41):
* **value-preserving (VP)**: the argmax move has the same game-theoretic sign (win/draw/loss) as
  the best move - the bar B3;
* VP on **non-trivial** positions only (legal moves differ in sign) - reported, no bar;
* distance-exact top-1 (the argmax is a fastest win / slowest loss) - reported, no bar.

## Controls (hard gates for any model reported as passing)

* Option-order invariance: permuting the option slots never changes the choice (must be 1.0).
* Mirror consistency: the choice on the mirrored board is the mirrored choice, where the mirrored
  position's best set is itself mirrored (report the rate; < 0.99 is a finding, not a pass).
* Owner shuffle: reassigning every stone's owner (heights kept) must *lower* VP on non-trivial
  positions by >= 10 points - the first version's failure mode ("does not read the board") must be absent.
* Export parity: PyTorch vs ONNX argmax agreement 1.0 on the eval set; int8, if shipped, reported.

## On-policy reporting (every game cell)

Every model decision in a game (noise moves excluded) is labelled afterwards by the exact solver:
on-policy VP overall, non-trivial, and per ply band.

## The previous version

The v1 demo model (`onepass-c4-8x24.onnx`) is played through the same protocol with the `onnx-v1`
player (its own text rendering and option shape), and the browser arena compares old and new
directly.
