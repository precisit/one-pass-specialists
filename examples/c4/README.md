# Worked example: a one-pass specialist that plays Connect Four

The model is published as [`precisit/onepass-c4`](https://huggingface.co/precisit/onepass-c4) (v2, MIT)
and playable at [precisit.github.io/onepass-web/demo/c4](https://precisit.github.io/onepass-web/demo/c4/).

A one-pass specialist gets a context and a list of options and returns one score per option in a
single forward pass. Here the context is the board (44 bytes) and the options are the legal columns
(`column 1` … `column 7`); the model plays the highest score. **No search, no rules** — the model is
this repository's own option scorer (`vendor/cua_s1/model.py`, `TinyTransformerScorer`), unchanged,
with a larger config: 8 layers × 256, 7.38 M parameters.

## Results

Frozen protocol ([`results/PROTOCOL.md`](results/PROTOCOL.md), fixed before training): 200 games per
match from the empty board, colours alternating, **both** players playing a uniformly random move 5 % of
the time (so 200 games are 200 games), 95 % Wilson intervals. Score = (wins + draws / 2) / games.

| opponent | v2 | v2, int8 ONNX (7.8 MB) | v1 (0.7 M, first version) | a perfect player |
| --- | ---: | ---: | ---: | ---: |
| depth-2 alpha-beta bot | 0.905 [0.856–0.938] | 0.910 | 0.03 | — |
| depth-4 alpha-beta bot | **0.915** [0.868–0.946] | 0.905 | 0.03 | 0.890 |
| depth-6 alpha-beta bot | **0.893** [0.842–0.928] | 0.878 | 0.02 | 0.925 |
| perfect player (exact solver) | 0.475 [0.407–0.544] | — | — | ≈ 0.5 |
| random | 1.000 | — | 0.885 | — |

The last column is the ceiling: the protocol's own random moves cost a perfect player 11 % of the
points against the depth-4 bot. Re-run with seeds 2027 and 2028: depth-4 0.893 / 0.908, depth-6 0.838 /
0.878. Head to head, v2 beats v1 0.985 (and 200–0 in the browser arena).

Move level, on a held-out set of 17 325 positions (≈ 480 per ply, half strong self-play, half games
between weak and strong players, keys excluded from training above ply 11):

| | v2 |
| --- | ---: |
| moves that keep the game-theoretic value (win stays win, draw stays draw) | **98.7 %** |
| … on positions where the choice matters | 97.0 % |
| moves that are a fastest win / slowest loss | 93.5 % |
| drop when every stone's owner is shuffled (heights kept) — it reads the board | −49 points |
| option order permuted — must not matter | 100 % unchanged |
| mirrored board gives the mirrored choice | 77 % (near-ties are not symmetric) |

Every number above is a line in [`results/`](results/) (JSON, one per measurement).

## What made the difference

The first version (v1: same architecture, 2 × 128) lost almost every game. It was trained on ~80 k
positions with at least 26 stones — the only ones a pure-Python solver could label — so it never saw
an exact label in the opening or midgame, where Connect Four is decided. v2 changed the data and the
target, not the idea:

| | v1 | v2 |
| --- | --- | --- |
| labels | Python negamax, endgame only | [`connect-four-ai`](https://github.com/benjaminrall/connect-four-ai) (MIT, Rust): every column of every position, exact |
| positions | ~80 k, mostly random fills | 41.6 M ([TonyCWang/ConnectFour](https://huggingface.co/datasets/TonyCWang/ConnectFour), MIT) + 513 k from weak-vs-strong games |
| target | one optimal move | listwise over all legal columns: softmax of exact scores, win/draw/loss dominant |
| input | text grid + move history, 224 bytes | perspective board, 42 fixed cells, 44 bytes |
| size | 2 × 128 (0.71 M) | 8 × 256 (7.38 M) |

What each step bought, measured along the way: with the v2 data and target even the v1 size reads the
board (0.91 value-preserving on the held-out set after 3 M samples, owner shuffle −28 points; v1: −0.1).
The soft target beat a uniform-over-optimal-set target at every checkpoint. The larger model
mattered more in games than on the held-out set: at 6 000 steps, 6 × 192 scored 0.63 against the
depth-4 bot and 8 × 256 scored 0.90, with validation numbers 0.3 points apart. Longer training fixed the
remaining weak spot (depth-2 bot: 0.64 → 0.91). A round of DAgger (the model plays, the solver labels
every position it reaches) did not move game results further and was not adopted.

## Reproduce

Requirements: Rust (for `c4label`), Python ≥ 3.10 with `numpy pyarrow torch safetensors onnx onnxruntime
huggingface_hub`, optionally `mlx` (Apple silicon; ~3× faster training through `mlx_scorer.py`, an exact
twin of the toolkit model — every checkpoint it saves is re-checked against the PyTorch class).

```bash
cd examples/c4
cargo build --release --manifest-path c4label/Cargo.toml     # exact labeller over connect-four-ai
export C4_WORK=$PWD/work                                    # data, runs; not committed

# 1. verify the teacher (Pons' test sets ship with connect-four-ai under crates/test-data)
python verify_teacher.py pons                               # 6 000 / 6 000 exact
python verify_teacher.py ucibook                            # 67 557 / 67 557 (UCI Connect-4, in $C4_WORK/data/uci)

# 2. corpora: TonyCWang (download CHUNK_0 into $C4_WORK/data/tonyc) + mixed-strength games
python build_corpus.py tonyc --split train --out $C4_WORK/corpus/tonyc-train
python build_corpus.py tonyc --split test  --out $C4_WORK/corpus/tonyc-test
c4label/target/release/c4label --gen 10000 --seed 1 > $C4_WORK/gen/s1.tsv      # also seeds 2, 3, 5
c4label/target/release/c4label --gen 3000 --seed 9001 > $C4_WORK/gen/eval.tsv   # held out
python build_corpus.py gen --lines $C4_WORK/gen/eval.tsv --out $C4_WORK/corpus/gen-eval
python eval_c4.py build-evalset --tonyc-test $C4_WORK/corpus/tonyc-test --gen $C4_WORK/corpus/gen-eval \
    --out $C4_WORK/corpus/evalset
python build_corpus.py gen --lines $C4_WORK/gen/s{1,2,3,5}.tsv --out $C4_WORK/corpus/gen-raw
python build_corpus.py merge --inputs $C4_WORK/corpus/tonyc-train --exclude $C4_WORK/corpus/evalset --out $C4_WORK/corpus/train
python build_corpus.py merge --inputs $C4_WORK/corpus/gen-raw --exclude $C4_WORK/corpus/evalset --out $C4_WORK/corpus/gen
# (validation: a TonyCWang test shard + a second held-out generator seed, eval keys excluded)

# 3. train (MLX; train_c4.py is the same recipe on PyTorch)
python train_mlx.py --train $C4_WORK/corpus/train $C4_WORK/corpus/gen --weights 0.8 0.2 --val $C4_WORK/corpus/val \
    --out $C4_WORK/runs/L --size L --lr 7e-4 --steps 6000 --warmup 500
python train_mlx.py --init $C4_WORK/runs/L/model --train $C4_WORK/corpus/train $C4_WORK/corpus/gen --weights 0.75 0.25 \
    --val $C4_WORK/corpus/val --out $C4_WORK/runs/L2 --lr 3.5e-4 --steps 12000 --warmup 300

# 4. evaluate with the frozen protocol
python eval_c4.py score --checkpoint $C4_WORK/runs/L2/model --evalset $C4_WORK/corpus/evalset --controls
python protocol.py --a model:$C4_WORK/runs/L2/model --b bot:4 --games 200
python protocol.py --a solver --b bot:4 --games 200                          # the ceiling

# 5. export for the browser, with parity gates (and optional int8)
python export_c4.py --checkpoint $C4_WORK/runs/L2/model --evalset $C4_WORK/corpus/evalset --out onepass-c4-v2.onnx
python -c "from onnxruntime.quantization import quantize_dynamic, QuantType; quantize_dynamic('onepass-c4-v2.onnx', 'onepass-c4-v2-int8.onnx', weight_type=QuantType.QInt8)"
python protocol.py --a onnx-v2:onepass-c4-v2-int8.onnx --b bot:4 --games 200     # the shipped file, re-measured
```

Compute for the published model: corpus generation on CPU (~2.4 CPU-seconds per generated game — weak
play reaches positions no opening book covers), training ~2 hours on one Apple M5 Pro (6 000 + 12 000
steps × 1 024 positions).

## Files

| file | what |
| --- | --- |
| `c4label/` | Rust CLI over connect-four-ai: batch labels (all 7 column scores), `--serve`, `--gen` mixed-strength games, `--value` |
| `solver.py` | Python wrapper (batch + persistent oracle) |
| `verify_teacher.py` | the labeller against Pons' test sets, UCI Connect-4, the opening book, and TonyCWang re-solves |
| `encode_c4.py` | the input contract (context bytes, options) — the browser demo implements the same bytes and self-tests them |
| `build_corpus.py` | TonyCWang / generated positions → deduplicated numpy corpora with manifests |
| `model_c4.py`, `train_c4.py` | the toolkit scorer + a batcher; listwise training with ply-balanced sampling and mirroring |
| `mlx_scorer.py`, `train_mlx.py` | the same model and recipe on MLX, parity-checked against the PyTorch class |
| `eval_c4.py`, `protocol.py` | the frozen eval set, move-level metrics and controls; the game protocol |
| `dagger.py` | one DAgger round: student games against a pool, every visited position labelled |
| `export_c4.py` | ONNX export with argmax and cross-shape parity gates; browser self-test rows |
| `engine.py` | board and the hand-written depth-limited bots used as opponents |
| `results/` | the protocol and every measurement as JSON |

## Credits

Labels: [connect-four-ai](https://github.com/benjaminrall/connect-four-ai) (MIT, Benjamin Rall), built
on the techniques of Pascal Pons' solver tutorial. Positions: [TonyCWang/ConnectFour](https://huggingface.co/datasets/TonyCWang/ConnectFour)
(MIT). Verification only: Pascal Pons' public test positions, and the UCI Connect-4 database (John Tromp,
CC BY 4.0). No AGPL code is used or included.
