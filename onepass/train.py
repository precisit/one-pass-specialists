"""Train a one-pass specialist with the vendored upstream trainer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._vendor import ensure_vendor

ensure_vendor()  # puts the vendored MIT upstream code on sys.path (see THIRD_PARTY_NOTICES.md)

from training.train import train  # noqa: E402  (path set above)

MODEL_CONFIG = {
    "encoder": "tinyx",
    "width": 128,
    "rank": 128,
    "layers": 2,
    "heads": 4,
    "context_tokens": 224,
    "option_tokens": 96,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sv"))
    parser.add_argument("--output", type=Path, default=Path("runs/sv-tinyx/model"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    result = train(
        dict(MODEL_CONFIG),
        args.data / "train.jsonl",
        args.data / "validation.jsonl",
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        warmup=0.05,
        workers=0,
        deterministic=False,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "history"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
