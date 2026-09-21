"""Generated corpora must be reproducible and split by form, not by row.

The split rule hashes the *form signature* of an episode, so every decision about one form
lands in exactly one split. If that ever breaks, held-out scores become meaningless — which
is why it is a test and not a paragraph in the README.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from onepass.synth import write_splits

SPLITS = ("train", "validation", "test")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate(directory: Path, episodes: int = 120, seed: int = 2026) -> dict:
    return write_splits(directory, episodes=episodes, seed=seed)


def test_splits_exist_and_rows_are_wellformed(tmp_path: Path) -> None:
    generate(tmp_path)
    for split in SPLITS:
        rows = read_rows(tmp_path / f"{split}.jsonl")
        assert rows, f"{split} split is empty"
        for row in rows:
            assert set(row) >= {"context", "options", "label", "meta"}
            assert row["context"].strip()
            assert 0 <= row["label"] < len(row["options"])
            assert len(set(row["options"])) == len(row["options"]), "duplicate options are ambiguous"
            assert row["meta"]["action"] in {"fill", "check", "click", "skip"}


def test_splits_are_form_disjoint(tmp_path: Path) -> None:
    generate(tmp_path)
    signatures = {
        split: {row["meta"]["form_signature"] for row in read_rows(tmp_path / f"{split}.jsonl")}
        for split in SPLITS
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = signatures[left] & signatures[right]
        assert not overlap, f"{left}/{right} share form signatures: {sorted(overlap)[:3]}"


def test_generation_is_deterministic_and_hash_bound(tmp_path: Path) -> None:
    first = generate(tmp_path / "a", episodes=60)
    second = generate(tmp_path / "b", episodes=60)
    assert (tmp_path / "a" / "train.jsonl").read_bytes() == (tmp_path / "b" / "train.jsonl").read_bytes()
    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 2026
    for split in SPLITS:
        digest = hashlib.sha256((tmp_path / "a" / f"{split}.jsonl").read_bytes()).hexdigest()
        assert manifest["splits"][split]["sha256"] == digest, f"{split}: manifest hash is stale"
        assert manifest["splits"][split]["rows"] == first["rows"][split]
