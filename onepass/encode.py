"""The byte-level input contract, shared by the PyTorch and Core ML paths.

Everything a one-pass specialist sees is bytes, because there is no tokenizer:

    context:  the document text plus the UI element, truncated to `context_bytes`
    options:  the candidate decisions, each truncated to `option_bytes`
    mask:     1 for real options, 0 for the padding that fills the export's fixed ceiling

Encoding rule (must match training exactly): take the UTF-8 bytes, truncate, **add 1 to every
byte** and zero-pad — so 0 is unambiguously padding, never data. Getting this wrong is the
most likely cause of a model that looks broken; `prepare_inputs` is the single place that
implements it, and both the export and the Core ML consumer call it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InputLimits:
    """Fixed shapes of an exported package. The option ceiling is baked into the export."""

    context_bytes: int = 224
    option_bytes: int = 96
    max_options: int = 40


class TooManyOptions(ValueError):
    """Raised instead of silently truncating: an oversized input must fail closed."""


def byte_ids(text: str, length: int) -> list[int]:
    """UTF-8 byte ids for `text`: each byte + 1, truncated to `length` bytes."""
    return [byte + 1 for byte in text.encode("utf-8", errors="replace")[:length]]


def pad(ids: list[int], length: int) -> list[int]:
    if len(ids) > length:
        raise ValueError(f"cannot pad {len(ids)} ids into {length} slots")
    return ids + [0] * (length - len(ids))


def encode_context(context: str, limits: InputLimits) -> np.ndarray:
    return np.array([pad(byte_ids(context, limits.context_bytes), limits.context_bytes)], dtype=np.int32)


def encode_options(options: tuple[str, ...] | list[str], limits: InputLimits) -> tuple[np.ndarray, np.ndarray]:
    """Return (option_ids (1, max_options, option_bytes), option_mask (1, max_options))."""
    if len(options) > limits.max_options:
        raise TooManyOptions(
            f"{len(options)} options exceed this export's ceiling of {limits.max_options}; "
            "re-export with a larger --max-options rather than truncating the input"
        )
    ids = np.zeros((1, limits.max_options, limits.option_bytes), dtype=np.int32)
    mask = np.zeros((1, limits.max_options), dtype=np.int32)
    for index, option in enumerate(options):
        ids[0, index] = np.array(pad(byte_ids(option, limits.option_bytes), limits.option_bytes), dtype=np.int32)
        mask[0, index] = 1
    return ids, mask


def prepare_inputs(context: str, options: tuple[str, ...] | list[str], limits: InputLimits) -> dict[str, np.ndarray]:
    """Inputs for `coremltools`' `MLModel.predict`, in the export's fixed shapes."""
    option_ids, option_mask = encode_options(options, limits)
    return {
        "context_ids": encode_context(context, limits),
        "option_ids": option_ids,
        "option_mask": option_mask,
    }


def score_options(logits: np.ndarray, option_count: int) -> list[float]:
    """Slice a fixed-ceiling logit vector down to the real options."""
    flat = np.asarray(logits).reshape(-1)
    if option_count > flat.size:
        raise ValueError(f"{option_count} options but only {flat.size} logits")
    return [float(value) for value in flat[:option_count]]


def choose(logits: np.ndarray, option_count: int) -> int:
    """Argmax over the real options only (padding must never win)."""
    scores = score_options(logits, option_count)
    return max(range(len(scores)), key=scores.__getitem__)
