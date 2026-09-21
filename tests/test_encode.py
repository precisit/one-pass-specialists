"""The byte-input contract: shapes, padding, and failing closed on oversized inputs."""

from __future__ import annotations

import numpy as np
import pytest

from onepass.encode import (
    InputLimits,
    TooManyOptions,
    byte_ids,
    choose,
    encode_options,
    pad,
    prepare_inputs,
)

LIMITS = InputLimits(context_bytes=16, option_bytes=8, max_options=4)


def test_byte_ids_are_offset_and_truncated() -> None:
    assert byte_ids("a", 4) == [98]  # "a" = 0x61 -> 0x62, so 0 is never data
    assert byte_ids("", 4) == []
    assert len(byte_ids("abcdefghij", 4)) == 4  # truncated to the limit
    assert byte_ids("å", 4) == [0xC3 + 1, 0xA5 + 1]  # UTF-8 multibyte stays byte-level


def test_pad_is_zeroes_and_refuses_overflow() -> None:
    assert pad([1, 2], 5) == [1, 2, 0, 0, 0]
    with pytest.raises(ValueError):
        pad([1, 2, 3], 2)


def test_prepare_inputs_shapes_and_dtypes() -> None:
    inputs = prepare_inputs("kontor", ("fyll A", "kryssa", "hoppa över"), LIMITS)
    assert inputs["context_ids"].shape == (1, 16)
    assert inputs["option_ids"].shape == (1, 4, 8)
    assert inputs["option_mask"].shape == (1, 4)
    for value in inputs.values():
        assert value.dtype == np.int32
    assert inputs["option_mask"][0].tolist() == [1, 1, 1, 0]
    assert inputs["option_ids"][0, 3].tolist() == [0] * 8  # padding slots are zero


def test_oversized_option_lists_fail_closed() -> None:
    with pytest.raises(TooManyOptions):
        encode_options([f"fyll {index}" for index in range(LIMITS.max_options + 1)], LIMITS)


def test_choose_ignores_padding_slots() -> None:
    logits = np.array([[1.0, 2.0, 3.0, 99.0]])  # 99.0 sits on a padded slot
    assert choose(logits, option_count=3) == 2
    with pytest.raises(ValueError):
        choose(logits, option_count=5)
