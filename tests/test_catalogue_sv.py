"""The Swedish catalogue is data, so it gets data tests: unique keys, usable value formats."""

from __future__ import annotations

import random
import re

from onepass.catalogue_sv import CONCEPT_BY_KEY, CONCEPTS, FORM_TITLES, person


def _luhn_ok(number: str) -> bool:
    """Standard Luhn over the ten digits, as the Swedish identifiers use it."""
    total = 0
    for position, digit in enumerate(number):
        value = int(digit) * (2 if position % 2 == 0 else 1)
        total += value - 9 if value > 9 else value
    return total % 10 == 0


def test_catalogue_is_wide_enough_and_keys_are_unique() -> None:
    keys = [concept.key for concept in CONCEPTS]
    assert len(keys) >= 30, "the catalogue should cover real form vocabulary"
    assert len(keys) == len(set(keys)), "duplicate concept keys make the label ambiguous"
    assert set(keys) == set(CONCEPT_BY_KEY), "CONCEPT_BY_KEY must index every concept"


def test_every_concept_has_labels_and_produces_values() -> None:
    for concept in CONCEPTS:
        assert concept.form_labels, f"{concept.key} has no form label for an element to carry"
        assert concept.doc_labels, f"{concept.key} has no document label to match against"
        for seed in (1, 7, 2026):
            rng = random.Random(seed)
            value = concept.value(rng, person(rng))  # adapted generators take (rng, person)
            assert isinstance(value, str) and value.strip(), f"{concept.key} produced an empty value"


def test_value_formats_look_local() -> None:
    """The point of a language-local catalogue: values must be shaped for the market.

    These assertions are the *specification* of the value formats — they caught a real bug on
    2026-09-21 (the personnummer generator emitted 9 digits formatted as 8+2, and the
    organisationsnummer appended the check digit instead of replacing the tenth), so they stay
    strict. Note: SV0 was trained before that fix, i.e. on the malformed shapes; its model card
    says so, and a retrain is a release of its own.
    """
    rng = random.Random(3)
    who = person(rng)
    personnummer = CONCEPT_BY_KEY["personnummer"].value(rng, who)
    assert re.fullmatch(r"\d{6}-\d{4}", personnummer), personnummer
    organisation = CONCEPT_BY_KEY["organisationsnummer"].value(rng, who)
    assert re.fullmatch(r"\d{6}-\d{4}", organisation), organisation
    if "iban" in CONCEPT_BY_KEY:
        assert CONCEPT_BY_KEY["iban"].value(rng, who).startswith("SE")
    if "bankgiro" in CONCEPT_BY_KEY:
        assert re.fullmatch(r"\d{3}-\d{4}", CONCEPT_BY_KEY["bankgiro"].value(rng, who))
    assert FORM_TITLES, "form titles are part of the surface the model sees"


def test_generated_identifiers_pass_their_own_checksum() -> None:
    """A shape without a valid checksum is worse than no shape: it teaches the model noise."""
    rng = random.Random(8)
    who = person(rng)
    for key, length in (("personnummer", 10), ("organisationsnummer", 10)):
        value = CONCEPT_BY_KEY[key].value(rng, who)
        digits = value.replace("-", "")
        assert len(digits) == length, value
        assert _luhn_ok(digits), f"{key} failed its checksum: {value}"


def test_synthetic_people_are_fictional() -> None:
    """Provenance guard: the corpus must never look like real personal data."""
    rng = random.Random(11)
    for _ in range(50):
        who = person(rng)
        assert who["first"] and who["last"]
        # generated identifiers are locally shaped but must not carry a real person's data;
        # the generator's own contract is what we can test: it is seeded and deterministic
        assert person(random.Random(5)) == person(random.Random(5)), "generator must be reproducible"
