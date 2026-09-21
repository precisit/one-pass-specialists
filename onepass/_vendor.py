"""Make the vendored MIT upstream code importable.

We vendor `cua_s1` (the model, checkpoint format and collator) and `training` (the training
loop) from https://github.com/trycua/cua/tree/main/libs/cua-s1 @ 9bbfa7d, unmodified, under
`<repo>/vendor/`. Vendoring instead of depending on a released package keeps the exact
revision that produced the published checkpoint reproducible; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent.parent / "vendor"


def ensure_vendor() -> Path:
    """Put the vendored directory on sys.path (idempotent) and return it."""
    if not (VENDOR / "cua_s1").is_dir():
        raise RuntimeError(
            f"vendored upstream code not found at {VENDOR} — the repository is incomplete "
            "(see THIRD_PARTY_NOTICES.md for what should be there)"
        )
    resolved = str(VENDOR)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return VENDOR
