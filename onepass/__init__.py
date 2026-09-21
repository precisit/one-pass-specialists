"""One-pass specialists: train small models that score a supplied option list in a single
forward pass, instead of generating text.

Pipeline (all runnable on a laptop):

    onepass.synth            generate a synthetic corpus from a concept catalogue
    onepass.train            train the tinyx encoder + option-attention head
    onepass.evaluate         score it: top-1, per-action, ECE, shuffled control, silent skips
    onepass.export_coreml    export to Core ML (fp16 / int8 / int4) with a parity check
    onepass.coreml_evaluate  measure the exported packages, including ANE vs CPU latency

The architecture, trainer and evaluation metrics are vendored, unmodified, from Cua's MIT
`libs/cua-s1` (see THIRD_PARTY_NOTICES.md). The catalogue, generator adaptation, metrics
wrappers and Core ML tooling are ours.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
