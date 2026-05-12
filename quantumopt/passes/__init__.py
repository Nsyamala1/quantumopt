"""quantumopt.passes — Trotter-aware and domain-specific transpiler passes."""

from quantumopt.passes.trotter_fusion import (
    TrotterBlock,
    TrotterBlockDetector,
    TrotterStepFusion,
    compile_trotter,
)
from quantumopt.passes.givens_synthesis import (
    GivensHoppingBlock,
    GivensRotationSynthesis,
    givens_hopping_circuit,
    compile_givens,
)

__all__ = [
    "TrotterBlock",
    "TrotterBlockDetector",
    "TrotterStepFusion",
    "compile_trotter",
    "GivensHoppingBlock",
    "GivensRotationSynthesis",
    "givens_hopping_circuit",
    "compile_givens",
]
