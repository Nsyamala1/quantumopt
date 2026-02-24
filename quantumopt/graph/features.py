"""
quantumopt.graph.features — Gate feature vector generation.

Creates fixed-dimension (20-dim) feature vectors for quantum gates, encoding:
- Gate type (one-hot, 12 dims)
- Qubit indices (normalized, 3 dims)
- Gate parameters (normalized angles, 3 dims)
- Auxiliary features (arity, parametric flag, 2 dims)
"""

import torch
import numpy as np
from typing import List, Optional


# Supported gate types for one-hot encoding (12 types)
GATE_TYPES = [
    "h", "cx", "rz", "ry", "rx",
    "x", "y", "z", "swap", "ccx",
    "measure", "unknown",
]

# Feature vector dimension
FEATURE_DIM = 20

# Number of gate type classes
NUM_GATE_TYPES = len(GATE_TYPES)  # 12

# Number of qubit index slots
NUM_QUBIT_SLOTS = 3

# Number of parameter slots
NUM_PARAM_SLOTS = 3

# Number of auxiliary features
NUM_AUX_FEATURES = 2


def _normalize_angle(angle: float) -> float:
    """Normalize a rotation angle to [0, 1] range.

    Maps angles from [-2π, 2π] to [0, 1] using modular arithmetic.

    Args:
        angle: Rotation angle in radians.

    Returns:
        Normalized value in [0, 1].
    """
    # Normalize to [0, 2π] first, then to [0, 1]
    normalized = (angle % (2 * np.pi)) / (2 * np.pi)
    return float(np.clip(normalized, 0.0, 1.0))


def gate_to_feature_vector(
    gate_name: str,
    qubits: List[int],
    params: Optional[List[float]] = None,
    num_qubits: int = 1,
) -> torch.Tensor:
    """Convert a gate specification to a fixed-dimension feature vector.

    Feature vector layout (20 dimensions):
        [0:12]  — One-hot gate type encoding (12 dims)
        [12:15] — Qubit indices normalized by num_qubits (3 dims, padded)
        [15:18] — Parameter values normalized to [0, 1] (3 dims, padded)
        [18:20] — Auxiliary: [gate_arity / 3.0, is_parametric] (2 dims)

    Args:
        gate_name: Name of the gate (e.g., "h", "cx", "rz").
                   Case-insensitive; unknown gates mapped to "unknown".
        qubits: List of qubit indices this gate acts on.
        params: Optional list of gate parameters (rotation angles in radians).
        num_qubits: Total number of qubits in the circuit (for normalization).

    Returns:
        torch.Tensor of shape [20] with float32 values.

    Examples:
        >>> gate_to_feature_vector("h", [0], num_qubits=3)  # Hadamard on qubit 0
        >>> gate_to_feature_vector("cx", [0, 1], num_qubits=3)  # CNOT on qubits 0,1
        >>> gate_to_feature_vector("rz", [2], params=[1.57], num_qubits=3)  # RZ(π/2)
    """
    if params is None:
        params = []

    features = torch.zeros(FEATURE_DIM, dtype=torch.float32)

    # --- Part 1: One-hot gate type encoding [0:12] ---
    gate_name_lower = gate_name.lower()
    if gate_name_lower in GATE_TYPES:
        gate_idx = GATE_TYPES.index(gate_name_lower)
    else:
        gate_idx = GATE_TYPES.index("unknown")
    features[gate_idx] = 1.0

    # --- Part 2: Qubit indices normalized by num_qubits [12:15] ---
    num_qubits_safe = max(num_qubits, 1)
    for i, qubit in enumerate(qubits[:NUM_QUBIT_SLOTS]):
        features[NUM_GATE_TYPES + i] = qubit / num_qubits_safe

    # --- Part 3: Parameter values normalized to [0, 1] [15:18] ---
    for i, param in enumerate(params[:NUM_PARAM_SLOTS]):
        features[NUM_GATE_TYPES + NUM_QUBIT_SLOTS + i] = _normalize_angle(float(param))

    # --- Part 4: Auxiliary features [18:20] ---
    gate_arity = len(qubits)
    is_parametric = 1.0 if len(params) > 0 else 0.0

    features[NUM_GATE_TYPES + NUM_QUBIT_SLOTS + NUM_PARAM_SLOTS] = gate_arity / 3.0
    features[NUM_GATE_TYPES + NUM_QUBIT_SLOTS + NUM_PARAM_SLOTS + 1] = is_parametric

    return features
