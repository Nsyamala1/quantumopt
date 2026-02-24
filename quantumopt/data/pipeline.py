"""
quantumopt.data.pipeline — Dataset generation pipeline.

Generates training datasets from quantum circuit benchmarks by creating
circuits of various types (VQE, QAOA, QFT, Grover, GHZ), running Qiskit
transpiler optimization, and recording improvement metrics for GNN training.
"""

import json
import os
import random
import logging
import numpy as np
from typing import Dict, Any, List, Optional

from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    EfficientSU2,
    QFT as QiskitQFT,
    GroverOperator,
)
from qiskit.compiler import transpile

logger = logging.getLogger(__name__)


# ─── Circuit Generators ─────────────────────────────────────────────────

def _make_vqe_circuit(num_qubits: int, reps: int = 2) -> QuantumCircuit:
    """Generate a VQE-style variational circuit (EfficientSU2 ansatz).

    Args:
        num_qubits: Number of qubits (3–10).
        reps: Number of repetition layers.

    Returns:
        Parameterized VQE circuit with random parameter values bound.
    """
    ansatz = EfficientSU2(num_qubits, reps=reps, entanglement="linear")
    # Bind random parameter values
    param_values = np.random.uniform(0, 2 * np.pi, len(ansatz.parameters))
    bound = ansatz.assign_parameters(dict(zip(ansatz.parameters, param_values)))
    bound.name = "vqe"
    return bound


def _make_qaoa_circuit(num_qubits: int, p: int = 2) -> QuantumCircuit:
    """Generate a QAOA-style circuit for MaxCut.

    Args:
        num_qubits: Number of qubits (3–10).
        p: Number of QAOA layers.

    Returns:
        QAOA circuit with random gamma/beta angles.
    """
    qc = QuantumCircuit(num_qubits)
    qc.name = "qaoa"

    # Initial superposition
    for i in range(num_qubits):
        qc.h(i)

    for layer in range(p):
        gamma = random.uniform(0, 2 * np.pi)
        beta = random.uniform(0, np.pi)

        # Problem unitary (random MaxCut-like edges)
        for i in range(num_qubits - 1):
            if random.random() > 0.3:  # ~70% edge probability
                qc.cx(i, i + 1)
                qc.rz(gamma, i + 1)
                qc.cx(i, i + 1)

        # Mixer unitary
        for i in range(num_qubits):
            qc.rx(2 * beta, i)

    return qc


def _make_qft_circuit(num_qubits: int) -> QuantumCircuit:
    """Generate a Quantum Fourier Transform circuit.

    Args:
        num_qubits: Number of qubits (3–10).

    Returns:
        QFT circuit.
    """
    qft = QiskitQFT(num_qubits)
    qc = qft.decompose()
    qc.name = "qft"
    return qc


def _make_grover_circuit(num_qubits: int) -> QuantumCircuit:
    """Generate a Grover's algorithm circuit (with a random oracle).

    Args:
        num_qubits: Number of qubits (2–6, kept small for sanity).

    Returns:
        Grover circuit with oracle + diffusion.
    """
    num_qubits = min(num_qubits, 6)  # keep manageable

    # Random oracle marking a single basis state
    oracle = QuantumCircuit(num_qubits)
    target_state = random.randint(0, 2**num_qubits - 1)
    target_bits = format(target_state, f"0{num_qubits}b")

    # X gates on qubits where target bit is 0
    for i, bit in enumerate(reversed(target_bits)):
        if bit == "0":
            oracle.x(i)

    # Multi-controlled Z via CX ladder
    if num_qubits >= 2:
        oracle.h(num_qubits - 1)
        if num_qubits == 2:
            oracle.cx(0, 1)
        else:
            oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        oracle.h(num_qubits - 1)

    # Undo X gates
    for i, bit in enumerate(reversed(target_bits)):
        if bit == "0":
            oracle.x(i)

    # Build full Grover circuit
    qc = QuantumCircuit(num_qubits)
    qc.name = "grover"
    qc.h(range(num_qubits))

    # Number of iterations: approx √(2^n)
    num_iters = max(1, int(np.sqrt(2**num_qubits) * np.pi / 4))
    num_iters = min(num_iters, 3)  # cap for training efficiency

    for _ in range(num_iters):
        qc.compose(oracle, inplace=True)
        # Diffusion operator
        qc.h(range(num_qubits))
        qc.x(range(num_qubits))
        qc.h(num_qubits - 1)
        if num_qubits == 2:
            qc.cx(0, 1)
        else:
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        qc.x(range(num_qubits))
        qc.h(range(num_qubits))

    return qc


def _make_ghz_circuit(num_qubits: int) -> QuantumCircuit:
    """Generate a GHZ state preparation circuit.

    Args:
        num_qubits: Number of qubits (3–20).

    Returns:
        GHZ circuit: H on qubit 0, then CX chain.
    """
    qc = QuantumCircuit(num_qubits)
    qc.name = "ghz"
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc


# Map of circuit type → generator function
CIRCUIT_GENERATORS = {
    "vqe": _make_vqe_circuit,
    "qaoa": _make_qaoa_circuit,
    "qft": _make_qft_circuit,
    "grover": _make_grover_circuit,
    "ghz": _make_ghz_circuit,
}


# ─── Dataset Generation ─────────────────────────────────────────────────

def _get_circuit_stats(qc: QuantumCircuit) -> Dict[str, Any]:
    """Extract statistics from a circuit.

    Returns:
        Dict with depth, gate_count, cx_count, and num_qubits.
    """
    ops = qc.count_ops()
    return {
        "depth": qc.depth(),
        "gate_count": sum(ops.values()),
        "cx_count": ops.get("cx", 0) + ops.get("ecr", 0) + ops.get("cz", 0),
        "num_qubits": qc.num_qubits,
    }


def generate_dataset(
    num_circuits: int = 1000,
    output_path: str = "data/circuits/",
    circuit_types: Optional[List[str]] = None,
    min_qubits: int = 3,
    max_qubits: int = 10,
    seed: int = 42,
) -> str:
    """Generate a labeled dataset for GNN training.

    For each circuit:
        1. Generate a random circuit of a chosen type (VQE, QAOA, QFT, Grover, GHZ)
        2. Run Qiskit transpiler at optimization_level=3 for a generic backend
        3. Record: original depth, optimized depth, gate counts, improvement ratio
        4. Save as JSON dataset

    Args:
        num_circuits: Number of circuits to generate (default: 1000).
        output_path: Directory to save the dataset (default: "data/circuits/").
        circuit_types: Circuit types to include (default: all supported types).
        min_qubits: Minimum qubits per circuit (default: 3).
        max_qubits: Maximum qubits per circuit (default: 10).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        Path to the saved dataset JSON file.
    """
    random.seed(seed)
    np.random.seed(seed)

    if circuit_types is None:
        circuit_types = list(CIRCUIT_GENERATORS.keys())

    os.makedirs(output_path, exist_ok=True)
    dataset_path = os.path.join(output_path, "training_dataset.json")

    dataset = []
    circuits_per_type = num_circuits // len(circuit_types)
    remainder = num_circuits % len(circuit_types)

    logger.info(f"Generating {num_circuits} circuits ({circuits_per_type} per type)...")

    for type_idx, ctype in enumerate(circuit_types):
        gen_func = CIRCUIT_GENERATORS[ctype]
        count = circuits_per_type + (1 if type_idx < remainder else 0)

        for i in range(count):
            try:
                num_qubits = random.randint(min_qubits, max_qubits)
                # Grover needs fewer qubits for tractability
                if ctype == "grover":
                    num_qubits = min(num_qubits, 6)

                qc = gen_func(num_qubits)
                original_stats = _get_circuit_stats(qc)

                # Transpile at optimization_level=3
                optimized = transpile(
                    qc,
                    basis_gates=["cx", "id", "rz", "sx", "x"],
                    coupling_map=None,  # all-to-all for now
                    optimization_level=3,
                )
                optimized_stats = _get_circuit_stats(optimized)

                # Compute improvement ratio (depth reduction)
                if original_stats["depth"] > 0:
                    improvement_ratio = 1.0 - (
                        optimized_stats["depth"] / original_stats["depth"]
                    )
                else:
                    improvement_ratio = 0.0

                # Clamp to [0, 1] — sometimes transpiler can increase depth
                improvement_ratio = max(0.0, min(1.0, improvement_ratio))

                entry = {
                    "circuit_type": ctype,
                    "circuit_qasm": qc.qasm() if hasattr(qc, 'qasm') else str(qc),
                    "original_stats": original_stats,
                    "optimized_stats": optimized_stats,
                    "improvement_ratio": round(improvement_ratio, 4),
                }
                dataset.append(entry)

            except Exception as e:
                logger.warning(f"Failed to generate {ctype} circuit {i}: {e}")
                continue

    # Save dataset
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)

    logger.info(f"Dataset saved: {len(dataset)} circuits → {dataset_path}")
    return dataset_path


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load a previously generated dataset.

    Args:
        dataset_path: Path to the dataset JSON file.

    Returns:
        List of circuit data dicts.
    """
    with open(dataset_path, "r") as f:
        return json.load(f)
