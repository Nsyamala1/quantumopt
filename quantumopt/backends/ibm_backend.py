"""
quantumopt.backends.ibm_backend — IBM Quantum hardware compilation.

Provides circuit compilation and optimization targeting IBM Quantum
backends using Qiskit's transpiler with custom optimization passes.
Falls back to FakeBrisbane simulator when no IBM token is available.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    Optimize1qGatesDecomposition,
    CommutativeCancellation,
    InverseCancellation,
)
from qiskit.circuit.library import CXGate

logger = logging.getLogger(__name__)

# Average gate error rates for fidelity estimation (approximate)
GATE_ERROR_RATES = {
    "cx": 0.01,     # ~1% per CX gate
    "ecr": 0.008,   # ~0.8% per ECR gate
    "cz": 0.01,     # ~1% per CZ gate
    "sx": 0.0003,   # ~0.03% per SX gate
    "rz": 0.0001,   # ~0.01% per RZ gate (virtual)
    "x": 0.0003,    # ~0.03% per X gate
    "id": 0.0001,   # ~0.01% for identity
    "measure": 0.02, # ~2% readout error
}


def _get_fake_backend(backend_name: str):
    """Get a FakeBackend for simulation when no IBM token is available.

    Args:
        backend_name: Name of the backend (e.g., "ibm_brisbane").

    Returns:
        A Qiskit fake backend object.
    """
    try:
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        return FakeBrisbane()
    except ImportError:
        logger.warning("Could not import FakeBrisbane, using generic coupling map")
        return None


def _estimate_fidelity(circuit: QuantumCircuit) -> float:
    """Estimate circuit fidelity based on gate counts and error rates.

    Simple multiplicative model:
        fidelity = ∏(1 - error_rate_i) for each gate i

    Args:
        circuit: The compiled quantum circuit.

    Returns:
        Estimated fidelity as a float in [0, 1].
    """
    fidelity = 1.0
    ops = circuit.count_ops()

    for gate_name, count in ops.items():
        error_rate = GATE_ERROR_RATES.get(gate_name, 0.001)
        fidelity *= (1 - error_rate) ** count

    return round(max(0.0, fidelity), 6)


def _get_circuit_stats(circuit: QuantumCircuit) -> Dict[str, Any]:
    """Extract detailed statistics from a circuit.

    Returns:
        Dict with depth, gate_count, cx_count, num_qubits, and gate breakdown.
    """
    ops = circuit.count_ops()
    two_qubit_gates = ops.get("cx", 0) + ops.get("ecr", 0) + ops.get("cz", 0)

    return {
        "depth": circuit.depth(),
        "gate_count": sum(ops.values()),
        "two_qubit_gate_count": two_qubit_gates,
        "num_qubits": circuit.num_qubits,
        "gate_breakdown": dict(ops),
    }


def _apply_extra_passes(circuit: QuantumCircuit) -> QuantumCircuit:
    """Apply additional optimization passes beyond Qiskit's default transpiler.

    Passes applied:
        - Optimize1qGatesDecomposition: Simplify single-qubit gate sequences
        - CommutativeCancellation: Cancel commuting gates
        - CXCancellation: Remove adjacent inverse CX pairs

    Args:
        circuit: Already-transpiled circuit.

    Returns:
        Further optimized circuit.
    """
    pm = PassManager([
        Optimize1qGatesDecomposition(),
        CommutativeCancellation(),
        InverseCancellation([CXGate()]),
    ])
    return pm.run(circuit)


def compile_for_ibm(
    circuit: QuantumCircuit,
    backend_name: str = "ibm_brisbane",
    optimization_level: int = 3,
) -> Tuple[QuantumCircuit, Dict[str, Any]]:
    """Compile a circuit for IBM Quantum hardware.

    Uses Qiskit transpiler with IBM coupling maps and applies additional
    optimization passes for further gate reduction.

    Falls back to FakeBrisbane if IBM_QUANTUM_TOKEN is not available.

    Args:
        circuit: Qiskit QuantumCircuit to compile.
        backend_name: IBM backend name (default: "ibm_brisbane").
        optimization_level: Qiskit optimization level 0-3 (default: 3).

    Returns:
        Tuple of (compiled_circuit, stats_dict) where stats_dict contains:
            depth, gate_count, two_qubit_gate_count, estimated_fidelity,
            num_qubits, gate_breakdown.
    """
    backend = None
    ibm_token = os.environ.get("IBM_QUANTUM_TOKEN")

    # Try to get a real backend if token is available
    if ibm_token:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_token)
            backend = service.backend(backend_name)
            logger.info(f"Using real IBM backend: {backend_name}")
        except Exception as e:
            logger.warning(f"Failed to connect to IBM Quantum: {e}")
            backend = None

    # Fall back to fake backend
    if backend is None:
        backend = _get_fake_backend(backend_name)
        if backend:
            logger.info(f"Using fake backend: FakeBrisbane")
        else:
            logger.info("Using generic transpilation (no backend)")

    # Transpile with Qiskit
    if backend:
        compiled = transpile(
            circuit,
            backend=backend,
            optimization_level=optimization_level,
        )
    else:
        # Generic transpilation without specific backend
        compiled = transpile(
            circuit,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            optimization_level=optimization_level,
        )

    # Apply additional optimization passes
    try:
        compiled = _apply_extra_passes(compiled)
    except Exception as e:
        logger.warning(f"Extra optimization passes failed: {e}")

    # Compute stats
    stats = _get_circuit_stats(compiled)
    stats["estimated_fidelity"] = _estimate_fidelity(compiled)
    stats["backend"] = backend_name
    stats["optimization_level"] = optimization_level

    return compiled, stats
