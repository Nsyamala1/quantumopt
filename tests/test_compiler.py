"""Tests for quantumopt.compiler — Main compilation pipeline."""

import pytest
import numpy as np
from qiskit import QuantumCircuit

from quantumopt.compiler import compile, CompileResult
from quantumopt.backends.ibm_backend import compile_for_ibm, _estimate_fidelity
from quantumopt.data.pipeline import _make_vqe_circuit, _make_qaoa_circuit, _make_ghz_circuit


class TestCompileResult:
    """Test CompileResult dataclass."""

    def test_dataclass_creation(self):
        """CompileResult should create with defaults."""
        qc = QuantumCircuit(2)
        result = CompileResult(optimized_circuit=qc)
        assert result.depth_reduction == "0%"
        assert result.gate_reduction == "0%"
        assert result.explanation is None
        assert result.benchmark == {}

    def test_repr(self):
        """CompileResult repr should be readable."""
        qc = QuantumCircuit(2)
        result = CompileResult(
            optimized_circuit=qc,
            depth_reduction="25%",
            original_stats={"depth": 10},
            optimized_stats={"depth": 7},
        )
        repr_str = repr(result)
        assert "25%" in repr_str


class TestCompile:
    """Test compile() function — end-to-end pipeline."""

    def test_basic_compile(self):
        """Compile a simple circuit — should return CompileResult."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        result = compile(qc, hardware="ibm_brisbane", priority="fidelity", explain=False)

        assert isinstance(result, CompileResult)
        assert result.optimized_circuit is not None
        assert result.original_stats["depth"] > 0
        assert result.compile_time > 0

    def test_vqe_circuit_compile(self):
        """Compile a VQE circuit — common use case for research labs."""
        qc = _make_vqe_circuit(num_qubits=4, reps=2)
        result = compile(qc, hardware="ibm_brisbane", priority="fidelity", explain=False)

        assert isinstance(result, CompileResult)
        assert result.optimized_circuit.num_qubits >= 4
        assert "depth" in result.original_stats

    def test_qaoa_circuit_compile(self):
        """Compile a QAOA circuit."""
        qc = _make_qaoa_circuit(num_qubits=4, p=1)
        result = compile(qc, hardware="ibm_brisbane", priority="depth", explain=False)

        assert isinstance(result, CompileResult)
        assert result.depth_reduction is not None

    def test_ghz_circuit_compile(self):
        """Compile a GHZ circuit."""
        qc = _make_ghz_circuit(num_qubits=5)
        result = compile(qc, hardware="ibm_brisbane", explain=False)

        assert isinstance(result, CompileResult)
        assert result.optimized_circuit is not None

    def test_compile_with_fallback_explanation(self):
        """Compile with explain=True should produce fallback explanation (no API key)."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        result = compile(qc, explain=True)

        # Without API key, should get fallback explanation
        assert result.explanation is not None
        assert "optimized" in result.explanation.lower() or "reduced" in result.explanation.lower()

    def test_compile_benchmark_dict(self):
        """Compile should produce a benchmark dict."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        result = compile(qc, explain=False)

        assert "original_depth" in result.benchmark
        assert "optimized_depth" in result.benchmark
        assert "depth_reduction" in result.benchmark
        assert "hardware" in result.benchmark

    def test_different_priorities(self):
        """Different priorities should all work."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(np.pi / 4, 2)

        for priority in ["fidelity", "depth", "speed"]:
            result = compile(qc, priority=priority, explain=False)
            assert isinstance(result, CompileResult)


class TestIBMBackend:
    """Test IBM backend compilation."""

    def test_compile_for_ibm_basic(self):
        """compile_for_ibm should return circuit + stats."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        compiled, stats = compile_for_ibm(qc)

        assert isinstance(compiled, QuantumCircuit)
        assert "depth" in stats
        assert "gate_count" in stats
        assert "estimated_fidelity" in stats
        assert 0.0 <= stats["estimated_fidelity"] <= 1.0

    def test_fidelity_estimation(self):
        """Fidelity should decrease with more gates."""
        qc_small = QuantumCircuit(2)
        qc_small.h(0)

        qc_large = QuantumCircuit(5)
        for i in range(5):
            qc_large.h(i)
        for i in range(4):
            qc_large.cx(i, i + 1)

        fid_small = _estimate_fidelity(qc_small)
        fid_large = _estimate_fidelity(qc_large)

        assert fid_small > fid_large  # more gates = lower fidelity
