"""Tests for quantumopt.graph — Circuit encoder and gate features."""

import pytest
import torch
import numpy as np
from qiskit import QuantumCircuit
from quantumopt.graph.encoder import circuit_to_pyg_graph
from quantumopt.graph.features import (
    gate_to_feature_vector,
    GATE_TYPES,
    FEATURE_DIM,
    _normalize_angle,
)


# ─── Feature Vector Tests ───────────────────────────────────────────────

class TestGateFeatureVector:
    """Test gate_to_feature_vector() function."""

    def test_output_shape(self):
        """Feature vector should be 20-dimensional."""
        vec = gate_to_feature_vector("h", [0], num_qubits=3)
        assert vec.shape == (FEATURE_DIM,)
        assert vec.dtype == torch.float32

    def test_hadamard_one_hot(self):
        """Hadamard gate should have index 0 set to 1.0."""
        vec = gate_to_feature_vector("h", [0], num_qubits=3)
        assert vec[0] == 1.0  # h is at index 0
        assert vec[1:12].sum() == 0.0  # all other gate types are 0

    def test_cx_one_hot(self):
        """CX gate should have index 1 set to 1.0."""
        vec = gate_to_feature_vector("cx", [0, 1], num_qubits=3)
        assert vec[1] == 1.0
        assert vec[0] == 0.0

    def test_unknown_gate_mapping(self):
        """Unknown gates should map to the 'unknown' category."""
        vec = gate_to_feature_vector("weird_custom_gate", [0], num_qubits=3)
        unknown_idx = GATE_TYPES.index("unknown")
        assert vec[unknown_idx] == 1.0

    def test_qubit_indices_normalized(self):
        """Qubit indices should be normalized by num_qubits."""
        vec = gate_to_feature_vector("cx", [1, 2], num_qubits=4)
        # Qubit features at positions 12, 13, 14
        assert vec[12] == pytest.approx(1 / 4)  # qubit 1 / 4
        assert vec[13] == pytest.approx(2 / 4)  # qubit 2 / 4
        assert vec[14] == 0.0  # no third qubit

    def test_rotation_parameters(self):
        """Rotation parameters should be normalized to [0, 1]."""
        angle = np.pi / 2  # π/2
        vec = gate_to_feature_vector("rz", [0], params=[angle], num_qubits=3)
        # Param features at positions 15, 16, 17
        expected = (angle % (2 * np.pi)) / (2 * np.pi)
        assert vec[15] == pytest.approx(expected, abs=1e-5)
        assert vec[16] == 0.0  # no second param
        assert vec[17] == 0.0  # no third param

    def test_auxiliary_features(self):
        """Auxiliary features: arity/3.0 and is_parametric flag."""
        # Single qubit, no params
        vec_h = gate_to_feature_vector("h", [0], num_qubits=3)
        assert vec_h[18] == pytest.approx(1 / 3.0)  # arity=1
        assert vec_h[19] == 0.0  # not parametric

        # Two qubit, no params
        vec_cx = gate_to_feature_vector("cx", [0, 1], num_qubits=3)
        assert vec_cx[18] == pytest.approx(2 / 3.0)  # arity=2
        assert vec_cx[19] == 0.0  # not parametric

        # Single qubit, with params
        vec_rz = gate_to_feature_vector("rz", [0], params=[1.0], num_qubits=3)
        assert vec_rz[18] == pytest.approx(1 / 3.0)  # arity=1
        assert vec_rz[19] == 1.0  # parametric

    def test_case_insensitive(self):
        """Gate names should be case-insensitive."""
        vec_lower = gate_to_feature_vector("h", [0], num_qubits=3)
        vec_upper = gate_to_feature_vector("H", [0], num_qubits=3)
        assert torch.allclose(vec_lower, vec_upper)

    def test_all_gate_types_encode(self):
        """Every known gate type should produce a valid feature vector."""
        for gate_type in GATE_TYPES:
            if gate_type == "unknown":
                continue
            qubits = [0] if gate_type not in ("cx", "swap") else [0, 1]
            if gate_type == "ccx":
                qubits = [0, 1, 2]
            vec = gate_to_feature_vector(gate_type, qubits, num_qubits=5)
            assert vec.shape == (FEATURE_DIM,)
            assert not torch.isnan(vec).any()


class TestNormalizeAngle:
    """Test angle normalization utility."""

    def test_zero(self):
        assert _normalize_angle(0.0) == pytest.approx(0.0, abs=1e-7)

    def test_pi(self):
        expected = np.pi / (2 * np.pi)  # 0.5
        assert _normalize_angle(np.pi) == pytest.approx(expected, abs=1e-5)

    def test_negative_angle(self):
        result = _normalize_angle(-np.pi / 2)
        assert 0.0 <= result <= 1.0


# ─── Circuit Encoder Tests ──────────────────────────────────────────────

class TestCircuitToGraph:
    """Test circuit_to_pyg_graph() function."""

    def test_simple_two_gate_circuit(self):
        """Simple H → CX circuit should produce graph with 2 nodes."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        graph = circuit_to_pyg_graph(qc)

        assert graph.x.shape == (2, FEATURE_DIM)  # 2 gates, 20 features each
        assert graph.edge_index.shape[0] == 2  # [2, num_edges]
        assert graph.num_nodes == 2

    def test_three_qubit_circuit(self):
        """3-qubit circuit with H, CX, RZ should encode correctly."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(np.pi / 4, 2)

        graph = circuit_to_pyg_graph(qc)

        assert graph.x.shape == (3, FEATURE_DIM)  # 3 gates
        assert graph.num_nodes == 3
        # H(q0) → CX(q0,q1) should create an edge (shared qubit 0)
        assert graph.edge_index.shape[1] >= 1

    def test_edge_connectivity(self):
        """Gates sharing qubits should be connected by edges."""
        qc = QuantumCircuit(2)
        qc.h(0)       # gate 0
        qc.h(1)       # gate 1
        qc.cx(0, 1)   # gate 2 — depends on both gate 0 and gate 1

        graph = circuit_to_pyg_graph(qc)

        assert graph.num_nodes == 3
        # Gate 2 (CX) should have incoming edges from gate 0 and gate 1
        assert graph.edge_index.shape[1] >= 2

    def test_independent_gates_no_edges(self):
        """Gates on completely independent qubits should have no edges."""
        qc = QuantumCircuit(3)
        qc.h(0)   # only touches qubit 0
        qc.x(1)   # only touches qubit 1
        qc.z(2)   # only touches qubit 2

        graph = circuit_to_pyg_graph(qc)

        assert graph.num_nodes == 3
        assert graph.edge_index.shape[1] == 0  # no dependencies

    def test_empty_circuit_raises(self):
        """Empty circuit should raise ValueError."""
        qc = QuantumCircuit(2)  # no gates added
        with pytest.raises(ValueError, match="no gates"):
            circuit_to_pyg_graph(qc)

    def test_feature_values_match(self):
        """Node features should match gate_to_feature_vector output."""
        qc = QuantumCircuit(2)
        qc.h(0)

        graph = circuit_to_pyg_graph(qc)

        expected_vec = gate_to_feature_vector("h", [0], num_qubits=2)
        assert torch.allclose(graph.x[0], expected_vec)

    def test_circuit_with_parameters(self):
        """Circuit with parametric gates should encode parameters."""
        qc = QuantumCircuit(1)
        qc.rz(np.pi / 3, 0)
        qc.ry(np.pi / 6, 0)

        graph = circuit_to_pyg_graph(qc)

        assert graph.num_nodes == 2
        assert graph.x.shape == (2, FEATURE_DIM)
        # Both should be parametric
        assert graph.x[0, 19] == 1.0  # is_parametric flag
        assert graph.x[1, 19] == 1.0

    def test_larger_circuit(self):
        """Larger circuit should handle many gates correctly."""
        qc = QuantumCircuit(5)
        for i in range(5):
            qc.h(i)
        for i in range(4):
            qc.cx(i, i + 1)
        qc.rz(np.pi / 4, 0)

        graph = circuit_to_pyg_graph(qc)

        expected_nodes = 5 + 4 + 1  # 5 H + 4 CX + 1 RZ
        assert graph.num_nodes == expected_nodes
        assert graph.x.shape == (expected_nodes, FEATURE_DIM)
        assert not torch.isnan(graph.x).any()
        assert graph.edge_index.shape[0] == 2
