"""Tests for quantumopt.models.gnn — GNN model."""

import os
import pytest
import torch
import numpy as np
import tempfile
from qiskit import QuantumCircuit
from torch_geometric.data import Data, Batch

from quantumopt.models.gnn import QuantumCircuitGNN, ACTION_LABELS
from quantumopt.graph.encoder import circuit_to_pyg_graph


class TestGNNArchitecture:
    """Test QuantumCircuitGNN model structure."""

    def test_model_creation(self):
        """Model should instantiate without errors."""
        model = QuantumCircuitGNN()
        assert isinstance(model, torch.nn.Module)

    def test_parameter_count(self):
        """Model should have a reasonable number of parameters."""
        model = QuantumCircuitGNN()
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
        assert num_params < 100_000  # should be lightweight

    def test_custom_dimensions(self):
        """Model should accept custom input/hidden dimensions."""
        model = QuantumCircuitGNN(input_dim=32, hidden_dim=128, num_actions=6)
        assert model.conv1.in_channels == 32
        assert model.conv1.out_channels == 128


class TestGNNForward:
    """Test forward pass with various inputs."""

    def test_single_graph_forward(self):
        """Forward pass with a single graph should produce correct output shapes."""
        model = QuantumCircuitGNN()
        model.eval()

        # Create a simple 3-qubit circuit graph
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(np.pi / 4, 2)
        data = circuit_to_pyg_graph(qc)

        with torch.no_grad():
            score, actions = model(data)

        assert score.shape == (1, 1)  # [batch_size=1, 1]
        assert actions.shape == (1, 4)  # [batch_size=1, 4 actions]
        assert 0.0 <= score.item() <= 1.0  # sigmoid output

    def test_batched_forward(self):
        """Forward pass with batched graphs should handle multiple circuits."""
        model = QuantumCircuitGNN()
        model.eval()

        # Create two circuit graphs
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit(3)
        qc2.h(0)
        qc2.h(1)
        qc2.cx(0, 2)

        data1 = circuit_to_pyg_graph(qc1)
        data2 = circuit_to_pyg_graph(qc2)

        batch = Batch.from_data_list([data1, data2])

        with torch.no_grad():
            score, actions = model(batch)

        assert score.shape == (2, 1)  # 2 graphs in batch
        assert actions.shape == (2, 4)

    def test_gradient_flow(self):
        """Gradients should flow through the model."""
        model = QuantumCircuitGNN()
        model.train()

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        data = circuit_to_pyg_graph(qc)

        score, actions = model(data)
        loss = score.sum() + actions.sum()
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestGNNPredict:
    """Test predict() convenience method."""

    def test_predict_returns_dict(self):
        """predict() should return a dict with score and actions."""
        model = QuantumCircuitGNN()

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        data = circuit_to_pyg_graph(qc)

        result = model.predict(data)

        assert "score" in result
        assert "actions" in result
        assert isinstance(result["score"], float)
        assert len(result["actions"]) == 4
        assert 0.0 <= result["score"] <= 1.0

    def test_predict_actions_sorted(self):
        """Actions should be sorted by confidence (descending)."""
        model = QuantumCircuitGNN()

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        data = circuit_to_pyg_graph(qc)

        result = model.predict(data)

        confidences = [a["confidence"] for a in result["actions"]]
        assert confidences == sorted(confidences, reverse=True)

    def test_predict_action_labels(self):
        """All action labels should be valid."""
        model = QuantumCircuitGNN()

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        data = circuit_to_pyg_graph(qc)

        result = model.predict(data)
        action_names = {a["action"] for a in result["actions"]}
        assert action_names == set(ACTION_LABELS)


class TestGNNWeights:
    """Test weight save/load functionality."""

    def test_save_and_load(self):
        """Model should save and load weights correctly."""
        model = QuantumCircuitGNN()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_weights.pt")
            model.save_weights(path)
            assert os.path.exists(path)

            # Load into a new model
            model2 = QuantumCircuitGNN()
            model2.load_weights(path)

            # Compare parameters
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                assert torch.allclose(p1, p2), f"Mismatch in {n1}"

    def test_load_missing_file(self):
        """Loading from non-existent path should raise FileNotFoundError."""
        model = QuantumCircuitGNN()
        with pytest.raises(FileNotFoundError):
            model.load_weights("/nonexistent/path/weights.pt")

    def test_save_creates_directory(self):
        """save_weights should create parent directories."""
        model = QuantumCircuitGNN()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "dir", "weights.pt")
            model.save_weights(path)
            assert os.path.exists(path)
