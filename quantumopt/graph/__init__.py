"""
quantumopt.graph — Circuit-to-graph encoding module.

Converts Qiskit QuantumCircuits into PyTorch Geometric graph representations
for GNN-based optimization prediction.
"""

from quantumopt.graph.encoder import circuit_to_pyg_graph
from quantumopt.graph.features import gate_to_feature_vector

__all__ = ["circuit_to_pyg_graph", "gate_to_feature_vector"]
