"""
quantumopt.graph.encoder — Circuit-to-graph encoder.

Converts Qiskit QuantumCircuits into PyTorch Geometric Data objects
by first converting to a DAG representation, then extracting nodes
(gates) and edges (dependencies).

Provides two encoders:
    - circuit_to_pyg_graph()      → 20-dim features (legacy GCN model)
    - circuit_to_pyg_graph_21d()  → 21-dim features (trained GAT model)
"""

import torch
import numpy as np
from torch_geometric.data import Data
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from quantumopt.graph.features import gate_to_feature_vector


# ═══════════════════════════════════════════════════════════════════
# 21-dim gate encoder (matches Kaggle training notebook)
# ═══════════════════════════════════════════════════════════════════
GATE_TYPES_21D = [
    'h', 'cx', 'rz', 'ry', 'rx', 'x', 'y', 'z', 'swap', 'ccx',
    't', 'tdg', 's', 'sdg', 'measure', 'barrier', 'reset', 'u', 'p', 'unknown',
]
_GATE_TO_IDX_21D = {g: i for i, g in enumerate(GATE_TYPES_21D)}
_NUM_GATE_TYPES_21D = len(GATE_TYPES_21D)  # 20
FEATURE_DIM_21D = _NUM_GATE_TYPES_21D + 1  # 21


def _gate_to_feature_21d(gate_name: str, qubit_indices: list, num_qubits: int) -> torch.Tensor:
    """Encode a gate as a 21-dim feature vector (matches trained GAT model).

    Layout:
        [0:20]  — One-hot gate type
        [20]    — Normalized qubit index (first qubit / total qubits)
    """
    one_hot = torch.zeros(_NUM_GATE_TYPES_21D)
    idx = _GATE_TO_IDX_21D.get(gate_name.lower(), _GATE_TO_IDX_21D['unknown'])
    one_hot[idx] = 1.0

    if qubit_indices and num_qubits > 0:
        norm_qubit = qubit_indices[0] / num_qubits
    else:
        norm_qubit = 0.0

    return torch.cat([one_hot, torch.tensor([norm_qubit])])


# ═══════════════════════════════════════════════════════════════════
# 21-dim graph encoder (for trained QuantumGAT)
# ═══════════════════════════════════════════════════════════════════
def circuit_to_pyg_graph_21d(qc: QuantumCircuit) -> Data:
    """Convert a Qiskit QuantumCircuit to a PyG graph with 21-dim features.

    This matches the feature encoding used during Kaggle GNN training.
    Each gate becomes a node with a 21-dim feature vector; each DAG
    dependency becomes a directed edge.

    Args:
        qc: A Qiskit QuantumCircuit.

    Returns:
        torch_geometric.data.Data with x [num_nodes, 21] and edge_index.

    Raises:
        ValueError: If the circuit has no gates.
    """
    dag = circuit_to_dag(qc)
    num_qubits = qc.num_qubits

    op_nodes = list(dag.op_nodes())
    if len(op_nodes) == 0:
        raise ValueError("Circuit has no gates — cannot encode an empty circuit.")

    node_to_idx = {node: idx for idx, node in enumerate(op_nodes)}

    # Build 21-dim node features
    features = []
    for node in op_nodes:
        qubit_indices = [qc.find_bit(q).index for q in node.qargs]
        feat = _gate_to_feature_21d(node.op.name, qubit_indices, num_qubits)
        features.append(feat)

    x = torch.stack(features, dim=0)  # [num_nodes, 21]

    # Build edges
    src_list, dst_list = [], []
    for edge in dag.edges():
        src_node, tgt_node, _ = edge
        if isinstance(src_node, DAGOpNode) and isinstance(tgt_node, DAGOpNode):
            if src_node in node_to_idx and tgt_node in node_to_idx:
                src_list.append(node_to_idx[src_node])
                dst_list.append(node_to_idx[tgt_node])

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        # Self-loops fallback for edgeless circuits
        n = len(op_nodes)
        edge_index = torch.tensor([list(range(n)), list(range(n))], dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


# ═══════════════════════════════════════════════════════════════════
# 20-dim graph encoder (legacy, for QuantumCircuitGNN)
# ═══════════════════════════════════════════════════════════════════
def circuit_to_pyg_graph(qc: QuantumCircuit) -> Data:
    """Convert a Qiskit QuantumCircuit to a PyTorch Geometric Data object.

    Process:
        1. Convert circuit to DAG using Qiskit's circuit_to_dag()
        2. Extract operation nodes (gates) as graph nodes with 20-dim feature vectors
        3. Extract edges (gate dependencies via shared qubits) as edge_index tensor
        4. Include barrier/measure nodes — skip only input/output DAG nodes

    Args:
        qc: A Qiskit QuantumCircuit (any number of qubits/gates).

    Returns:
        torch_geometric.data.Data with:
            - x: Node feature matrix [num_nodes, 20]
            - edge_index: Edge connectivity [2, num_edges]
            - num_nodes: Number of gate nodes

    Raises:
        ValueError: If the circuit has no gates.
    """
    dag = circuit_to_dag(qc)
    num_qubits = qc.num_qubits

    op_nodes = list(dag.op_nodes())
    if len(op_nodes) == 0:
        raise ValueError("Circuit has no gates — cannot encode an empty circuit.")

    node_to_idx = {node: idx for idx, node in enumerate(op_nodes)}

    # --- Build node features ---
    node_features = []
    for node in op_nodes:
        gate_name = node.op.name
        qubit_indices = [qc.find_bit(q).index for q in node.qargs]
        params = [float(p) for p in node.op.params] if node.op.params else []

        feature_vec = gate_to_feature_vector(
            gate_name=gate_name,
            qubits=qubit_indices,
            params=params,
            num_qubits=num_qubits,
        )
        node_features.append(feature_vec)

    x = torch.stack(node_features, dim=0)  # [num_nodes, 20]

    # --- Build edge index from DAG edges ---
    source_indices = []
    target_indices = []

    for edge in dag.edges():
        src_node, tgt_node, _edge_data = edge
        if isinstance(src_node, DAGOpNode) and isinstance(tgt_node, DAGOpNode):
            if src_node in node_to_idx and tgt_node in node_to_idx:
                source_indices.append(node_to_idx[src_node])
                target_indices.append(node_to_idx[tgt_node])

    if len(source_indices) > 0:
        edge_index = torch.tensor(
            [source_indices, target_indices], dtype=torch.long
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data
