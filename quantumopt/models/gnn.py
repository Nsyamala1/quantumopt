"""
quantumopt.models.gnn — Graph Neural Network for circuit optimization.

Architecture:
    3 × GCNConv layers (20→64→64→64) with ReLU + dropout
    → global_mean_pool for graph-level readout
    → Score head: Linear(64→32→1) for optimization improvement ratio
    → Action head: Linear(64→32→4) for recommended optimization actions

Actions correspond to:
    0: cancel_redundant_gates
    1: merge_rotations
    2: commute_and_cancel
    3: reorder_for_routing
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


# Action labels for interpretability
ACTION_LABELS = [
    "cancel_redundant_gates",
    "merge_rotations",
    "commute_and_cancel",
    "reorder_for_routing",
]


class QuantumCircuitGNN(nn.Module):
    """GNN model for predicting quantum circuit optimization potential.

    Dual-output architecture:
        1. **Score** (regression): Predicted improvement ratio [0, 1]
           — how much the circuit can be optimized.
        2. **Actions** (4-class classification): Which optimization
           strategies are most likely to be effective.

    Args:
        input_dim: Dimension of node feature vectors (default: 20).
        hidden_dim: Hidden layer dimension (default: 64).
        dropout: Dropout rate (default: 0.2).
        num_actions: Number of optimization action classes (default: 4).
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        num_actions: int = 4,
    ):
        super().__init__()

        self.dropout = dropout

        # --- GCN Layers ---
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Batch normalization for each GCN layer
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # --- Score Head (regression) ---
        self.score_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)  # 64 → 32
        self.score_fc2 = nn.Linear(hidden_dim // 2, 1)            # 32 → 1

        # --- Action Head (classification) ---
        self.action_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)  # 64 → 32
        self.action_fc2 = nn.Linear(hidden_dim // 2, num_actions)  # 32 → 4

    def forward(self, data: Data):
        """Forward pass through the GNN.

        Args:
            data: PyG Data object with:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - batch: Batch assignment vector (optional, for batched graphs)

        Returns:
            Tuple of (score, actions):
                score: Tensor [batch_size, 1] — predicted improvement ratio
                actions: Tensor [batch_size, 4] — action class logits
        """
        x, edge_index = data.x, data.edge_index

        # Get batch vector (defaults to single graph if not provided)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # --- GCN Layers with ReLU + BatchNorm + Dropout ---
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # --- Global Mean Pooling → graph-level representation ---
        graph_embedding = global_mean_pool(x, batch)  # [batch_size, hidden_dim]

        # --- Score Head ---
        score = F.relu(self.score_fc1(graph_embedding))
        score = torch.sigmoid(self.score_fc2(score))  # [0, 1] range

        # --- Action Head ---
        actions = F.relu(self.action_fc1(graph_embedding))
        actions = self.action_fc2(actions)  # raw logits for 4 classes

        return score, actions

    def predict(self, data: Data):
        """Run inference on a single graph (convenience method).

        Returns:
            Dict with 'score' (float) and 'actions' (list of action labels
            sorted by confidence).
        """
        self.eval()
        with torch.no_grad():
            score, action_logits = self.forward(data)

        score_val = score.item()
        action_probs = F.softmax(action_logits, dim=-1).squeeze().tolist()

        # Sort actions by probability (descending)
        ranked_actions = sorted(
            zip(ACTION_LABELS, action_probs),
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "score": score_val,
            "actions": [{"action": name, "confidence": prob} for name, prob in ranked_actions],
        }

    def save_weights(self, path: str = "quantumopt/models/weights/gnn_best.pt"):
        """Save model weights to disk.

        Args:
            path: File path for the saved weights.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str = "quantumopt/models/weights/gnn_best.pt"):
        """Load model weights from disk.

        Args:
            path: File path to load weights from.

        Raises:
            FileNotFoundError: If the weights file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"GNN weights not found at: {path}")
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.eval()
