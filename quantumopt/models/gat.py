"""
quantumopt.models.gat — Graph Attention Network for trained circuit optimizer.

This is the model architecture matching gnn_best.pt trained on Kaggle.
Uses 21-dim node features (20 gate one-hot + normalized qubit index)
and GAT attention layers instead of GCN.

Architecture:
    GATConv(21, 64, heads=4)  → 256 dims + ELU
    GATConv(256, 64, heads=4) → 256 dims + ELU
    GATConv(256, 32, heads=1) → 32 dims + ELU
    global_mean_pool → graph-level embedding
    Linear(32→64) + ReLU + Dropout(0.3)
    Linear(64→32) + ReLU
    Linear(32→1) + sigmoid → improvement_ratio ∈ [0, 1]
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data


# Action labels inherited from the old model for pipeline compatibility
ACTION_LABELS = [
    "cancel_redundant_gates",
    "merge_rotations",
    "commute_and_cancel",
    "reorder_for_routing",
]


class QuantumGAT(nn.Module):
    """Graph Attention Network for predicting quantum circuit optimization ratios.

    21-dim input features, 3 multi-head GAT layers, graph pooling, FC head.
    This architecture matches the model trained via train_gnn.ipynb on Kaggle.

    Args:
        input_dim: Dimension of node feature vectors (default: 21).
        hidden_dim: Hidden layer dimension (default: 64).
        output_dim: Output dimension (default: 1).
    """

    def __init__(self, input_dim: int = 21, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()

        # GAT layers with multi-head attention
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)       # → 256
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)  # → 256
        self.conv3 = GATConv(hidden_dim * 4, 32, heads=1, concat=True)          # → 32

        # Activation and regularization
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.3)

        # Fully connected head
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

        self.relu = nn.ReLU()

    def forward(self, data: Data):
        """Forward pass through the GAT.

        Args:
            data: PyG Data object with x, edge_index, and optionally batch.

        Returns:
            Tensor [batch_size] — predicted improvement ratios in [0, 1].
        """
        x, edge_index = data.x, data.edge_index
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )

        # GAT message passing
        x = self.elu(self.conv1(x, edge_index))
        x = self.elu(self.conv2(x, edge_index))
        x = self.elu(self.conv3(x, edge_index))

        # Graph-level readout
        x = global_mean_pool(x, batch)

        # FC head
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        # Sigmoid — improvement ratio is between 0 and 1
        x = torch.sigmoid(x)
        return x.squeeze(-1)

    def predict(self, data: Data) -> dict:
        """Run inference on a single graph.

        Returns:
            Dict with 'score' (float) and 'actions' (list of recommended
            optimization actions sorted by heuristic confidence).
        """
        self.eval()
        with torch.no_grad():
            score = self.forward(data)

        score_val = float(score.item()) if score.dim() == 0 else float(score[0].item())

        # Heuristic action ranking based on predicted improvement
        if score_val > 0.6:
            actions = [
                {"action": "merge_rotations", "confidence": 0.8},
                {"action": "cancel_redundant_gates", "confidence": 0.7},
                {"action": "commute_and_cancel", "confidence": 0.5},
                {"action": "reorder_for_routing", "confidence": 0.3},
            ]
        elif score_val > 0.3:
            actions = [
                {"action": "cancel_redundant_gates", "confidence": 0.6},
                {"action": "commute_and_cancel", "confidence": 0.5},
                {"action": "merge_rotations", "confidence": 0.4},
                {"action": "reorder_for_routing", "confidence": 0.3},
            ]
        else:
            actions = [
                {"action": "reorder_for_routing", "confidence": 0.4},
                {"action": "cancel_redundant_gates", "confidence": 0.3},
                {"action": "commute_and_cancel", "confidence": 0.2},
                {"action": "merge_rotations", "confidence": 0.1},
            ]

        return {"score": score_val, "actions": actions}

    def save_weights(self, path: str = "quantumopt/models/weights/gnn_best.pt"):
        """Save model weights to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str = "quantumopt/models/weights/gnn_best.pt"):
        """Load model weights from disk.

        Raises:
            FileNotFoundError: If the weights file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"GAT weights not found at: {path}")

        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict)
        self.eval()
