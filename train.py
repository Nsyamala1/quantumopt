"""
train.py — GNN Training Script for QuantumOpt.

Trains the QuantumCircuitGNN model to predict circuit improvement ratios
from PyG graph representations of quantum circuits.

Usage:
    # Generate dataset first, then train:
    python train.py --generate --num-circuits 500 --epochs 50

    # Train on existing dataset:
    python train.py --dataset data/circuits/training_dataset.json --epochs 100

Designed to run on Kaggle free GPU (T4, 14GB VRAM).
"""

import os
import sys
import json
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from qiskit import QuantumCircuit
from qiskit.qasm2 import loads as qasm2_loads

from quantumopt.graph.encoder import circuit_to_pyg_graph
from quantumopt.models.gnn import QuantumCircuitGNN
from quantumopt.data.pipeline import generate_dataset, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _circuit_from_entry(entry: dict) -> QuantumCircuit:
    """Reconstruct a QuantumCircuit from a dataset entry.

    Tries QASM first, falls back to circuit reconstruction.
    """
    qasm_str = entry.get("circuit_qasm", "")
    if qasm_str and qasm_str.startswith("OPENQASM"):
        try:
            return qasm2_loads(qasm_str)
        except Exception:
            pass

    # Fallback: create a dummy circuit with matching stats
    stats = entry.get("original_stats", {})
    n = stats.get("num_qubits", 3)
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(min(n - 1, stats.get("depth", 3))):
        qc.cx(i % n, (i + 1) % n)
    return qc


def prepare_dataset(dataset_entries: list) -> list:
    """Convert dataset entries to PyG Data objects with labels.

    Args:
        dataset_entries: List of dicts from pipeline.generate_dataset().

    Returns:
        List of PyG Data objects with y (improvement_ratio) label.
    """
    graphs = []
    skipped = 0

    for entry in dataset_entries:
        try:
            qc = _circuit_from_entry(entry)
            data = circuit_to_pyg_graph(qc)
            data.y = torch.tensor([entry["improvement_ratio"]], dtype=torch.float32)
            graphs.append(data)
        except Exception as e:
            skipped += 1
            continue

    if skipped > 0:
        logger.warning(f"Skipped {skipped}/{len(dataset_entries)} circuits during encoding")

    return graphs


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: QuantumCircuitGNN,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 10,
    save_path: str = "quantumopt/models/weights/gnn_best.pt",
    device: str = "cpu",
):
    """Train the GNN model.

    Args:
        train_loader: Training data loader.
        val_loader: Validation data loader.
        model: QuantumCircuitGNN instance.
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: L2 regularization.
        patience: Early stopping patience.
        save_path: Path to save best model weights.
        device: Device to train on ("cpu" or "cuda").

    Returns:
        Dict with training history (train_loss, val_loss per epoch).
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    logger.info(f"Training on {device} — {epochs} epochs, lr={lr}")
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_count = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            score, _ = model(batch)
            loss = criterion(score.squeeze(), batch.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            train_count += batch.num_graphs

        train_loss /= max(train_count, 1)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                score, _ = model(batch)
                loss = criterion(score.squeeze(), batch.y.squeeze())
                mae = torch.abs(score.squeeze() - batch.y.squeeze()).sum()
                val_loss += loss.item() * batch.num_graphs
                val_mae += mae.item()
                val_count += batch.num_graphs

        val_loss /= max(val_count, 1)
        val_mae /= max(val_count, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MAE: {val_mae:.6f} | "
            f"LR: {current_lr:.6f}"
        )

        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_weights(save_path)
            logger.info(f"  → Saved best model (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    return history


def evaluate_model(
    test_loader: DataLoader,
    model: QuantumCircuitGNN,
    device: str = "cpu",
) -> dict:
    """Evaluate the model on the test set.

    Returns:
        Dict with test_loss, test_mae statistics.
    """
    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    total_mae = 0.0
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            score, _ = model(batch)
            loss = criterion(score.squeeze(), batch.y.squeeze())
            mae = torch.abs(score.squeeze() - batch.y.squeeze()).sum()
            total_loss += loss.item() * batch.num_graphs
            total_mae += mae.item()
            count += batch.num_graphs

    return {
        "test_loss": total_loss / max(count, 1),
        "test_mae": total_mae / max(count, 1),
        "test_samples": count,
    }


def main():
    parser = argparse.ArgumentParser(description="Train QuantumOpt GNN model")
    parser.add_argument("--dataset", type=str, default="data/circuits/training_dataset.json",
                        help="Path to training dataset JSON")
    parser.add_argument("--generate", action="store_true",
                        help="Generate dataset before training")
    parser.add_argument("--num-circuits", type=int, default=500,
                        help="Number of circuits to generate (if --generate)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--save-path", type=str,
                        default="quantumopt/models/weights/gnn_best.pt",
                        help="Path to save best model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # --- Step 1: Generate or Load Dataset ---
    if args.generate:
        logger.info(f"Generating dataset with {args.num_circuits} circuits...")
        args.dataset = generate_dataset(num_circuits=args.num_circuits)

    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found: {args.dataset}")
        logger.info("Run with --generate to create a dataset first.")
        sys.exit(1)

    dataset_entries = load_dataset(args.dataset)
    logger.info(f"Loaded {len(dataset_entries)} circuits from {args.dataset}")

    # --- Step 2: Convert to PyG Graphs ---
    logger.info("Converting circuits to graphs...")
    start = time.time()
    graphs = prepare_dataset(dataset_entries)
    logger.info(f"Encoded {len(graphs)} graphs in {time.time()-start:.1f}s")

    if len(graphs) < 10:
        logger.error("Not enough valid graphs for training. Need at least 10.")
        sys.exit(1)

    # --- Step 3: Split 70/15/15 ---
    np.random.seed(42)
    indices = np.random.permutation(len(graphs))
    n_train = int(0.7 * len(graphs))
    n_val = int(0.15 * len(graphs))

    train_graphs = [graphs[i] for i in indices[:n_train]]
    val_graphs = [graphs[i] for i in indices[n_train:n_train + n_val]]
    test_graphs = [graphs[i] for i in indices[n_train + n_val:]]

    logger.info(f"Split: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    # --- Step 4: Train ---
    model = QuantumCircuitGNN()
    history = train_model(
        train_loader, val_loader, model,
        epochs=args.epochs, lr=args.lr,
        patience=args.patience, save_path=args.save_path,
        device=device,
    )

    # --- Step 5: Evaluate on Test Set ---
    model.load_weights(args.save_path)
    test_results = evaluate_model(test_loader, model, device=device)

    logger.info("=" * 60)
    logger.info(f"Test Loss: {test_results['test_loss']:.6f}")
    logger.info(f"Test MAE:  {test_results['test_mae']:.6f}")
    logger.info(f"Test Samples: {test_results['test_samples']}")
    logger.info("=" * 60)
    logger.info(f"Best model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
