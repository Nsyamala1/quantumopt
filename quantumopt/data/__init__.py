"""
quantumopt.data — Dataset generation and processing module.

Generates training datasets from quantum circuit benchmarks (MQTBench)
and processes them for GNN training.
"""

from quantumopt.data.pipeline import generate_dataset, load_dataset

__all__ = ["generate_dataset", "load_dataset"]
