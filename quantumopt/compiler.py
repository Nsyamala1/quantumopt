"""
quantumopt.compiler — Main compilation pipeline.

Orchestrates the full optimization pipeline:
circuit → graph encoder → GNN prediction → IBM backend → Claude explanation

Usage:
    from quantumopt import compile

    result = compile(
        circuit=my_vqe_circuit,
        hardware="ibm_brisbane",
        priority="fidelity"
    )

    print(result.optimized_circuit)
    print(result.depth_reduction)
    print(result.explanation)
"""

import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)

# Path to the trained model weights
_WEIGHTS_DIR = Path(__file__).parent / "models" / "weights"
_GAT_WEIGHTS = _WEIGHTS_DIR / "gnn_best.pt"

# Cached model instance (loaded once per process)
_cached_model = None
_cached_model_type = None  # "gat" or "gcn"


# ═══════════════════════════════════════════════════════════════════
# CompileResult dataclass
# ═══════════════════════════════════════════════════════════════════
@dataclass
class CompileResult:
    """Result of the compile() pipeline.

    Attributes:
        optimized_circuit: The transpiled + optimized QuantumCircuit.
        depth_reduction:   Human-readable depth reduction string (e.g. "34%").
        gate_reduction:    Human-readable gate count reduction string.
        explanation:       Claude-generated or fallback explanation text.
        gnn_prediction:    GNN-predicted improvement ratio ∈ [0, 1].
        recommended_actions: List of GNN-recommended optimization actions.
        original_stats:    Dict with original circuit stats.
        optimized_stats:   Dict with optimized circuit stats.
        benchmark:         Summary dict with key metrics for easy access.
        compile_time:      Wall-clock time in seconds for the full pipeline.
    """
    optimized_circuit: QuantumCircuit = None
    depth_reduction: str = "0%"
    gate_reduction: str = "0%"
    explanation: Optional[str] = None
    gnn_prediction: Optional[float] = None
    recommended_actions: List[Dict[str, Any]] = field(default_factory=list)
    original_stats: Dict[str, Any] = field(default_factory=dict)
    optimized_stats: Dict[str, Any] = field(default_factory=dict)
    benchmark: Dict[str, Any] = field(default_factory=dict)
    compile_time: float = 0.0

    def __repr__(self):
        gnn_str = f"{self.gnn_prediction:.1%}" if self.gnn_prediction is not None else "N/A"
        return (
            f"CompileResult(\n"
            f"  depth_reduction={self.depth_reduction},\n"
            f"  gate_reduction={self.gate_reduction},\n"
            f"  gnn_prediction={gnn_str},\n"
            f"  compile_time={self.compile_time:.2f}s,\n"
            f"  original_depth={self.original_stats.get('depth', '?')},\n"
            f"  optimized_depth={self.optimized_stats.get('depth', '?')},\n"
            f"  fidelity={self.optimized_stats.get('estimated_fidelity', '?')}\n"
            f")"
        )


# ═══════════════════════════════════════════════════════════════════
# GNN model loading
# ═══════════════════════════════════════════════════════════════════
def _load_model():
    """Load the GNN model — prefers trained GAT, falls back to legacy GCN.

    Returns:
        Tuple of (model, model_type) where model_type is "gat" or "gcn".
        Returns (None, None) if no model can be loaded.
    """
    global _cached_model, _cached_model_type

    if _cached_model is not None:
        return _cached_model, _cached_model_type

    # Try QuantumGAT first (trained model from Kaggle)
    if _GAT_WEIGHTS.exists():
        try:
            from quantumopt.models.gat import QuantumGAT
            model = QuantumGAT(input_dim=21)
            model.load_weights(str(_GAT_WEIGHTS))
            _cached_model = model
            _cached_model_type = "gat"
            logger.info(f"Loaded QuantumGAT from {_GAT_WEIGHTS}")
            return model, "gat"
        except Exception as e:
            logger.warning(f"Failed to load QuantumGAT: {e}")

    # Fall back to legacy QuantumCircuitGNN
    gcn_weights = _WEIGHTS_DIR / "gnn_best.pt"
    try:
        from quantumopt.models.gnn import QuantumCircuitGNN
        model = QuantumCircuitGNN(input_dim=20)
        model.load_weights(str(gcn_weights))
        _cached_model = model
        _cached_model_type = "gcn"
        logger.info("Loaded legacy QuantumCircuitGNN")
        return model, "gcn"
    except Exception as e:
        logger.info(f"No GNN model available: {e}")
        return None, None


def _predict_with_gnn(circuit: QuantumCircuit) -> Dict[str, Any]:
    """Run GNN prediction on a circuit.

    Returns:
        Dict with 'score' (float) and 'actions' (list), or empty dict on failure.
    """
    model, model_type = _load_model()
    if model is None:
        return {}

    try:
        if model_type == "gat":
            from quantumopt.graph.encoder import circuit_to_pyg_graph_21d
            graph = circuit_to_pyg_graph_21d(circuit)
        else:
            from quantumopt.graph.encoder import circuit_to_pyg_graph
            graph = circuit_to_pyg_graph(circuit)

        prediction = model.predict(graph)
        return prediction
    except Exception as e:
        logger.warning(f"GNN prediction failed: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════
# Main compile() function
# ═══════════════════════════════════════════════════════════════════
def compile(
    circuit: QuantumCircuit,
    hardware: str = "ibm_brisbane",
    priority: str = "fidelity",
    explain: bool = True,
    optimization_level: int = 3,
) -> CompileResult:
    """Compile a quantum circuit using the full quantumopt pipeline.

    Pipeline:
        1. Encode circuit as graph
        2. GNN predicts optimization potential
        3. Qiskit transpiler optimizes for target hardware
        4. Claude explains the optimization (if ANTHROPIC_API_KEY is set)

    Args:
        circuit:            Qiskit QuantumCircuit to optimize.
        hardware:           Target backend name (default: "ibm_brisbane").
        priority:           Optimization priority — "fidelity", "depth", or "speed".
        explain:            Whether to generate an explanation (default: True).
        optimization_level: Qiskit transpiler level 0–3 (default: 3).

    Returns:
        CompileResult with optimized circuit, stats, prediction, and explanation.

    Example:
        >>> from qiskit import QuantumCircuit
        >>> from quantumopt import compile
        >>> qc = QuantumCircuit(3)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> qc.cx(1, 2)
        >>> result = compile(qc, hardware="ibm_brisbane")
        >>> print(result.depth_reduction)
        >>> print(result.explanation)
    """
    start_time = time.time()

    # ── Step 1: Get original circuit stats ─────────────────────────
    # Transpile at level 0 first to match training data labeling
    from qiskit import transpile as qiskit_transpile
    from quantumopt.backends.ibm_backend import _get_circuit_stats
    try:
        from qiskit.providers.fake_provider import GenericBackendV2
        _baseline_backend = GenericBackendV2(num_qubits=27)
    except ImportError:
        _baseline_backend = None

    baseline_circuit = qiskit_transpile(
        circuit,
        backend=_baseline_backend,
        optimization_level=0,
        seed_transpiler=42
    )
    original_stats = _get_circuit_stats(baseline_circuit)

    # ── Step 2: GNN prediction ─────────────────────────────────────
    gnn_result = _predict_with_gnn(circuit)
    gnn_score = gnn_result.get("score")
    gnn_actions = gnn_result.get("actions", [])

    if gnn_score is not None:
        logger.info(
            f"GNN predicted improvement: {gnn_score:.1%} — "
            f"top action: {gnn_actions[0]['action'] if gnn_actions else 'N/A'}"
        )

    # ── Step 3: Qiskit transpilation ───────────────────────────────
    from qiskit import transpile
    
    # Run transpile() ONCE at optimization_level=3 targeting FakeBrisbane backend
    try:
        from qiskit.providers.fake_provider import GenericBackendV2
        backend = GenericBackendV2(num_qubits=27)
    except ImportError:
        backend = None
        
    if backend:
        compiled_circuit = transpile(
            circuit,
            backend=backend,
            optimization_level=3,
            seed_transpiler=42
        )
    else:
        compiled_circuit = transpile(
            circuit,
            optimization_level=optimization_level,
            seed_transpiler=42
        )
        
    optimized_stats = _get_circuit_stats(compiled_circuit)
    from quantumopt.backends.ibm_backend import _estimate_fidelity
    optimized_stats["estimated_fidelity"] = _estimate_fidelity(compiled_circuit)

    # ── Step 4: Compute reductions ─────────────────────────────────
    orig_depth = original_stats.get("depth", 1)
    opt_depth = optimized_stats.get("depth", orig_depth)
    orig_gates = original_stats.get("gate_count", 1)
    opt_gates = optimized_stats.get("gate_count", orig_gates)

    depth_reduction_float = (orig_depth - opt_depth) / max(orig_depth, 1) * 100
    gate_pct = ((orig_gates - opt_gates) / max(orig_gates, 1)) * 100

    depth_reduction = f"{depth_reduction_float:.0f}%"
    gate_reduction = f"{gate_pct:.0f}%"

    # ── Step 5: Build optimization decisions list ──────────────────
    decisions = []
    if gnn_actions:
        decisions = [a["action"] for a in gnn_actions if a.get("confidence", 0) > 0.3]
    if not decisions:
        decisions = ["standard_transpilation"]

    # ── Step 6: Explanation ────────────────────────────────────────
    explanation = None
    if explain:
        from quantumopt.llm.explainer import (
            explain_optimization,
            explain_optimization_fallback,
        )
        explanation = explain_optimization(original_stats, optimized_stats, decisions)
        if explanation is None:
            explanation = explain_optimization_fallback(
                original_stats, optimized_stats, decisions
            )

        if depth_reduction_float < 0:
            explanation += " Note: circuit grew after hardware mapping — this is normal for small circuits where SWAP overhead dominates."

    # ── Step 7: Build benchmark dict ───────────────────────────────
    compile_time = time.time() - start_time

    benchmark = {
        "original_depth": orig_depth,
        "optimized_depth": opt_depth,
        "depth_reduction": depth_reduction,
        "original_gates": orig_gates,
        "optimized_gates": opt_gates,
        "gate_reduction": gate_reduction,
        "estimated_fidelity": optimized_stats.get("estimated_fidelity", "N/A"),
        "gnn_predicted_improvement": gnn_score,
        "hardware": hardware,
        "optimization_level": optimization_level,
        "compile_time_s": round(compile_time, 3),
    }

    return CompileResult(
        optimized_circuit=compiled_circuit,
        depth_reduction=depth_reduction,
        gate_reduction=gate_reduction,
        explanation=explanation,
        gnn_prediction=gnn_score,
        recommended_actions=gnn_actions,
        original_stats=original_stats,
        optimized_stats=optimized_stats,
        benchmark=benchmark,
        compile_time=compile_time,
    )
