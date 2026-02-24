"""
quantumopt.benchmarks.compare — Benchmark comparison module.

Compares QuantumOpt's circuit optimization against Qiskit's built-in
transpiler, measuring depth reduction, gate count reduction, and runtime.
Generates JSON and text reports suitable for research presentations.
"""

import json
import time
import logging
import statistics
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

from quantumopt.compiler import compile as quantumopt_compile
from quantumopt.backends.ibm_backend import _get_circuit_stats
from quantumopt.data.pipeline import CIRCUIT_GENERATORS

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkReport:
    """Benchmark comparison report.

    Attributes:
        num_circuits: Number of circuits tested.
        quantumopt_results: Per-circuit results for QuantumOpt.
        qiskit_results: Per-circuit results for Qiskit transpiler.
        summary: Aggregate comparison statistics.
    """
    num_circuits: int = 0
    quantumopt_results: List[Dict[str, Any]] = field(default_factory=list)
    qiskit_results: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: str = "benchmark_report.json"):
        """Save report as JSON."""
        report = {
            "num_circuits": self.num_circuits,
            "summary": self.summary,
            "quantumopt_results": self.quantumopt_results,
            "qiskit_results": self.qiskit_results,
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"JSON report saved to {path}")

    def to_text(self, path: str = "benchmark_report.txt"):
        """Save report as human-readable text."""
        lines = []
        lines.append("=" * 70)
        lines.append("QuantumOpt Benchmark Report")
        lines.append("=" * 70)
        lines.append(f"Circuits tested: {self.num_circuits}")
        lines.append("")

        s = self.summary
        lines.append("--- QuantumOpt Results ---")
        lines.append(f"  Avg depth reduction:      {s.get('quantumopt_avg_depth_reduction', 'N/A')}")
        lines.append(f"  Median depth reduction:   {s.get('quantumopt_median_depth_reduction', 'N/A')}")
        lines.append(f"  Avg gate reduction:        {s.get('quantumopt_avg_gate_reduction', 'N/A')}")
        lines.append(f"  Avg compile time:          {s.get('quantumopt_avg_time', 'N/A')}s")
        lines.append("")

        lines.append("--- Qiskit Transpiler (Level 3) Results ---")
        lines.append(f"  Avg depth reduction:      {s.get('qiskit_avg_depth_reduction', 'N/A')}")
        lines.append(f"  Median depth reduction:   {s.get('qiskit_median_depth_reduction', 'N/A')}")
        lines.append(f"  Avg gate reduction:        {s.get('qiskit_avg_gate_reduction', 'N/A')}")
        lines.append(f"  Avg compile time:          {s.get('qiskit_avg_time', 'N/A')}s")
        lines.append("")

        lines.append("--- Comparison ---")
        lines.append(f"  QuantumOpt advantage (depth): {s.get('advantage_depth', 'N/A')}")
        lines.append(f"  QuantumOpt advantage (gates): {s.get('advantage_gates', 'N/A')}")
        lines.append("")
        lines.append("=" * 70)

        text = "\n".join(lines)
        with open(path, "w") as f:
            f.write(text)
        logger.info(f"Text report saved to {path}")
        return text


def _generate_test_circuits(
    num_circuits: int = 50,
    circuit_types: Optional[List[str]] = None,
) -> List[QuantumCircuit]:
    """Generate test circuits for benchmarking.

    Args:
        num_circuits: Number of circuits to generate.
        circuit_types: Types to include (default: all).

    Returns:
        List of QuantumCircuit objects.
    """
    import random
    import numpy as np

    random.seed(123)
    np.random.seed(123)

    if circuit_types is None:
        circuit_types = list(CIRCUIT_GENERATORS.keys())

    circuits = []
    per_type = num_circuits // len(circuit_types)

    for ctype in circuit_types:
        gen = CIRCUIT_GENERATORS[ctype]
        for _ in range(per_type):
            n_qubits = random.randint(3, 8)
            if ctype == "grover":
                n_qubits = min(n_qubits, 5)
            try:
                qc = gen(n_qubits)
                circuits.append(qc)
            except Exception:
                continue

    return circuits


def _compute_reduction(before: int, after: int) -> float:
    """Compute percentage reduction as a float."""
    if before <= 0:
        return 0.0
    return max(0.0, ((before - after) / before) * 100)


def run_benchmark(
    test_circuits: Optional[List[QuantumCircuit]] = None,
    hardware: str = "ibm_brisbane",
    num_circuits: int = 50,
) -> BenchmarkReport:
    """Run benchmark comparing QuantumOpt vs Qiskit transpiler.

    For each test circuit:
        1. Run QuantumOpt compilation pipeline
        2. Run Qiskit transpiler at optimization_level=3

    Measures: depth reduction %, gate count reduction %, runtime.
    Generates: summary statistics (mean, median, std for all metrics).

    Args:
        test_circuits: List of circuits to test (generates if None).
        hardware: Target hardware backend (default: "ibm_brisbane").
        num_circuits: Number of circuits to generate if test_circuits is None.

    Returns:
        BenchmarkReport with comparison statistics.
    """
    if test_circuits is None:
        logger.info(f"Generating {num_circuits} test circuits...")
        test_circuits = _generate_test_circuits(num_circuits)

    report = BenchmarkReport(num_circuits=len(test_circuits))

    qopt_depth_reductions = []
    qopt_gate_reductions = []
    qopt_times = []

    qiskit_depth_reductions = []
    qiskit_gate_reductions = []
    qiskit_times = []

    for i, circuit in enumerate(test_circuits):
        original_stats = _get_circuit_stats(circuit)
        circuit_name = getattr(circuit, "name", f"circuit_{i}")

        # --- QuantumOpt Compilation ---
        try:
            t_start = time.time()
            result = quantumopt_compile(
                circuit, hardware=hardware, priority="fidelity", explain=False
            )
            t_qopt = time.time() - t_start

            qopt_stats = result.optimized_stats
            qopt_depth_red = _compute_reduction(original_stats["depth"], qopt_stats["depth"])
            qopt_gate_red = _compute_reduction(original_stats["gate_count"], qopt_stats["gate_count"])

            qopt_depth_reductions.append(qopt_depth_red)
            qopt_gate_reductions.append(qopt_gate_red)
            qopt_times.append(t_qopt)

            report.quantumopt_results.append({
                "circuit": circuit_name,
                "original_depth": original_stats["depth"],
                "optimized_depth": qopt_stats["depth"],
                "depth_reduction": round(qopt_depth_red, 2),
                "gate_reduction": round(qopt_gate_red, 2),
                "time": round(t_qopt, 3),
            })
        except Exception as e:
            logger.warning(f"QuantumOpt failed on {circuit_name}: {e}")

        # --- Qiskit Transpiler ---
        try:
            t_start = time.time()
            qiskit_compiled = transpile(
                circuit,
                basis_gates=["cx", "id", "rz", "sx", "x"],
                optimization_level=3,
            )
            t_qiskit = time.time() - t_start

            qiskit_stats = _get_circuit_stats(qiskit_compiled)
            qiskit_depth_red = _compute_reduction(original_stats["depth"], qiskit_stats["depth"])
            qiskit_gate_red = _compute_reduction(original_stats["gate_count"], qiskit_stats["gate_count"])

            qiskit_depth_reductions.append(qiskit_depth_red)
            qiskit_gate_reductions.append(qiskit_gate_red)
            qiskit_times.append(t_qiskit)

            report.qiskit_results.append({
                "circuit": circuit_name,
                "original_depth": original_stats["depth"],
                "optimized_depth": qiskit_stats["depth"],
                "depth_reduction": round(qiskit_depth_red, 2),
                "gate_reduction": round(qiskit_gate_red, 2),
                "time": round(t_qiskit, 3),
            })
        except Exception as e:
            logger.warning(f"Qiskit transpiler failed on {circuit_name}: {e}")

    # --- Compute Summary Statistics ---
    def _safe_stats(values):
        if not values:
            return {"mean": 0, "median": 0, "std": 0}
        return {
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "std": round(statistics.stdev(values), 2) if len(values) > 1 else 0,
        }

    qopt_depth_stats = _safe_stats(qopt_depth_reductions)
    qopt_gate_stats = _safe_stats(qopt_gate_reductions)
    qiskit_depth_stats = _safe_stats(qiskit_depth_reductions)
    qiskit_gate_stats = _safe_stats(qiskit_gate_reductions)

    report.summary = {
        "quantumopt_avg_depth_reduction": f"{qopt_depth_stats['mean']:.1f}%",
        "quantumopt_median_depth_reduction": f"{qopt_depth_stats['median']:.1f}%",
        "quantumopt_avg_gate_reduction": f"{qopt_gate_stats['mean']:.1f}%",
        "quantumopt_avg_time": f"{statistics.mean(qopt_times):.3f}" if qopt_times else "N/A",

        "qiskit_avg_depth_reduction": f"{qiskit_depth_stats['mean']:.1f}%",
        "qiskit_median_depth_reduction": f"{qiskit_depth_stats['median']:.1f}%",
        "qiskit_avg_gate_reduction": f"{qiskit_gate_stats['mean']:.1f}%",
        "qiskit_avg_time": f"{statistics.mean(qiskit_times):.3f}" if qiskit_times else "N/A",

        "advantage_depth": f"{qopt_depth_stats['mean'] - qiskit_depth_stats['mean']:.1f}% points",
        "advantage_gates": f"{qopt_gate_stats['mean'] - qiskit_gate_stats['mean']:.1f}% points",
    }

    logger.info(f"Benchmark complete: {len(test_circuits)} circuits")
    logger.info(f"QuantumOpt avg depth reduction: {report.summary['quantumopt_avg_depth_reduction']}")
    logger.info(f"Qiskit avg depth reduction: {report.summary['qiskit_avg_depth_reduction']}")

    return report
