#!/usr/bin/env python3
"""
generate_benchmark_datasets.py — Generate and benchmark QAOA, VQE, and IQP circuit datasets.

Part 1: Generate 500 circuits for each of QAOA, VQE, and IQP algorithm types.
Part 2: Benchmark each dataset (100 samples) with Qiskit transpile / quantumopt.compile().
Part 3: Compute summary statistics and write a human-readable report.

Output files:
  dataset_v3_qaoa.json, dataset_v3_vqe.json, dataset_v3_iqp.json
  benchmark_results_v3.json
  benchmark_analysis_v3.json
  benchmark_report_v3.txt

Usage:
    python generate_benchmark_datasets.py
"""

import json
import random
import time
import traceback
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_PER_TYPE = 500
BENCHMARK_SAMPLE = 100
PROGRESS_EVERY_GEN = 50       # Print progress every N circuits during generation
PROGRESS_EVERY_BENCH = 25     # Print progress every N circuits during benchmarking

QAOA_QUBITS  = [4, 6, 8, 10, 12, 16, 20]
QAOA_PLAYERS = [1, 2, 3, 5]
VQE_QUBITS   = [4, 6, 8, 10, 12, 16]
VQE_REPS     = [1, 2, 3, 4]
IQP_QUBITS   = [4, 6, 8, 10, 12, 16]
IQP_DEPTHS   = [1, 2, 3, 4]

CIRCUITS_PER_CONFIG = 10  # ~10 circuits per (n_qubits, depth/layers) configuration

BENCH_BASIS_GATES = ["cx", "u"]
BENCH_OPT_LEVEL   = 3

# ── Check if quantumopt.compile is available ─────────────────────────────────
try:
    from quantumopt import compile as qopt_compile
    _QUANTUMOPT_AVAILABLE = True
    print("✅ quantumopt.compile() is available — will use it for benchmarking.")
except ImportError:
    _QUANTUMOPT_AVAILABLE = False
    print("ℹ️  quantumopt not installed — using raw Qiskit transpile for benchmarking.")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _circuit_stats(qc: QuantumCircuit) -> dict:
    """Extract gate statistics from a QuantumCircuit."""
    ops = dict(qc.count_ops())
    n_cx = ops.get("cx", 0) + ops.get("cnot", 0) + ops.get("ecr", 0) + ops.get("cz", 0)
    return {
        "n_qubits":   qc.num_qubits,
        "depth":      qc.depth(),
        "gate_count": sum(ops.values()),
        "gate_types": ops,
        "n_cx":       n_cx,
    }


def _qasm_export(qc: QuantumCircuit) -> str:
    """Export circuit as OpenQASM 2.0 string, with fallbacks."""
    try:
        return qc.qasm()
    except Exception:
        pass
    try:
        from qiskit.qasm2 import dumps as qasm2_dumps
        return qasm2_dumps(qc)
    except Exception:
        pass
    try:
        from qiskit.qasm3 import dumps as qasm3_dumps
        return qasm3_dumps(qc)
    except Exception:
        pass
    raise RuntimeError("Could not export to QASM")


def _save_json(data, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  💾 Saved {len(data) if isinstance(data, list) else ''} → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — Circuit Generators
# ═══════════════════════════════════════════════════════════════════════════════

# ─── QAOA ────────────────────────────────────────────────────────────────────

def _build_qaoa_circuit(n_qubits: int, p_layers: int) -> QuantumCircuit:
    """Build a QAOA circuit for a random MaxCut problem."""
    # Random graph with 50% edge probability
    edges = [
        (i, j)
        for i in range(n_qubits)
        for j in range(i + 1, n_qubits)
        if random.random() < 0.5
    ]
    # Ensure at least one edge
    if not edges:
        edges = [(0, 1)]

    beta  = [random.uniform(0, 2 * np.pi) for _ in range(p_layers)]
    gamma = [random.uniform(0, 2 * np.pi) for _ in range(p_layers)]

    qc = QuantumCircuit(n_qubits)

    # Initial superposition
    for q in range(n_qubits):
        qc.h(q)

    for layer in range(p_layers):
        # Cost unitary — ZZ on edges
        for (u, v) in edges:
            qc.cx(u, v)
            qc.rz(2 * gamma[layer], v)
            qc.cx(u, v)
        # Mixer unitary — Rx per qubit
        for q in range(n_qubits):
            qc.rx(2 * beta[layer], q)

    return qc


def generate_qaoa_dataset(target: int = TARGET_PER_TYPE) -> list:
    """Generate QAOA circuits."""
    print("\n" + "═" * 60)
    print("PART 1a — Generating QAOA circuits")
    print("═" * 60)

    records = []
    circuit_id = 0
    configs = [(nq, p) for nq in QAOA_QUBITS for p in QAOA_PLAYERS]

    while len(records) < target:
        for (n_qubits, p_layers) in configs:
            if len(records) >= target:
                break
            for _ in range(CIRCUITS_PER_CONFIG):
                if len(records) >= target:
                    break
                try:
                    qc = _build_qaoa_circuit(n_qubits, p_layers)
                    qasm = _qasm_export(qc)
                    stats = _circuit_stats(qc)
                    circuit_id += 1
                    records.append({
                        "id":           f"qaoa_{circuit_id:04d}",
                        "type":         "QAOA",
                        "config":       {"n_qubits": n_qubits, "p_layers": p_layers},
                        "qasm":         qasm,
                        "before_stats": stats,
                        "generated_at": _now_iso(),
                    })
                    count = len(records)
                    if count % PROGRESS_EVERY_GEN == 0:
                        print(f"  QAOA: {count}/{target} circuits generated | {n_qubits}q p={p_layers}")
                except Exception as e:
                    print(f"  ⚠️  QAOA {n_qubits}q p={p_layers} failed: {e}")

    print(f"  ✅ QAOA done — {len(records)} circuits")
    return records


# ─── VQE ─────────────────────────────────────────────────────────────────────

def _build_vqe_circuit(n_qubits: int, reps: int) -> QuantumCircuit:
    """Build an EfficientSU2 VQE ansatz with random bound parameters."""
    ansatz = EfficientSU2(
        num_qubits=n_qubits,
        entanglement="linear",
        reps=reps,
    )
    # Bind random parameters in [0, 2π]
    param_values = {p: random.uniform(0, 2 * np.pi) for p in ansatz.parameters}
    qc = ansatz.assign_parameters(param_values)
    return qc.decompose()  # expand to gate-level (ry, rz, cx) for valid QASM export


def generate_vqe_dataset(target: int = TARGET_PER_TYPE) -> list:
    """Generate VQE circuits."""
    print("\n" + "═" * 60)
    print("PART 1b — Generating VQE circuits")
    print("═" * 60)

    records = []
    circuit_id = 0
    configs = [(nq, r) for nq in VQE_QUBITS for r in VQE_REPS]

    while len(records) < target:
        for (n_qubits, reps) in configs:
            if len(records) >= target:
                break
            for _ in range(CIRCUITS_PER_CONFIG):
                if len(records) >= target:
                    break
                try:
                    qc = _build_vqe_circuit(n_qubits, reps)
                    qasm = _qasm_export(qc)
                    stats = _circuit_stats(qc)
                    circuit_id += 1
                    records.append({
                        "id":           f"vqe_{circuit_id:04d}",
                        "type":         "VQE",
                        "config":       {"n_qubits": n_qubits, "reps": reps},
                        "qasm":         qasm,
                        "before_stats": stats,
                        "generated_at": _now_iso(),
                    })
                    count = len(records)
                    if count % PROGRESS_EVERY_GEN == 0:
                        print(f"  VQE: {count}/{target} circuits generated | {n_qubits}q reps={reps}")
                except Exception as e:
                    print(f"  ⚠️  VQE {n_qubits}q reps={reps} failed: {e}")

    print(f"  ✅ VQE done — {len(records)} circuits")
    return records


# ─── IQP ─────────────────────────────────────────────────────────────────────

def _build_iqp_circuit(n_qubits: int, depth: int) -> QuantumCircuit:
    """Build an IQP circuit: H → RZ phase layer(s) → CZ pairs → H."""
    qc = QuantumCircuit(n_qubits)

    # Initial H layer
    for q in range(n_qubits):
        qc.h(q)

    for _ in range(depth):
        # Random diagonal (RZ) phase gates
        for q in range(n_qubits):
            angle = random.uniform(0, 2 * np.pi)
            qc.rz(angle, q)

        # Random CZ pairs — pick n_qubits//2 random pairs without repeating qubits
        qubits = list(range(n_qubits))
        random.shuffle(qubits)
        for k in range(0, len(qubits) - 1, 2):
            qc.cz(qubits[k], qubits[k + 1])

    # Final H layer
    for q in range(n_qubits):
        qc.h(q)

    return qc


def generate_iqp_dataset(target: int = TARGET_PER_TYPE) -> list:
    """Generate IQP circuits."""
    print("\n" + "═" * 60)
    print("PART 1c — Generating IQP circuits")
    print("═" * 60)

    records = []
    circuit_id = 0
    configs = [(nq, d) for nq in IQP_QUBITS for d in IQP_DEPTHS]

    while len(records) < target:
        for (n_qubits, depth) in configs:
            if len(records) >= target:
                break
            for _ in range(CIRCUITS_PER_CONFIG):
                if len(records) >= target:
                    break
                try:
                    qc = _build_iqp_circuit(n_qubits, depth)
                    qasm = _qasm_export(qc)
                    stats = _circuit_stats(qc)
                    circuit_id += 1
                    records.append({
                        "id":           f"iqp_{circuit_id:04d}",
                        "type":         "IQP",
                        "config":       {"n_qubits": n_qubits, "depth": depth},
                        "qasm":         qasm,
                        "before_stats": stats,
                        "generated_at": _now_iso(),
                    })
                    count = len(records)
                    if count % PROGRESS_EVERY_GEN == 0:
                        print(f"  IQP: {count}/{target} circuits generated | {n_qubits}q depth={depth}")
                except Exception as e:
                    print(f"  ⚠️  IQP {n_qubits}q depth={depth} failed: {e}")

    print(f"  ✅ IQP done — {len(records)} circuits")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — Benchmarking
# ═══════════════════════════════════════════════════════════════════════════════

# Native gate set used for the level-0 baseline — broad enough to cover all
# abstract gates (rz, ry, rx, cz, h) so both before and after live in the
# same native gate space and the comparison is apples-to-apples.
_NATIVE_BASIS = ["cx", "u", "sx", "rz", "x"]


def _benchmark_one_qiskit(record: dict) -> dict | None:
    """Benchmark a single circuit record using raw Qiskit transpile.

    Compares optimization_level=0 baseline (native gate space) against
    optimization_level=3, so the before/after gate counts are in the same
    gate set and the reduction percentage is meaningful.
    """
    try:
        qc = QuantumCircuit.from_qasm_str(record["qasm"])
    except Exception as e:
        print(f"  ⚠️  [{record['id']}] QASM reload failed: {e}")
        return None

    # ── Baseline: level-0 in native gate space ───────────────────────────
    try:
        baseline = transpile(
            qc,
            basis_gates=_NATIVE_BASIS,
            optimization_level=0,
            seed_transpiler=42,
        )
    except Exception as e:
        print(f"  ⚠️  [{record['id']}] Baseline transpile failed: {e}")
        return None

    before_gates = sum(dict(baseline.count_ops()).values())
    before_depth = baseline.depth()

    # ── Optimized: level-3 in native gate space ──────────────────────────
    try:
        t0 = time.time()
        optimized = transpile(
            qc,
            basis_gates=BENCH_BASIS_GATES,
            optimization_level=BENCH_OPT_LEVEL,
            seed_transpiler=42,
        )
        compile_time_s = round(time.time() - t0, 4)
    except Exception as e:
        print(f"  ⚠️  [{record['id']}] Transpile failed: {e}")
        return None

    after_gates = sum(dict(optimized.count_ops()).values())
    after_depth = optimized.depth()

    gate_reduction  = (before_gates - after_gates) / max(before_gates, 1) * 100
    depth_reduction = (before_depth - after_depth) / max(before_depth, 1) * 100

    return {
        "id":              record["id"],
        "type":            record["type"],
        "config":          record["config"],
        "before": {
            "gate_count": before_gates,
            "depth":      before_depth,
        },
        "after": {
            "gate_count": after_gates,
            "depth":      after_depth,
        },
        "gate_reduction_pct":  round(gate_reduction, 2),
        "depth_reduction_pct": round(depth_reduction, 2),
        "compile_time_s":      compile_time_s,
        "method":              "qiskit_transpile",
    }


def _benchmark_one_quantumopt(record: dict) -> dict | None:
    """Benchmark a single circuit record using quantumopt.compile().

    Uses result.original_stats and result.optimized_stats directly —
    quantumopt already computes a fair internal before/after comparison
    (level-0 vs level-3) within the same native gate space (rz, sx, cx
    via GenericBackendV2). This avoids any mixed-basis comparison.
    """
    try:
        qc = QuantumCircuit.from_qasm_str(record["qasm"])
    except Exception as e:
        print(f"  ⚠️  [{record['id']}] QASM reload failed: {e}")
        return None

    # ── Run quantumopt — it handles its own before/after internally ──────
    try:
        t0 = time.time()
        result = qopt_compile(qc, hardware="ibm_brisbane", explain=False)
        compile_time_s = round(time.time() - t0, 4)
    except Exception as e:
        print(f"  ⚠️  [{record['id']}] quantumopt.compile() failed: {e}")
        # Fall back to raw Qiskit transpile
        return _benchmark_one_qiskit(record)

    # Both stats are in the same internal gate space (rz/sx/cx) —
    # original_stats = level-0 transpile, optimized_stats = level-3 transpile
    before_gates = result.original_stats["gate_count"]
    before_depth = result.original_stats["depth"]
    after_gates  = result.optimized_stats["gate_count"]
    after_depth  = result.optimized_stats["depth"]

    gate_reduction  = (before_gates - after_gates) / max(before_gates, 1) * 100
    depth_reduction = (before_depth - after_depth) / max(before_depth, 1) * 100

    return {
        "id":              record["id"],
        "type":            record["type"],
        "config":          record["config"],
        "before": {
            "gate_count": before_gates,
            "depth":      before_depth,
        },
        "after": {
            "gate_count": after_gates,
            "depth":      after_depth,
        },
        "gate_reduction_pct":  round(gate_reduction, 2),
        "depth_reduction_pct": round(depth_reduction, 2),
        "compile_time_s":      compile_time_s,
        "method":              "quantumopt_compile",
        "gnn_prediction":      result.gnn_prediction,
    }


def benchmark_dataset(dataset: list, label: str, sample_n: int = BENCHMARK_SAMPLE) -> list:
    """Sample `sample_n` circuits from `dataset` and benchmark each one."""
    print(f"\n{'─' * 60}")
    print(f"PART 2 — Benchmarking {label} ({sample_n} circuits sampled)")
    print(f"{'─' * 60}")

    sample = random.sample(dataset, min(sample_n, len(dataset)))
    results = []

    for idx, record in enumerate(sample, 1):
        if _QUANTUMOPT_AVAILABLE:
            res = _benchmark_one_quantumopt(record)
        else:
            res = _benchmark_one_qiskit(record)

        if res is not None:
            results.append(res)

        if idx % PROGRESS_EVERY_BENCH == 0:
            print(f"  {label}: {idx}/{len(sample)} benchmarked | {len(results)} successful so far")

    print(f"  ✅ {label} benchmarking done — {len(results)}/{len(sample)} successful")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3 — Analysis and Report
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_stats(results: list) -> dict:
    """Compute summary statistics for a list of benchmark results."""
    if not results:
        return {}

    gate_reds  = [r["gate_reduction_pct"] for r in results]
    depth_reds = [r["depth_reduction_pct"] for r in results]
    n = len(results)

    improved  = sum(1 for x in gate_reds if x > 0)
    worsened  = sum(1 for x in gate_reds if x < 0)
    unchanged = n - improved - worsened

    # Breakdown by qubit count
    by_qubits: dict[int, list] = defaultdict(list)
    for r in results:
        nq = r["config"].get("n_qubits", 0)
        by_qubits[nq].append(r["gate_reduction_pct"])

    qubit_breakdown = {
        nq: {
            "n_circuits":        len(vals),
            "mean_gate_reduction": round(mean(vals), 2),
        }
        for nq, vals in sorted(by_qubits.items())
    }

    return {
        "n_circuits":              n,
        "pct_improved":            round(improved  / n * 100, 1),
        "pct_worsened":            round(worsened  / n * 100, 1),
        "pct_unchanged":           round(unchanged / n * 100, 1),
        "mean_gate_reduction":     round(mean(gate_reds),  2),
        "median_gate_reduction":   round(median(gate_reds), 2),
        "best_gate_reduction":     round(max(gate_reds),   2),
        "worst_gate_reduction":    round(min(gate_reds),   2),
        "mean_depth_reduction":    round(mean(depth_reds),  2),
        "median_depth_reduction":  round(median(depth_reds), 2),
        "qubit_breakdown":         qubit_breakdown,
    }


def write_report(analysis: dict, path: Path):
    """Write a human-readable benchmark report."""
    lines = []
    sep = "═" * 70

    lines.append(sep)
    lines.append("QUANTUMOPT BENCHMARK REPORT v3")
    lines.append(f"Generated: {_now_iso()}")
    lines.append(f"Method: {'quantumopt.compile()' if _QUANTUMOPT_AVAILABLE else 'Qiskit transpile (optimization_level=3)'}")
    lines.append(sep)
    lines.append("")

    for ctype, stats in analysis.items():
        if not stats:
            continue
        lines.append(f"{'─' * 70}")
        lines.append(f"  {ctype} CIRCUITS")
        lines.append(f"{'─' * 70}")
        lines.append(f"  Total circuits benchmarked : {stats['n_circuits']}")
        lines.append("")
        lines.append("  IMPROVEMENT SUMMARY")
        lines.append(f"    Circuits improved  (gate_reduction > 0) : {stats['pct_improved']:5.1f}%")
        lines.append(f"    Circuits worsened  (gate_reduction < 0) : {stats['pct_worsened']:5.1f}%")
        lines.append(f"    Circuits unchanged (gate_reduction = 0) : {stats['pct_unchanged']:5.1f}%")
        lines.append("")
        lines.append("  GATE REDUCTION STATISTICS")
        lines.append(f"    Mean   gate reduction : {stats['mean_gate_reduction']:+6.2f}%")
        lines.append(f"    Median gate reduction : {stats['median_gate_reduction']:+6.2f}%")
        lines.append(f"    Best   gate reduction : {stats['best_gate_reduction']:+6.2f}%")
        lines.append(f"    Worst  gate reduction : {stats['worst_gate_reduction']:+6.2f}%")
        lines.append("")
        lines.append("  DEPTH REDUCTION STATISTICS")
        lines.append(f"    Mean   depth reduction : {stats['mean_depth_reduction']:+6.2f}%")
        lines.append(f"    Median depth reduction : {stats['median_depth_reduction']:+6.2f}%")
        lines.append("")
        lines.append("  BREAKDOWN BY QUBIT COUNT")
        lines.append(f"    {'Qubits':>8}  {'Circuits':>9}  {'Mean Gate Reduction':>22}")
        lines.append(f"    {'──────':>8}  {'────────':>9}  {'───────────────────':>22}")
        for nq, bk in stats["qubit_breakdown"].items():
            lines.append(
                f"    {nq:>8}  {bk['n_circuits']:>9}  {bk['mean_gate_reduction']:>21.2f}%"
            )
        lines.append("")

    lines.append(sep)
    lines.append("END OF REPORT")
    lines.append(sep)

    path.write_text("\n".join(lines))
    print(f"\n  📄 Report written → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  QuantumOpt Benchmark Dataset Generator v3                     ║")
    print("║  QAOA • VQE • IQP — 500 circuits each, 100 benchmarked each   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Started: {_now_iso()}\n")

    start = time.time()

    # ── Part 1: Generate datasets ──────────────────────────────────────────
    qaoa_data = generate_qaoa_dataset()
    _save_json(qaoa_data, Path("dataset_v3_qaoa.json"))

    vqe_data = generate_vqe_dataset()
    _save_json(vqe_data, Path("dataset_v3_vqe.json"))

    iqp_data = generate_iqp_dataset()
    _save_json(iqp_data, Path("dataset_v3_iqp.json"))

    # ── Part 2: Benchmark ──────────────────────────────────────────────────
    qaoa_bench = benchmark_dataset(qaoa_data, "QAOA")
    vqe_bench  = benchmark_dataset(vqe_data,  "VQE")
    iqp_bench  = benchmark_dataset(iqp_data,  "IQP")

    all_results = {
        "QAOA": qaoa_bench,
        "VQE":  vqe_bench,
        "IQP":  iqp_bench,
    }
    _save_json(
        {
            "generated_at": _now_iso(),
            "method": "quantumopt_compile" if _QUANTUMOPT_AVAILABLE else "qiskit_transpile",
            "results": all_results,
        },
        Path("benchmark_results_v3.json"),
    )

    # ── Part 3: Analysis ───────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("PART 3 — Computing analysis")
    print("═" * 60)

    analysis = {
        ctype: _compute_stats(bench)
        for ctype, bench in all_results.items()
    }

    _save_json(
        {
            "generated_at": _now_iso(),
            "method": "quantumopt_compile" if _QUANTUMOPT_AVAILABLE else "qiskit_transpile",
            "analysis": analysis,
        },
        Path("benchmark_analysis_v3.json"),
    )

    write_report(analysis, Path("benchmark_report_v3.txt"))

    elapsed = time.time() - start
    print(f"\n✅ All done in {elapsed / 60:.1f} minutes.")
    print("   Output files:")
    print("     dataset_v3_qaoa.json  — QAOA dataset")
    print("     dataset_v3_vqe.json   — VQE dataset")
    print("     dataset_v3_iqp.json   — IQP dataset")
    print("     benchmark_results_v3.json  — raw benchmark results")
    print("     benchmark_analysis_v3.json — computed statistics")
    print("     benchmark_report_v3.txt    — human-readable report")


if __name__ == "__main__":
    main()
