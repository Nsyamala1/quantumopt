#!/usr/bin/env python3
"""
generate_dataset.py — Build a labeled quantum circuit dataset for GNN-based compiler training.

Sources:
  1. MQTBench (pip install mqt.bench)
  2. Qiskit Circuit Library (TwoLocal, EfficientSU2, RealAmplitudes, QAOAAnsatz, QFT, GroverOperator, GHZ)
  3. Synthetic random circuits (qiskit.circuit.random.random_circuit)

Labeling:
  - FakeBrisbane() backend (no IBM account needed)
  - Transpile at optimization_level=0 (baseline) and optimization_level=3 (optimized)
  - Record depths, gate counts, CX counts, improvement ratio

Output: dataset.json (+ dataset_checkpoint.json every 1000 circuits)
Failures: failed.log

Usage:
    python generate_dataset.py
"""

import json
import sys
import time
import random
import signal
import logging
import traceback
from collections import defaultdict
from pathlib import Path

# ── Qiskit imports ──────────────────────────────────────────────────────────
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import (
    TwoLocal,
    EfficientSU2,
    RealAmplitudes,
    QFT as QiskitQFT,
    GroverOperator,
)

# FakeBrisbane – no IBM account needed
try:
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
except ImportError:
    from qiskit.providers.fake_provider import FakeBrisbaneV2 as FakeBrisbane

# ── Constants ───────────────────────────────────────────────────────────────
TARGET_TOTAL = 15_000
RANDOM_COUNT = 10_000
CHECKPOINT_EVERY = 1_000
PROGRESS_EVERY = 100
TIMEOUT_SECONDS = 30  # Max time per circuit labeling

OUTPUT_FILE = Path("dataset.json")
CHECKPOINT_FILE = Path("dataset_checkpoint.json")
FAILED_LOG = Path("failed.log")

QUBIT_RANGE = range(3, 21)       # 3 to 20 for Qiskit library & random
MQTBENCH_QUBIT_RANGE = range(3, 11)  # 3 to 10 for MQTBench (large circuits too slow)

# Mapping: label name -> mqt.bench benchmark name
# NOTE: grover removed — exponentially slow, covered by Qiskit library at small sizes
MQTBENCH_ALGORITHMS = {
    "vqe_real_amp": "vqe_real_amp",
    "vqe_su2": "vqe_su2",
    "vqe_two_local": "vqe_two_local",
    "qaoa": "qaoa",
    "qft": "qft",
    "ghz": "ghz",
    "bernstein_vazirani": "bv",
    "deutsch_jozsa": "dj",
    "amplitude_estimation": "ae",
    "phase_estimation_exact": "qpeexact",
    "phase_estimation_inexact": "qpeinexact",
    "graphstate": "graphstate",
    "wstate": "wstate",
    "qnn": "qnn",
    "qwalk": "qwalk",
}

# ── Logging setup ───────────────────────────────────────────────────────────
fail_logger = logging.getLogger("failures")
fail_logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(FAILED_LOG, mode="w")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
fail_logger.addHandler(_fh)


# ═══════════════════════════════════════════════════════════════════════════
# Backend singleton
# ═══════════════════════════════════════════════════════════════════════════
_backend = None


def get_backend():
    """Lazily create and cache FakeBrisbane backend."""
    global _backend
    if _backend is None:
        print("⏳ Initializing FakeBrisbane backend (one-time)...")
        _backend = FakeBrisbane()
        print("✅ Backend ready.\n")
    return _backend


# ═══════════════════════════════════════════════════════════════════════════
# Timeout helper
# ═══════════════════════════════════════════════════════════════════════════
class CircuitTimeoutError(Exception):
    """Raised when a circuit labeling exceeds the time limit."""
    pass


def _timeout_handler(signum, frame):
    raise CircuitTimeoutError(f"Circuit labeling timed out after {TIMEOUT_SECONDS}s")


# ═══════════════════════════════════════════════════════════════════════════
# Labeling: transpile & extract stats
# ═══════════════════════════════════════════════════════════════════════════
def _count_cx(circuit: QuantumCircuit) -> int:
    """Count CX (CNOT) gates including any ECR/CZ mapped equivalents."""
    ops = circuit.count_ops()
    return ops.get("cx", 0) + ops.get("cnot", 0)


def _count_two_qubit(circuit: QuantumCircuit) -> int:
    """Count all two-qubit gates (cx, ecr, cz, etc.)."""
    ops = circuit.count_ops()
    two_q_names = {"cx", "cnot", "ecr", "cz", "swap", "iswap", "rzz", "rxx", "ryy", "cp"}
    return sum(ops.get(name, 0) for name in two_q_names)


def label_circuit(circuit: QuantumCircuit, algorithm: str, num_qubits: int) -> dict:
    """Transpile circuit at level 0 and level 3, return labeled record.

    Raises on any failure (including timeout) so the caller can log it.
    """
    # Set a per-circuit timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    backend = get_backend()

    # Baseline transpilation (level 0) — closest to original structure
    baseline = transpile(
        circuit, backend=backend, optimization_level=0, seed_transpiler=42
    )
    # Optimized transpilation (level 3)
    optimized = transpile(
        circuit, backend=backend, optimization_level=3, seed_transpiler=42
    )

    original_depth = baseline.depth()
    optimized_depth = optimized.depth()
    original_gates = sum(baseline.count_ops().values())
    optimized_gates = sum(optimized.count_ops().values())

    improvement = (
        (original_depth - optimized_depth) / original_depth
        if original_depth > 0
        else 0.0
    )

    cx_original = _count_two_qubit(baseline)
    cx_optimized = _count_two_qubit(optimized)

    # Export QASM (try 2.0 first, fall back to 3.0 exporter)
    try:
        qasm_str = circuit.qasm()
    except Exception:
        try:
            from qiskit.qasm2 import dumps as qasm2_dumps
            qasm_str = qasm2_dumps(circuit)
        except Exception:
            from qiskit.qasm3 import dumps as qasm3_dumps
            qasm_str = qasm3_dumps(circuit)

    # Cancel the alarm — we finished in time
    signal.alarm(0)

    return {
        "algorithm": algorithm,
        "num_qubits": num_qubits,
        "original_qasm": qasm_str,
        "original_depth": original_depth,
        "original_gates": int(original_gates),
        "optimized_depth": optimized_depth,
        "optimized_gates": int(optimized_gates),
        "improvement_ratio": round(improvement, 4),
        "cx_count_original": cx_original,
        "cx_count_optimized": cx_optimized,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Source 1: MQTBench circuits
# ═══════════════════════════════════════════════════════════════════════════
def generate_mqtbench_circuits():
    """Yield (circuit, algorithm_name, num_qubits) tuples from MQTBench."""
    try:
        from mqt.bench import get_benchmark, BenchmarkLevel
    except ImportError:
        print("⚠️  mqt.bench not installed — skipping MQTBench circuits.")
        print("   Install with: pip install mqt.bench")
        return

    for label, bench_name in MQTBENCH_ALGORITHMS.items():
        for nq in MQTBENCH_QUBIT_RANGE:
            try:
                qc = get_benchmark(
                    benchmark=bench_name,
                    level=BenchmarkLevel.INDEP,
                    circuit_size=nq,
                )
                if isinstance(qc, QuantumCircuit):
                    yield qc, label, nq
            except Exception as e:
                fail_logger.debug(f"MQTBench {label} ({bench_name}) {nq}q: {e}")
                continue


# ═══════════════════════════════════════════════════════════════════════════
# Source 2: Qiskit Circuit Library
# ═══════════════════════════════════════════════════════════════════════════
def generate_qiskit_library_circuits():
    """Yield (circuit, algorithm_name, num_qubits) from Qiskit circuit library."""

    # --- VQE-style variational circuits ---
    for nq in QUBIT_RANGE:
        # TwoLocal
        try:
            qc = TwoLocal(
                nq,
                rotation_blocks=["ry", "rz"],
                entanglement_blocks="cx",
                entanglement="linear",
                reps=2,
            )
            qc = qc.decompose()
            yield qc, "vqe_twolocal", nq
        except Exception as e:
            fail_logger.debug(f"TwoLocal {nq}q: {e}")

        # EfficientSU2
        try:
            qc = EfficientSU2(nq, reps=2)
            qc = qc.decompose()
            yield qc, "vqe_efficientsu2", nq
        except Exception as e:
            fail_logger.debug(f"EfficientSU2 {nq}q: {e}")

        # RealAmplitudes
        try:
            qc = RealAmplitudes(nq, reps=3)
            qc = qc.decompose()
            yield qc, "vqe_realamplitudes", nq
        except Exception as e:
            fail_logger.debug(f"RealAmplitudes {nq}q: {e}")

    # --- QAOA ---
    for nq in QUBIT_RANGE:
        try:
            from qiskit.circuit.library import QAOAAnsatz
            from qiskit.quantum_info import SparsePauliOp
            import numpy as np

            # Random MaxCut-style cost operator
            terms = []
            for i in range(nq - 1):
                z_str = ["I"] * nq
                z_str[i] = "Z"
                z_str[i + 1] = "Z"
                terms.append(("".join(z_str), random.uniform(-1, 1)))
            cost_op = SparsePauliOp.from_list(terms)
            qc = QAOAAnsatz(cost_operator=cost_op, reps=2)
            qc = qc.decompose().decompose()
            yield qc, "qaoa", nq
        except Exception as e:
            fail_logger.debug(f"QAOAAnsatz {nq}q: {e}")

    # --- QFT ---
    for nq in QUBIT_RANGE:
        try:
            qc = QiskitQFT(nq)
            qc = qc.decompose()
            yield qc, "qft", nq
        except Exception as e:
            fail_logger.debug(f"QFT {nq}q: {e}")

    # --- GroverOperator ---
    for nq in range(3, 9):  # Grover capped at 8 qubits (exponentially slow)
        try:
            oracle = QuantumCircuit(nq)
            # Mark a random state
            target_state = random.randint(0, 2**nq - 1)
            bin_str = format(target_state, f"0{nq}b")
            for i, bit in enumerate(bin_str):
                if bit == "0":
                    oracle.x(i)
            oracle.h(nq - 1)
            oracle.mcx(list(range(nq - 1)), nq - 1)
            oracle.h(nq - 1)
            for i, bit in enumerate(bin_str):
                if bit == "0":
                    oracle.x(i)

            grover_op = GroverOperator(oracle)
            qc = grover_op.decompose().decompose()
            yield qc, "grover", nq
        except Exception as e:
            fail_logger.debug(f"Grover {nq}q: {e}")

    # --- GHZ states ---
    for nq in QUBIT_RANGE:
        try:
            qc = QuantumCircuit(nq)
            qc.h(0)
            for i in range(nq - 1):
                qc.cx(i, i + 1)
            yield qc, "ghz", nq
        except Exception as e:
            fail_logger.debug(f"GHZ {nq}q: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Source 3: Synthetic random circuits
# ═══════════════════════════════════════════════════════════════════════════
def generate_random_circuits(count: int = RANDOM_COUNT):
    """Yield (circuit, 'random', num_qubits) for random circuits."""
    for _ in range(count):
        nq = random.randint(3, 20)
        depth = random.randint(5, 50)
        try:
            qc = random_circuit(nq, depth, max_operands=2, seed=None)
            yield qc, "random", nq
        except Exception as e:
            fail_logger.debug(f"random_circuit {nq}q depth={depth}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint & save helpers
# ═══════════════════════════════════════════════════════════════════════════
def save_json(data: list, path: Path):
    """Save data list to a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def print_summary_table(dataset: list):
    """Print a summary table: count and avg improvement per algorithm."""
    stats = defaultdict(lambda: {"count": 0, "total_improvement": 0.0})

    for record in dataset:
        algo = record["algorithm"]
        stats[algo]["count"] += 1
        stats[algo]["total_improvement"] += record["improvement_ratio"]

    # Sort by count descending
    sorted_algos = sorted(stats.items(), key=lambda x: -x[1]["count"])

    print("\n" + "=" * 62)
    print(f"{'Algorithm':<25} {'Count':>8} {'Avg Improvement':>18}")
    print("-" * 62)

    total_count = 0
    total_improvement = 0.0

    for algo, s in sorted_algos:
        avg = s["total_improvement"] / s["count"] if s["count"] else 0
        print(f"  {algo:<23} {s['count']:>8}   {avg:>14.1%}")
        total_count += s["count"]
        total_improvement += s["total_improvement"]

    print("-" * 62)
    overall_avg = total_improvement / total_count if total_count else 0
    print(f"  {'TOTAL':<23} {total_count:>8}   {overall_avg:>14.1%}")
    print("=" * 62)


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════
def load_checkpoint() -> list:
    """Resume from checkpoint if it exists."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE) as f:
                data = json.load(f)
            print(f"📂 Resuming from checkpoint: {len(data)} circuits already done")
            return data
        except Exception as e:
            print(f"⚠️  Checkpoint file corrupt, starting fresh: {e}")
    return []


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Quantum Circuit Dataset Generator for GNN Compiler        ║")
    print("║  Target: 15,000+ labeled circuits → dataset.json           ║")
    print("║  Timeout: 30s per circuit | MQTBench: 3-10 qubits          ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # ── Resume from checkpoint if available ────────────────────────────
    dataset = load_checkpoint()
    existing_keys = set()
    for r in dataset:
        existing_keys.add((r["algorithm"], r["num_qubits"]))
    if dataset:
        print(f"  → Skipping {len(existing_keys)} already-labeled (algo, qubit) combos\n")

    failed_count = 0
    timeout_count = 0
    total_attempted = 0
    start_time = time.time()

    # Pre-warm the backend
    get_backend()

    # ── Collect all circuit generators ──────────────────────────────────
    generators = [
        ("MQTBench", generate_mqtbench_circuits()),
        ("Qiskit Library", generate_qiskit_library_circuits()),
        ("Random Circuits", generate_random_circuits(RANDOM_COUNT)),
    ]

    for source_name, gen in generators:
        print(f"\n{'─' * 50}")
        print(f"📦 Source: {source_name}")
        print(f"{'─' * 50}")

        source_count = 0

        for circuit, algo, nq in gen:
            total_attempted += 1

            # Skip if already in checkpoint (for resume)
            # For random circuits we don't skip (keys aren't unique)
            if algo != "random" and (algo, nq) in existing_keys:
                continue

            try:
                record = label_circuit(circuit, algo, nq)
                dataset.append(record)
                source_count += 1

                # Progress reporting
                total_done = len(dataset)
                if total_done % PROGRESS_EVERY == 0:
                    elapsed = time.time() - start_time
                    rate = (total_done - len(existing_keys)) / elapsed if elapsed > 0 else 0
                    remaining = max(0, TARGET_TOTAL - total_done)
                    eta_min = remaining / rate / 60 if rate > 0 else 0
                    print(
                        f"  ✓ {total_done}/{TARGET_TOTAL} circuits — "
                        f"{algo} {nq}q  "
                        f"[{rate:.1f}/s, ETA ~{eta_min:.0f}m]"
                    )

                # Checkpoint
                if total_done % CHECKPOINT_EVERY == 0:
                    save_json(dataset, CHECKPOINT_FILE)
                    print(f"  💾 Checkpoint saved at {total_done} circuits")

            except KeyboardInterrupt:
                signal.alarm(0)  # Cancel any pending alarm
                print("\n\n⚠️  Interrupted! Saving progress...")
                save_json(dataset, CHECKPOINT_FILE)
                save_json(dataset, OUTPUT_FILE)
                print(f"  Saved {len(dataset)} circuits to {OUTPUT_FILE}")
                sys.exit(0)

            except CircuitTimeoutError:
                timeout_count += 1
                fail_logger.error(f"TIMEOUT | {algo} | {nq}q | {source_name} | >{TIMEOUT_SECONDS}s")
                if timeout_count <= 10 or timeout_count % 20 == 0:
                    print(f"  ⏰ Timeout: {algo} {nq}q skipped (>{TIMEOUT_SECONDS}s)")

            except Exception as e:
                failed_count += 1
                signal.alarm(0)  # Cancel any pending alarm
                tb = traceback.format_exc()
                fail_logger.error(
                    f"FAILED | {algo} | {nq}q | {source_name}\n{tb}"
                )
                # Don't print every failure, just log to file
                if failed_count % 50 == 0:
                    print(f"  ⚠️  {failed_count} failures so far (see {FAILED_LOG})")

        print(f"  ↳ {source_count} circuits from {source_name}")

    # ── Final save ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    save_json(dataset, OUTPUT_FILE)
    print(f"\n✅ Done! {len(dataset)} circuits saved to {OUTPUT_FILE}")
    print(f"⏱  Total time: {elapsed / 60:.1f} minutes")
    print(f"❌ Failed: {failed_count} | Timeouts: {timeout_count} (details in {FAILED_LOG})")

    # ── Summary table ──────────────────────────────────────────────────
    print_summary_table(dataset)

    # ── Cleanup checkpoint if full run completed ───────────────────────
    if CHECKPOINT_FILE.exists():
        print(f"\n🗂  Final checkpoint retained at {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
