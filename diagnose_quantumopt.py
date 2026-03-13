#!/usr/bin/env python3
"""
diagnose_quantumopt.py — Inspect exactly what quantumopt.compile() sees
and reports vs what the raw circuit and Qiskit transpile report.

Usage:
    python diagnose_quantumopt.py
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import random
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2

random.seed(42)
np.random.seed(42)

# ── 1. Build 6-qubit VQE circuit ──────────────────────────────────────────────
ansatz = EfficientSU2(num_qubits=6, entanglement="linear", reps=2)
params = {p: random.uniform(0, 2 * np.pi) for p in ansatz.parameters}
qc = ansatz.assign_parameters(params).decompose()

print("═" * 65)
print("  STEP 1 — Raw circuit (after .decompose())")
print("═" * 65)
raw_ops = dict(qc.count_ops())
raw_total = sum(raw_ops.values())
print(f"  num_qubits  : {qc.num_qubits}")
print(f"  depth       : {qc.depth()}")
print(f"  gate_count  : {raw_total}")
print(f"  gate_types  : {dict(sorted(raw_ops.items()))}")

# ── 2. Run quantumopt.compile() ───────────────────────────────────────────────
print("\n" + "═" * 65)
print("  STEP 2 — Running quantumopt.compile(explain=False)")
print("═" * 65)

from quantumopt import compile as qopt_compile
result = qopt_compile(qc, hardware="ibm_brisbane", explain=False)

print("\n  result.original_stats  (dict):")
for k, v in result.original_stats.items():
    print(f"    {k:<28}: {v}")

print("\n  result.optimized_stats  (dict):")
for k, v in result.optimized_stats.items():
    print(f"    {k:<28}: {v}")

print(f"\n  result.gate_reduction    : {result.gate_reduction!r}")
print(f"  result.depth_reduction   : {result.depth_reduction!r}")
print(f"  result.gnn_prediction    : {result.gnn_prediction}")
print(f"  result.compile_time      : {result.compile_time:.3f}s")

# Also inspect the optimized circuit object directly
opt_qc = result.optimized_circuit
if opt_qc is not None:
    opt_ops = dict(opt_qc.count_ops())
    print(f"\n  optimized_circuit direct count:")
    print(f"    depth      : {opt_qc.depth()}")
    print(f"    gate_count : {sum(opt_ops.values())}")
    print(f"    gate_types : {dict(sorted(opt_ops.items()))}")

# ── 3. Raw Qiskit transpile at level 3 for comparison ────────────────────────
print("\n" + "═" * 65)
print("  STEP 3 — Raw Qiskit transpile(optimization_level=3, basis=['cx','u'])")
print("═" * 65)
qiskit_opt = transpile(
    qc,
    basis_gates=["cx", "u"],
    optimization_level=3,
    seed_transpiler=42,
)
qiskit_ops = dict(qiskit_opt.count_ops())
qiskit_total = sum(qiskit_ops.values())
print(f"  depth       : {qiskit_opt.depth()}")
print(f"  gate_count  : {qiskit_total}")
print(f"  gate_types  : {dict(sorted(qiskit_ops.items()))}")

# ── 4. Summary ────────────────────────────────────────────────────────────────
print("\n" + "═" * 65)
print("  SUMMARY")
print("═" * 65)
orig_gates = result.original_stats.get("gate_count", "?")
opt_gates  = result.optimized_stats.get("gate_count", "?")
print(f"  Raw abstract gates (decomposed):   {raw_total}")
print(f"  original_stats gate_count:         {orig_gates}  ← what quantumopt uses as 'before'")
print(f"  optimized_stats gate_count:        {opt_gates}  ← what quantumopt uses as 'after'")
print(f"  Qiskit level-3 gate_count:         {qiskit_total}  ← independent reference")
print(f"  quantumopt reported gate_reduction : {result.gate_reduction}")
if isinstance(orig_gates, int) and isinstance(opt_gates, int):
    actual_pct = (orig_gates - opt_gates) / max(orig_gates, 1) * 100
    print(f"  Recalculated from stats:           {actual_pct:+.1f}%")
print("═" * 65)
