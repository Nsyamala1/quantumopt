#!/usr/bin/env python3
"""
diagnose_swap.py — Shows exactly how SWAP gates are inserted during hardware mapping.

Prints gate counts and gate types for a 4-qubit QAOA circuit at three stages:
  1. Raw (abstract gates, no transpilation)
  2. optimization_level=0 → naive decomposition into native gates, no routing tricks
  3. optimization_level=3 → full optimizer + routing; shows real SWAP overhead

Usage:
    python diagnose_swap.py
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import random
import numpy as np
from qiskit import QuantumCircuit, transpile

random.seed(0)
np.random.seed(0)

# ── Build a simple 4-qubit QAOA circuit (2 layers, fixed edges) ───────────────
n_qubits = 4
p_layers = 2

edges  = [(0, 1), (1, 2), (2, 3), (0, 3)]   # ring topology (MaxCut)
beta   = [np.pi / 4, np.pi / 8]
gamma  = [np.pi / 3, np.pi / 6]

qc = QuantumCircuit(n_qubits, name="QAOA_4q_p2")
for q in range(n_qubits):
    qc.h(q)
for layer in range(p_layers):
    for (u, v) in edges:
        qc.cx(u, v)
        qc.rz(2 * gamma[layer], v)
        qc.cx(u, v)
    for q in range(n_qubits):
        qc.rx(2 * beta[layer], q)

BASIS = ["cx", "u", "sx", "rz", "x"]

# ── Helper ────────────────────────────────────────────────────────────────────
def print_circuit_info(label: str, circuit: QuantumCircuit):
    ops = dict(circuit.count_ops())
    total = sum(ops.values())
    gate_str = "  ".join(f"{g}={n}" for g, n in sorted(ops.items()))
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(f"  Qubits : {circuit.num_qubits}")
    print(f"  Depth  : {circuit.depth()}")
    print(f"  Total gates : {total}")
    print(f"  Gate breakdown : {gate_str if gate_str else '(none)'}")
    swap_count = ops.get("swap", 0)
    if swap_count:
        print(f"  ⚠️  SWAP gates inserted : {swap_count}")
    else:
        print(f"  ✅ No SWAP gates")

# ── Stage 1: Raw circuit ──────────────────────────────────────────────────────
print_circuit_info("STAGE 1 — Raw circuit (abstract gates, no transpilation)", qc)

# ── Stage 2: optimization_level=0 ────────────────────────────────────────────
lvl0 = transpile(qc, basis_gates=BASIS, optimization_level=0, seed_transpiler=42)
print_circuit_info("STAGE 2 — optimization_level=0  (naive decomposition, no routing tricks)", lvl0)

# ── Stage 3: optimization_level=3 ────────────────────────────────────────────
lvl3 = transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=42)
print_circuit_info("STAGE 3 — optimization_level=3  (full optimization + routing)", lvl3)

# ── Summary ───────────────────────────────────────────────────────────────────
lvl0_gates = sum(dict(lvl0.count_ops()).values())
lvl3_gates = sum(dict(lvl3.count_ops()).values())
gate_red   = (lvl0_gates - lvl3_gates) / max(lvl0_gates, 1) * 100
depth_red  = (lvl0.depth() - lvl3.depth()) / max(lvl0.depth(), 1) * 100

print(f"\n{'═' * 60}")
print("  SUMMARY  (level-0 baseline  →  level-3 optimized)")
print(f"{'═' * 60}")
print(f"  Gate count : {lvl0_gates}  →  {lvl3_gates}   ({gate_red:+.1f}%)")
print(f"  Depth      : {lvl0.depth()}  →  {lvl3.depth()}   ({depth_red:+.1f}%)")
print(f"{'═' * 60}\n")
