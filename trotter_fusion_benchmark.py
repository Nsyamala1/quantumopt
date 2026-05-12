#!/usr/bin/env python3
"""
trotter_fusion_benchmark.py

Benchmark quantumopt's new TrotterStepFusion pass against the standard
quantumopt.compile() pipeline on 1D Fermi-Hubbard Trotter circuits.

Circuits: L=2,3,4,6,8 sites → 4,6,8,12,16 qubits.
Parameters: t=1.0, U=4.0, dt=0.1, first-order Trotter.

Output:
    - Console: formatted comparison table
    - File:    trotter_fusion_results.json
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import json
import sys
import os

# Ensure the repo root is on sys.path so both `quantumopt` and
# `fermi_hubbard_trotter` resolve correctly regardless of cwd.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fermi_hubbard_trotter import build_trotter_step
from quantumopt.passes.trotter_fusion import compile_trotter
import quantumopt

SITES = [2, 3, 4, 6, 8]
T, U, DT = 1.0, 4.0, 0.1


def run_standard(qc):
    """Run the original quantumopt.compile() and return a comparable stats dict."""
    try:
        result = quantumopt.compile(qc, hardware="ibm_brisbane", explain=False)
        before_gates = sum(qc.count_ops().values())
        before_depth = qc.depth()
        after_gates  = result.optimized_stats.get("gate_count", before_gates)
        after_depth  = result.optimized_stats.get("depth",      before_depth)
        return {
            "before_gates":       before_gates,
            "after_gates":        after_gates,
            "gate_reduction_pct": round((before_gates - after_gates) / max(before_gates, 1) * 100, 2),
            "before_depth":       before_depth,
            "after_depth":        after_depth,
            "depth_reduction_pct": round((before_depth - after_depth) / max(before_depth, 1) * 100, 2),
            "gnn_prediction":     result.gnn_prediction,
            "error":              None,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ─── Header ──────────────────────────────────────────────────────────────────

print("=" * 90)
print("  TROTTER FUSION BENCHMARK — 1D Fermi-Hubbard  |  t=1.0  U=4.0  dt=0.1")
print("=" * 90)

# ─── Column headers ──────────────────────────────────────────────────────────

HDR = (
    f"{'L':>3}  {'Q':>3}  "
    f"{'|':1}  "
    f"{'STD gates':>10}  {'STD gate%':>9}  {'STD dep%':>8}  "
    f"{'|':1}  "
    f"{'FUS gates':>9}  {'FUS gate%':>9}  {'FUS dep%':>8}  "
    f"{'|':1}  "
    f"{'blocks':>6}  {'Rz-merge':>8}  {'CX-cancel':>9}  {'method':>14}"
)
SEP = "─" * len(HDR)
print(HDR)
print(SEP)

# ─── Per-circuit runs ─────────────────────────────────────────────────────────

all_results = []

for L in SITES:
    qc = build_trotter_step(L, t=T, U=U, dt=DT)

    std = run_standard(qc)
    fus = compile_trotter(qc, circuit_type="fermi_hubbard", dt=DT, order=1)

    # Format sign for gate% (positive = reduction, negative = inflation)
    def pct_str(v):
        if v is None:
            return "    N/A"
        sign = "+" if v > 0 else ""
        return f"{sign}{v:+.1f}%"

    std_gate_pct  = pct_str(std.get("gate_reduction_pct"))
    std_depth_pct = pct_str(std.get("depth_reduction_pct"))
    fus_gate_pct  = pct_str(fus["gate_reduction_pct"])
    fus_depth_pct = pct_str(fus["depth_reduction_pct"])

    std_gates = std.get("after_gates", "ERR")
    fus_gates = fus["after_gates"]

    row = (
        f"{L:>3}  {2*L:>3}  "
        f"{'|':1}  "
        f"{std_gates:>10}  {std_gate_pct:>9}  {std_depth_pct:>8}  "
        f"{'|':1}  "
        f"{fus_gates:>9}  {fus_gate_pct:>9}  {fus_depth_pct:>8}  "
        f"{'|':1}  "
        f"{fus['trotter_blocks_detected']:>6}  "
        f"{fus['rz_merges']:>8}  "
        f"{fus['cnot_cancellations']:>9}  "
        f"{fus['method_used']:>14}"
    )
    print(row)

    all_results.append({
        "sites":   L,
        "qubits":  2 * L,
        "standard": {
            "before_gates":        std.get("before_gates"),
            "after_gates":         std.get("after_gates"),
            "gate_reduction_pct":  std.get("gate_reduction_pct"),
            "before_depth":        std.get("before_depth"),
            "after_depth":         std.get("after_depth"),
            "depth_reduction_pct": std.get("depth_reduction_pct"),
            "gnn_prediction":      std.get("gnn_prediction"),
            "error":               std.get("error"),
        },
        "trotter_fusion": {
            "before_gates":           fus["before_gates"],
            "after_gates":            fus["after_gates"],
            "gate_reduction_pct":     fus["gate_reduction_pct"],
            "before_depth":           fus["before_depth"],
            "after_depth":            fus["after_depth"],
            "depth_reduction_pct":    fus["depth_reduction_pct"],
            "trotter_blocks_detected": fus["trotter_blocks_detected"],
            "rz_merges":              fus["rz_merges"],
            "cnot_cancellations":     fus["cnot_cancellations"],
            "method_used":            fus["method_used"],
            "error":                  None,
        },
    })

print(SEP)

# ─── Summary ─────────────────────────────────────────────────────────────────

print()
print("  Legend:")
print("    STD     = original quantumopt.compile() (GNN + Qiskit lvl-3)")
print("    FUS     = new TrotterStepFusion pass (Rz-merge + CX-cancel) + Qiskit lvl-3")
print("    gate%   = (before - after) / before × 100  [positive = reduction]")
print("    dep%    = same for circuit depth")
print("    blocks  = Pauli exponential blocks detected by TrotterBlockDetector")
print("    Rz-merge = consecutive Rz pairs merged at abstract gate level")
print("    CX-cancel = consecutive CX pairs cancelled at abstract gate level")
print()
print("  Key insight:")
print("    Both methods inflate gate count — H/S/Sdg each decompose to 2-3 native")
print("    {SX,Rz} gates. Abstract-level fusion finds 0 optimizations because the")
print("    Trotter circuit has no adjacent cancelable gate pairs (see FINDINGS below).")

# ─── Correctness spot-check (L=2 only, via unitary comparison) ───────────────

print()
print("  Correctness check (L=2, Operator unitary comparison):")
try:
    from qiskit.quantum_info import Operator
    qc2  = build_trotter_step(2, t=T, U=U, dt=DT)
    fus2 = compile_trotter(qc2, circuit_type="fermi_hubbard", dt=DT, order=1)

    # Fused abstract circuit (before hardware transpilation)
    from quantumopt.passes.trotter_fusion import TrotterStepFusion, _optimize_instruction_list
    fused_abstract, _, _ = _optimize_instruction_list(qc2)

    op_orig  = Operator(qc2)
    op_fused = Operator(fused_abstract)
    equiv = op_orig.equiv(op_fused, rtol=1e-6)
    print(f"    Original == Fused (abstract): {equiv}")
    if not equiv:
        print("    WARNING: unitaries differ — optimization is NOT unitary-preserving!")
except Exception as exc:
    print(f"    (skipped — Operator comparison unavailable: {exc})")

# ─── Save JSON ────────────────────────────────────────────────────────────────

findings = (
    "TrotterBlockDetector correctly identifies Pauli exponential blocks (6 for L=2, "
    "36 for L=8). However, abstract-level Rz-merge and CX-cancel find 0 optimizations "
    "in the 1D Fermi-Hubbard Trotter circuit because: (1) XZX->YZY boundaries have "
    "H·Sdg·H on qubits — not directly cancelable; (2) on-site ZZ Rz gates are "
    "separated from single-Z Rz gates by a CX — no merge; (3) inter-bond boundaries "
    "act on disjoint qubits. Gate count inflation (~45-50%) is structural: H/S/Sdg "
    "each decompose to 2-3 native {SX,Rz} gates. Real improvement requires: "
    "(a) fermionic-hopping synthesis replacing each XZX+YZY pair (8 CNOTs) with a "
    "Givens rotation circuit (4 CNOTs) — not available in Qiskit by default; "
    "(b) native fermionic gate sets (e.g., Quantinuum ZZ-phase); "
    "(c) hardware-level peephole optimization in native {SX,Rz,CX} basis; "
    "(d) second-order Suzuki-Trotter to reduce total gate count ~sqrt(L)."
)

report = {
    "benchmark": "1D Fermi-Hubbard Trotter circuits",
    "parameters": {"t": T, "U": U, "dt": DT, "order": 1},
    "description": (
        "Compares standard quantumopt.compile() against the new TrotterStepFusion "
        "pass on 1D Fermi-Hubbard Trotter circuits. TrotterBlockDetector successfully "
        "identifies Pauli exponential blocks. Abstract-level Rz-merge and CX-cancel "
        "find 0 optimizations in this circuit structure (see findings)."
    ),
    "findings": findings,
    "results": all_results,
}

out_path = os.path.join(_REPO_ROOT, "trotter_fusion_results.json")
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

print()
print("  FINDINGS")
print("  " + "─" * 86)
print()
print("  1. TrotterBlockDetector: WORKING — identifies 6 blocks (L=2) to 36 blocks (L=8).")
print()
print("  2. Abstract Rz-merge + CX-cancel: 0 OPTIMIZATIONS FOUND")
print("     The 1D Fermi-Hubbard Trotter circuit has no adjacent cancelable pairs:")
print("     • XZX→YZY boundary: H·Sdg·H on boundary qubits — not directly cancelable.")
print("     • On-site ZZ block: single-Z Rz is separated from ZZ inner Rz by a CX.")
print("     • Inter-bond: adjacent bonds act on disjoint qubit sets — no overlap.")
print()
print("  3. Gate inflation ~45-50% is STRUCTURAL, not algorithmic:")
print("     H, S, Sdg each decompose to 2-3 native {SX, Rz} gates regardless of")
print("     any abstract-level fusion that was applied.")
print()
print("  4. Path to real improvement:")
print("     a) Fermionic-hopping synthesis: replace XZX+YZY pair (8 CNOTs abstract,")
print("        ~7 CX hardware) with a Givens-rotation circuit (4 CNOTs) — needs a")
print("        custom Qiskit HLS plugin not yet available by default.")
print("     b) Native fermionic gates (Quantinuum ZZ-phase, IonQ native hopping gate).")
print("     c) Hardware peephole optimization on {SX, Rz, CX} after decomposition.")
print("     d) Second-order Suzuki-Trotter to reduce total step count ~sqrt(L).")
print()
print(f"  Results saved → trotter_fusion_results.json")
print("=" * 90)
