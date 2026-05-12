#!/usr/bin/env python3
"""
suzuki_trotter_benchmark.py

Benchmarks second-order Suzuki-Trotter decomposition combined with
Givens rotation synthesis for 1D Fermi-Hubbard circuits.

Compares three approaches at same total evolution time T=0.4:
  A. First-order Trotter  (4 steps × dt=0.1)
  B. First-order + Givens synthesis
  C. Second-order Suzuki-Trotter + Givens synthesis (2 steps × dt=0.2)

Key result: combined CX reduction from Suzuki (fewer steps) +
Givens (6 CX per hopping pair vs 8 CX naive).
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import json, sys, os, time
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fermi_hubbard_trotter import add_hopping_term, add_onsite_term, build_trotter_step
from quantumopt.passes.givens_synthesis import (
    GivensRotationSynthesis, _frobenius_dist, _cached_params,
)

# ── Simulation parameters ────────────────────────────────────────────────────

SITES      = [2, 4, 6, 8]
T_TOTAL    = 0.4          # fixed total evolution time for all methods
T_HOP      = 1.0          # Fermi-Hubbard hopping amplitude
U_ONSITE   = 4.0          # Fermi-Hubbard on-site interaction

N_STEPS_1 = 4             # first-order: 4 steps × dt=0.10 → T=0.4
DT_1      = T_TOTAL / N_STEPS_1   # 0.10

N_STEPS_2 = 2             # second-order: 2 steps × dt=0.20 → T=0.4
DT_2      = T_TOTAL / N_STEPS_2   # 0.20
# half-hop in Suzuki-2 uses dt/2=0.10 → angle θ=-0.05 (precomputed ✓)


# ── Circuit building helpers ──────────────────────────────────────────────────

def _add_hopping_block(qc: QuantumCircuit, L: int, t: float, dt: float) -> None:
    """All hopping bonds (both spins) at time step dt."""
    for i in range(L - 1):
        add_hopping_term(qc, 2*i,   2*i+1,   2*(i+1),   t, dt)  # spin-up
        add_hopping_term(qc, 2*i+1, 2*(i+1), 2*(i+1)+1, t, dt)  # spin-down


def _add_onsite_block(qc: QuantumCircuit, L: int, U: float, dt: float) -> None:
    """All on-site interaction terms at time step dt."""
    for i in range(L):
        add_onsite_term(qc, 2*i, 2*i+1, U, dt)


def build_trotter_n_steps(
    L: int, t: float, U: float, dt: float, n_steps: int
) -> QuantumCircuit:
    """First-order Trotter circuit for n_steps steps."""
    n_qubits = 2 * L
    qc = QuantumCircuit(n_qubits, name=f"FH_1st_L{L}_n{n_steps}")
    step = build_trotter_step(L, t, U, dt)
    for _ in range(n_steps):
        qc.compose(step, inplace=True)
    return qc


def build_fermi_hubbard_suzuki2(
    L: int, t: float, U: float, dt: float, n_steps: int
) -> QuantumCircuit:
    """
    Second-order Suzuki-Trotter for L-site 1D Fermi-Hubbard.

    Each step: e^{-iH_hop dt/2} · e^{-iH_onsite dt} · e^{-iH_hop dt/2}
    Repeated n_steps times (naive form — no leapfrog merging).

    With dt=0.2: half-hops use dt/2=0.1 → angle θ=-0.05 (precomputed ✓).
    Error O(dt³) per step vs O(dt²) for first-order.
    """
    n_qubits = 2 * L
    qc = QuantumCircuit(n_qubits, name=f"FH_Suz2_L{L}_n{n_steps}")
    for _ in range(n_steps):
        _add_hopping_block(qc, L, t, dt / 2)   # e^{-iH_hop dt/2}
        _add_onsite_block(qc, L, U, dt)         # e^{-iH_onsite dt}
        _add_hopping_block(qc, L, t, dt / 2)   # e^{-iH_hop dt/2}
    return qc


# ── Exact Hamiltonian for Trotter error ───────────────────────────────────────

_PAULI_MAT = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_on_qubits(n_qubits: int, qp_pairs: list) -> np.ndarray:
    """
    Full Hilbert-space Pauli matrix for qp_pairs = [(qubit_idx, 'X'/'Y'/'Z'), ...].
    Qiskit convention: qubit 0 = least significant bit.
    kron order: P_{n-1} ⊗ ... ⊗ P_1 ⊗ P_0
    """
    single = ['I'] * n_qubits
    for q, p in qp_pairs:
        single[q] = p
    mat = _PAULI_MAT[single[0]]
    for i in range(1, n_qubits):
        mat = np.kron(_PAULI_MAT[single[i]], mat)
    return mat


def build_fh_hamiltonian(L: int, t: float, U: float) -> np.ndarray:
    """
    Exact FH Hamiltonian matrix (2L qubits, JW encoding).

    H = -t/2 Σ_bonds (XZX + YZY)  [hopping]
        + U/4 Σ_sites (I - Z_up - Z_dn + Z_up Z_dn)  [on-site]
    """
    n = 2 * L
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(L - 1):
        for (ql, qjw, qr) in [
            (2*i,   2*i+1,   2*(i+1)),       # spin-up bond
            (2*i+1, 2*(i+1), 2*(i+1)+1),     # spin-down bond
        ]:
            H += (-t / 2) * _pauli_on_qubits(n, [(ql,'X'),(qjw,'Z'),(qr,'X')])
            H += (-t / 2) * _pauli_on_qubits(n, [(ql,'Y'),(qjw,'Z'),(qr,'Y')])

    for i in range(L):
        qu, qd = 2*i, 2*i+1
        H += (U / 4) * np.eye(dim, dtype=complex)
        H += (-U / 4) * _pauli_on_qubits(n, [(qu, 'Z')])
        H += (-U / 4) * _pauli_on_qubits(n, [(qd, 'Z')])
        H += (U / 4)  * _pauli_on_qubits(n, [(qu,'Z'),(qd,'Z')])

    return H


def trotter_error(circuit: QuantumCircuit, H: np.ndarray, T: float):
    """
    ||U_exact - U_trotter||_F up to global phase.
    Returns None if circuit is too large for full Operator computation.
    """
    try:
        U_exact   = expm(-1j * H * T)
        U_trotter = Operator(circuit).data
        return float(_frobenius_dist(U_exact, U_trotter))
    except (MemoryError, ValueError):
        return None


def cx_count(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.name == "cx")


# ── Pre-warm Givens cache ─────────────────────────────────────────────────────

print("=" * 110)
print("  SUZUKI-TROTTER BENCHMARK — 1D Fermi-Hubbard  |  t=1.0  U=4.0")
print(f"  Total evolution time T={T_TOTAL}  |  "
      f"First-order: {N_STEPS_1}×dt={DT_1}  |  "
      f"Suzuki-2: {N_STEPS_2}×dt={DT_2}")
print("=" * 110)

theta_hop  = -DT_1 * T_HOP / 2.0          # -0.05 (full first-order hop angle)
theta_half = -(DT_2 / 2) * T_HOP / 2.0   # -0.05 (half-hop in Suzuki-2, same angle!)

print(f"\n  Givens angle check:")
print(f"    First-order full-hop θ = {theta_hop:.4f}")
print(f"    Suzuki-2 half-hop   θ = {theta_half:.4f}  (same → precomputed params apply ✓)")

print("  Pre-warming Givens cache...")
t0 = time.perf_counter()
_cached_params(theta_hop)
print(f"  Cache ready in {time.perf_counter()-t0:.3f}s\n")


# ── Benchmark loop ────────────────────────────────────────────────────────────

HDR = (
    f"{'L':>3}  {'Q':>3}  | "
    f"{'A:1st(CX)':>9}  {'B:1st+Giv':>9}  {'C:Suz2+Giv':>10}  | "
    f"{'B-red%':>7}  {'C-red%':>7}  | "
    f"{'A dep':>5}  {'B dep':>5}  {'C dep':>5}  | "
    f"{'A err':>10}  {'B err':>10}  {'C err':>10}"
)
SEP = "─" * len(HDR)
print(HDR)
print(SEP)

all_results = []

for L in SITES:
    n_qubits = 2 * L

    H_exact = build_fh_hamiltonian(L, T_HOP, U_ONSITE) if L <= 4 else None

    # Circuit A: first-order baseline
    qc_A  = build_trotter_n_steps(L, T_HOP, U_ONSITE, DT_1, N_STEPS_1)
    cx_A  = cx_count(qc_A)
    dep_A = qc_A.depth()
    err_A = trotter_error(qc_A, H_exact, T_TOTAL) if H_exact is not None else None

    # Circuit B: first-order + Givens
    res_B = GivensRotationSynthesis(verbose=False).apply(qc_A)
    qc_B  = res_B["circuit"]
    cx_B  = res_B["after_cx"]
    dep_B = qc_B.depth()
    err_B = trotter_error(qc_B, H_exact, T_TOTAL) if H_exact is not None else None

    # Circuit C: Suzuki-2 + Givens
    qc_C_raw = build_fermi_hubbard_suzuki2(L, T_HOP, U_ONSITE, DT_2, N_STEPS_2)
    res_C    = GivensRotationSynthesis(verbose=False).apply(qc_C_raw)
    qc_C     = res_C["circuit"]
    cx_C     = res_C["after_cx"]
    dep_C    = qc_C.depth()
    err_C    = trotter_error(qc_C, H_exact, T_TOTAL) if H_exact is not None else None

    red_B = (cx_A - cx_B) / max(cx_A, 1) * 100
    red_C = (cx_A - cx_C) / max(cx_A, 1) * 100

    def fe(e):
        return f"{e:.2e}" if e is not None else "   N/A    "

    print(
        f"{L:>3}  {n_qubits:>3}  | "
        f"{cx_A:>9}  {cx_B:>9}  {cx_C:>10}  | "
        f"{red_B:>+6.1f}%  {red_C:>+6.1f}%  | "
        f"{dep_A:>5}  {dep_B:>5}  {dep_C:>5}  | "
        f"{fe(err_A):>10}  {fe(err_B):>10}  {fe(err_C):>10}"
    )

    all_results.append({
        "sites": L, "qubits": n_qubits,
        "first_order": {
            "n_steps": N_STEPS_1, "dt": DT_1,
            "cx": cx_A, "depth": dep_A, "trotter_error": err_A,
        },
        "first_order_givens": {
            "cx": cx_B, "depth": dep_B, "trotter_error": err_B,
            "cx_reduction_pct": round(red_B, 2),
            "pairs_replaced": res_B["blocks_replaced"],
        },
        "suzuki2_givens": {
            "n_steps": N_STEPS_2, "dt": DT_2,
            "cx": cx_C, "depth": dep_C, "trotter_error": err_C,
            "cx_reduction_pct": round(red_C, 2),
            "pairs_replaced": res_C["blocks_replaced"],
        },
    })

print(SEP)

total_A = sum(r["first_order"]["cx"] for r in all_results)
total_B = sum(r["first_order_givens"]["cx"] for r in all_results)
total_C = sum(r["suzuki2_givens"]["cx"] for r in all_results)
print(f"\n  Totals:  A={total_A}  B={total_B}  C={total_C}")
print(f"  B vs A: {(total_A-total_B)/total_A*100:.1f}% reduction  |  "
      f"C vs A: {(total_A-total_C)/total_A*100:.1f}% combined reduction")


# ── L=2 Validation ────────────────────────────────────────────────────────────

print()
print("=" * 110)
print("  L=2 VALIDATION: Trotter error (same total time T=0.4)")
print("=" * 110)

r2 = all_results[0]
assert r2["sites"] == 2
eA = r2["first_order"]["trotter_error"]
eB = r2["first_order_givens"]["trotter_error"]
eC = r2["suzuki2_givens"]["trotter_error"]

print(f"\n  A. First-order Trotter  ({N_STEPS_1} steps × dt={DT_1:.1f}): "
      f"||U_exact - U_trotter||_F = {eA:.4e}")
print(f"  B. First-order + Givens ({N_STEPS_1} steps × dt={DT_1:.1f}): "
      f"||U_exact - U_trotter||_F = {eB:.4e}  "
      f"[expected ≈ A — Givens is unitary-preserving]")
print(f"  C. Suzuki-2 + Givens    ({N_STEPS_2} steps × dt={DT_2:.1f}): "
      f"||U_exact - U_trotter||_F = {eC:.4e}")

if eA is not None and eC is not None:
    if eC < eA:
        print(f"\n  VALIDATED: Suzuki-2 error ({eC:.2e}) < First-order error ({eA:.2e})")
        print(f"  Error improvement: {eA/eC:.2f}× better accuracy with {N_STEPS_2} steps vs {N_STEPS_1} steps")
    else:
        print(f"\n  NOTE: Suzuki-2 error ({eC:.2e}) >= First-order error ({eA:.2e})")
        print("  (Both errors are small; check if dt=0.2 is still in asymptotic regime)")

cxA2 = r2["first_order"]["cx"]
cxC2 = r2["suzuki2_givens"]["cx"]
print(f"\n  CX counts: A={cxA2}  B={r2['first_order_givens']['cx']}  C={cxC2}")
print(f"  Combined reduction A→C: {(cxA2-cxC2)/cxA2*100:.1f}%  "
      f"({cxA2} → {cxC2} CX, saving {cxA2-cxC2})")


# ── Extrapolation to 120-qubit target ─────────────────────────────────────────

print()
print("=" * 110)
print("  EXTRAPOLATION TO 120-QUBIT TARGET (L=60 sites, 2 qubits/site)")
print("=" * 110)

def _predict_cx(L: int, n_steps: int, hop_cx_per_pair: int, onsite_hops: int) -> int:
    # hop_cx_per_pair: 8 (naive) or 6 (Givens)
    # onsite_hops: number of onsite blocks = n_steps for 1st-order, n_steps for Suzuki-2
    hop_pairs_per_block = 2 * (L - 1)     # bonds × spins
    onsite_cx_per_block = L * 2           # ZZ term per site
    hop_blocks = onsite_hops              # same count for both methods below
    return hop_blocks * hop_pairs_per_block * hop_cx_per_pair + n_steps * onsite_cx_per_block

L60 = 60
pairs60 = 2 * (L60 - 1)  # 118 hopping pairs per block
onsite60 = L60 * 2       # 120 CX per onsite block

# First-order: n_steps hop blocks + n_steps onsite blocks
cx_A60 = N_STEPS_1 * (pairs60 * 8 + onsite60)
cx_B60 = N_STEPS_1 * (pairs60 * 6 + onsite60)
# Suzuki-2: each step has 2 half-hop blocks + 1 onsite block
cx_C60 = N_STEPS_2 * (2 * pairs60 * 6 + onsite60)

print(f"\n  L=60 sites (120 qubits), T={T_TOTAL}:")
print(f"    Hopping pairs per block: {pairs60}  |  Onsite CX per block: {onsite60}")
print(f"\n    A. First-order ({N_STEPS_1} steps, naive):         {cx_A60:>6,} CX")
print(f"    B. First-order + Givens ({N_STEPS_1} steps):      {cx_B60:>6,} CX  "
      f"({(cx_A60-cx_B60)/cx_A60*100:.1f}% vs A)")
print(f"    C. Suzuki-2 + Givens ({N_STEPS_2} steps):         {cx_C60:>6,} CX  "
      f"({(cx_A60-cx_C60)/cx_A60*100:.1f}% vs A)")
print(f"\n    Moderna target: ~9,000 CX per step")
print(f"    Suzuki-2 + Givens per step: ~{cx_C60//N_STEPS_2:,} CX  "
      f"({'BELOW' if cx_C60//N_STEPS_2 < 9000 else 'above'} target)")
print(f"    Additional optimization needed: "
      f"{max(0, cx_C60//N_STEPS_2 - 9000):,} CX to eliminate per step")


# ── Save results ──────────────────────────────────────────────────────────────

def _to_json(obj):
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json(v) for v in obj]
    return obj

report = _to_json({
    "benchmark": "Suzuki-Trotter + Givens synthesis for 1D Fermi-Hubbard",
    "parameters": {
        "t": T_HOP, "U": U_ONSITE, "T_total": T_TOTAL,
        "first_order": {"n_steps": N_STEPS_1, "dt": DT_1},
        "suzuki2":     {"n_steps": N_STEPS_2, "dt": DT_2},
    },
    "methods": {
        "A": f"First-order Trotter baseline ({N_STEPS_1} steps × dt={DT_1})",
        "B": f"First-order + Givens synthesis ({N_STEPS_1} steps × dt={DT_1})",
        "C": f"Second-order Suzuki-Trotter + Givens ({N_STEPS_2} steps × dt={DT_2})",
    },
    "key_insight": (
        "Suzuki-2 half-hops use dt/2=0.1, giving angle θ=-0.05 — identical "
        "to the first-order full-hop angle. Precomputed Givens params apply "
        "to all hopping blocks in both methods without re-optimization."
    ),
    "validation_L2": {
        "T_total": T_TOTAL,
        "trotter_error_A": eA,
        "trotter_error_B": eB,
        "trotter_error_C": eC,
        "suzuki2_lower_error": bool(eC < eA) if eA and eC else None,
        "error_improvement_factor": float(eA / eC) if eA and eC and eC > 0 else None,
    },
    "extrapolation_L60_120q": {
        "cx_A_first_order":       cx_A60,
        "cx_B_first_order_givens": cx_B60,
        "cx_C_suzuki2_givens":    cx_C60,
        "cx_C_per_step":          cx_C60 // N_STEPS_2,
        "moderna_target_per_step": 9000,
        "gap_to_moderna_target":  max(0, cx_C60 // N_STEPS_2 - 9000),
    },
    "results": all_results,
})

out_path = os.path.join(_REPO_ROOT, "suzuki_results.json")
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n  Results saved → suzuki_results.json")
print("=" * 110)
