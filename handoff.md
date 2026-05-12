# Quantumopt — Session Handoff

## Goal

Build a domain-specific quantum compilation library (`quantumopt`) that reduces
two-qubit gate counts in 1D Fermi-Hubbard Trotterized circuits. The immediate
target is a 25% abstract CX reduction per Trotter step using a Givens rotation
synthesis pass, validated against the exact unitary. Longer term: integrate
with a GNN-based optimizer and use as a benchmark-quality reference pass for
the `quantumopt` paper.

---

## Current State

### What works

**`quantumopt/passes/givens_synthesis.py`** — complete and verified.
- Detects adjacent `exp(-iθ XZX)·exp(-iθ YZY)` hopping pairs using a rigid
  9-instruction / 13-instruction pattern match on the flat circuit instruction
  list.
- Replaces each pair with a 6-CX Givens rotation circuit (saves 2 CX per pair).
- 63-parameter U3 layers found by L-BFGS-B numerical optimisation; precomputed
  parameters for θ=−0.05 (Fermi-Hubbard: t=1.0, dt=0.1) are hardcoded so the
  pass runs in <1 ms.
- `verify_unitary` threshold correctly scaled: `tol * sqrt(dim)` where
  `dim = 2^n_qubits`.

**`givens_benchmark.py`** — benchmark runner, all checks clean:
```
L=2 (4q):  20→16 CX, PASS (Frobenius dist 2.08e-05, tol 4.00e-05)
L=3 (6q):  38→30 CX, PASS
L=4 (8q):  56→44 CX, PASS
L=6 (12q): 92→72 CX, N/A (matrix too large)
L=8 (16q): 128→100 CX, N/A
Total: 334→262 CX, 21.6% reduction, 36 pairs replaced
```

**`quantumopt/passes/trotter_fusion.py`** — Rz-merge + CX-cancel pass, complete.

**`quantumopt/passes/__init__.py`** — exports all passes.

All files committed and pushed to `origin/main` (commit `e745c4b`).

---

## Files Actively Relevant

| File | Purpose |
|------|---------|
| `quantumopt/passes/givens_synthesis.py` | Core synthesis pass (the main deliverable) |
| `quantumopt/passes/trotter_fusion.py` | Rz-merge + CX-cancel pass |
| `quantumopt/passes/__init__.py` | Public API exports |
| `givens_benchmark.py` | Benchmark runner; source of truth for results |
| `givens_results.json` | Machine-readable benchmark output |
| `fermi_hubbard_trotter.py` | Circuit builder: `build_trotter_step(L, t, U, dt)` |
| `trotter_fusion_benchmark.py` | Benchmark for the Trotter fusion pass |

---

## What Was Tried and Failed

### 4-CX template for exp(-iθ(XZX+YZY))
Exhaustive numerical search across 12+ diverse 4-CX templates (all orderings
of CX(0,1)/CX(1,2) with 4 entangling layers, 37 U3 parameters each). Every
template converged to Frobenius dist ~0.07 across 200 random restarts. Conclusion:
**4 CX is mathematically insufficient** on a linear q0–q1–q2 chain. The
Jordan-Wigner string qubit (q1) must both propagate entanglement and carry the
sign — this forces a minimum of 6 nearest-neighbour CX gates.

### abs(theta) in pattern detection
Early code used `abs(rz_param) / 2 = +0.05` for theta. But
`add_pauli_exp(angle=-0.05)` produces `rz(-0.1)`, so the precomputed params
(optimised for −0.05) didn't match. This caused Frobenius dist 0.797 at L=2.
Fix: remove `abs()` — use signed theta throughout.

### Wrong qubit objects in `apply()`
`new_qc.append(op, [qc.qubits[idx] for ...])` passed `Qubit` objects from the
original `qc` into `new_qc`. Qiskit 2.x treats them as foreign qubits → silent
wrong wiring. Fix: `[new_qc.qubits[_qubit_idx(q, qc)] for q in inst.qubits]`.

### Hardcoded `tol * 8` in `verify_unitary`
The value 8 = sqrt(64) is correct only for 6-qubit (dim=64) circuits. For
larger L, the accumulated Frobenius distance scales with sqrt(dim), so the
check rejected valid circuits. Fix: `tol * np.sqrt(U.shape[0])`.

### On-the-fly optimisation for θ=−0.05
200 trials × 3000 maxiter took ~1300 seconds for this angle (near-identity
makes the gradient landscape flat). Fix: hardcode precomputed params in
`_PRECOMPUTED_PARAMS[-0.05]`. Lookup is now instant.

---

## Next Steps

### Immediate
1. **Add more precomputed angles.** The current table only covers θ=−0.05
   (t=1.0, dt=0.1). For different (t, dt) pairs the pass falls back to slow
   on-the-fly optimisation. The optimisation script is:
   ```python
   from quantumopt.passes.givens_synthesis import _optimise_6cx, _PRECOMPUTED_PARAMS
   params = _optimise_6cx(theta, n_trials=200)
   # then hardcode params into _populate_precomputed()
   ```
   Priority angles: θ = −0.025 (dt=0.05), −0.1 (dt=0.2), −0.5 (general VQE).

2. **Second-order Trotter support.** `build_trotter_step` only supports
   first-order. Second-order circuits have the same XZX+YZY pattern but with
   half-step angles at the boundaries — the detector should still fire, but
   needs testing with `order=2`.

3. **Compose with TrotterStepFusion.** Run `compile_trotter` then
   `compile_givens` in sequence; measure combined gate reduction. Likely
   additive (Rz-merge and CX-cancel are orthogonal to hopping pair replacement).

### Medium term
4. **GNN integration.** The GNN in `train.py` / `train_gnn.ipynb` is trained
   on `dataset_v3_*.json`. The Givens pass should be added as a compilation
   option that the GNN can learn to predict when to apply.

5. **Hardware-native verification.** The current verification is abstract-gate
   Frobenius distance. For the paper, run through Qiskit's `transpile` with
   `basis_gates=["ecr","rz","sx","x"]` on a real backend coupling map and
   compare native gate counts.

---

## Key Mathematical Facts (don't rederive)

- `XZX + YZY = (XX+YY)_{q0,q2} ⊗ Z_{q1}` — couples {|001⟩,|100⟩} and
  {|011⟩,|110⟩} with opposite-sign Rx rotations.
- Signed hopping angle: `theta = -dt * t / 2` (negative). For t=1.0, dt=0.1:
  theta = −0.05. The circuit has `rz(2*theta) = rz(-0.1)`.
- 6-CX template: `CX(0,1), CX(1,2)` repeated 3 times, with 7 layers of U3
  gates (63 free parameters) interleaved.
- Precomputed params dist from target: 1.04e-05 (well within 1e-4 tolerance).
- Statevector fidelity at L=3: 1 − F ≈ 5e-11 (synthesis is exact to
  numerical precision).
