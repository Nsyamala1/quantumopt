"""
quantumopt.passes.trotter_fusion

Trotter-aware compilation pass for Trotterized quantum simulation circuits.

Designed for circuits from the 1D Fermi-Hubbard model (and similar lattice
models) where the gate pattern is repeated Pauli exponential blocks of the form:

    [basis change] → CNOT cascade → Rz(θ) → CNOT cascade† → [undo basis]

Two classes of optimization are applied at the *abstract* gate level, before
hardware transpilation, to avoid the gate-count inflation caused by decomposing
H/S/Sdg into native {SX, Rz} gates:

  1. Rz merge   — consecutive Rz(θ₁), Rz(θ₂) on the same qubit (with no
                  intervening gate on that qubit) are replaced by Rz(θ₁+θ₂);
                  if |θ₁+θ₂| < ε the pair is removed entirely.

  2. CNOT cancel — consecutive CX(a,b), CX(a,b) on the same control/target
                  pair (with no intervening gate on either qubit) are both
                  removed (CX² = I).

Both passes are O(n_gates) — safe for 9 000-gate, 120-qubit circuits.

Public API
----------
    TrotterBlock          namedtuple describing one detected block
    TrotterBlockDetector  detects Pauli exponential blocks in a circuit
    TrotterStepFusion     Qiskit TransformationPass applying both optimizations
    compile_trotter()     end-to-end Trotter-aware compile function
"""

from __future__ import annotations

import math
import numpy as np
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass

# Angle tolerance: rotations smaller than this (mod 2π) are treated as identity
_ANGLE_TOL = 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# TrotterBlock — data container for a detected Pauli exponential block
# ═══════════════════════════════════════════════════════════════════════════════

TrotterBlock = namedtuple(
    "TrotterBlock",
    [
        "pauli_type",   # str  — 'XZX', 'YZY', 'ZZ', 'Z', or 'unknown'
        "qubits",       # list[int] — qubit indices the block acts on
        "angle",        # float — rotation angle θ (the Rz gets 2θ)
        "start_idx",    # int  — index into instruction list of first gate
        "end_idx",      # int  — index into instruction list of last gate
    ],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _qubit_index(qc: QuantumCircuit, qubit) -> int:
    return qc.find_bit(qubit).index


def _inst_qubit_indices(qc: QuantumCircuit, inst: CircuitInstruction) -> List[int]:
    return [_qubit_index(qc, q) for q in inst.qubits]


def _normalize_angle(theta: float) -> float:
    """Reduce angle to (-π, π]."""
    return ((theta + math.pi) % (2 * math.pi)) - math.pi


# ═══════════════════════════════════════════════════════════════════════════════
# TrotterBlockDetector
# ═══════════════════════════════════════════════════════════════════════════════

class TrotterBlockDetector:
    """Detect Pauli exponential blocks in a Trotterized circuit.

    Scans the circuit's instruction list for the core signature of every Pauli
    exponential:

        CX(ctrl, tgt) → Rz(θ, tgt) → CX(ctrl, tgt)           [innermost pair]

    This triple is unique to Pauli exponentials and appears whether the Pauli
    string is XZX, YZY, ZZ, or a longer chain.  The surrounding basis-change
    gates (H, Sdg, S) are also recorded when present.

    Scalability: O(n_gates) single-pass scan.
    """

    def detect(self, qc: QuantumCircuit) -> List[TrotterBlock]:
        """Return a list of TrotterBlock objects found in *qc*.

        Args:
            qc: Any QuantumCircuit (abstract gates; no hardware decomposition).

        Returns:
            List of TrotterBlock namedtuples, one per detected block.
            Empty list if no Trotter structure is found.
        """
        instructions = qc.data
        n = len(instructions)
        blocks: List[TrotterBlock] = []

        # Track the last instruction index that touched each qubit.
        # This lets us check "no intervening gate between A and B on qubit q".
        last_idx: Dict[int, int] = {}  # qubit_index → last instr index

        # pending_cx[i] = (ctrl, tgt) for instruction i if it is a CX
        pending_cx: Dict[Tuple[int, int], int] = {}  # (ctrl,tgt) → instr_idx

        for i, inst in enumerate(instructions):
            name = inst.operation.name
            q_idxs = _inst_qubit_indices(qc, inst)

            if name == "cx" and len(q_idxs) == 2:
                ctrl, tgt = q_idxs
                key = (ctrl, tgt)

                # Check if there's a pending CX(ctrl,tgt) whose tgt qubit has
                # not been touched by anything else since then.
                if key in pending_cx:
                    prev_i = pending_cx[key]
                    # Confirm no gate touched ctrl or tgt between prev_i and i
                    ctrl_clear = (last_idx.get(ctrl, prev_i) == prev_i)
                    tgt_clear  = (last_idx.get(tgt,  prev_i) == prev_i)
                    if ctrl_clear and tgt_clear:
                        # This is a CX-...-CX pair; check what's between them
                        # (should be an Rz on tgt for a Trotter inner block)
                        pass  # handled by the Rz branch below

                pending_cx[key] = i
                for q in q_idxs:
                    last_idx[q] = i

            elif name == "rz" and len(q_idxs) == 1:
                tgt = q_idxs[0]
                theta = float(inst.operation.params[0])

                # Look for CX(?, tgt) immediately before this Rz on tgt.
                # The previous instruction on tgt must be a CX with tgt as target.
                prev_on_tgt = last_idx.get(tgt, -1)
                if prev_on_tgt < 0:
                    last_idx[tgt] = i
                    continue

                prev_inst = instructions[prev_on_tgt]
                prev_q = _inst_qubit_indices(qc, prev_inst)

                if prev_inst.operation.name == "cx" and prev_q[1] == tgt:
                    ctrl = prev_q[0]
                    cx_start = prev_on_tgt

                    # Now scan forward past the Rz to find CX(ctrl, tgt) closing pair.
                    # We record this pending triple for resolution when the closing CX arrives.
                    # Store as a "half-open" block keyed by (ctrl, tgt).
                    key = (ctrl, tgt)
                    pending_cx[key] = i   # point to Rz so the next CX(ctrl,tgt) closes it

                    # Estimate Pauli type from basis gates before cx_start on ctrl and tgt
                    pauli_type = self._infer_pauli_type(instructions, cx_start, ctrl, tgt)

                    # Find the outer CX boundary to determine full qubit span
                    outer_qubits = self._find_outer_cx_chain(instructions, cx_start, qc)

                    last_idx[tgt] = i

                    # Store block info keyed by (ctrl,tgt) for the closing CX to complete
                    pending_cx[("half", ctrl, tgt)] = {
                        "cx_start": cx_start,
                        "rz_idx": i,
                        "angle": theta / 2.0,   # angle = θ/2 since Rz gets 2*angle
                        "pauli_type": pauli_type,
                        "qubits": outer_qubits,
                    }
                    continue

                last_idx[tgt] = i

            else:
                for q in q_idxs:
                    last_idx[q] = i

            # Check if this CX closes a half-open block
            if name == "cx" and len(q_idxs) == 2:
                ctrl, tgt = q_idxs
                half_key = ("half", ctrl, tgt)
                if half_key in pending_cx:
                    info = pending_cx.pop(half_key)
                    blocks.append(TrotterBlock(
                        pauli_type=info["pauli_type"],
                        qubits=info["qubits"],
                        angle=info["angle"],
                        start_idx=info["cx_start"],
                        end_idx=i,
                    ))

        return blocks

    def _infer_pauli_type(
        self,
        instructions: list,
        cx_start: int,
        ctrl: int,
        tgt: int,
    ) -> str:
        """Infer XZX / YZY / ZZ / Z from the basis-change gates before the CNOT."""
        # Look back from cx_start for H, Sdg, S gates on ctrl
        ctrl_basis = None
        for j in range(cx_start - 1, max(cx_start - 5, -1), -1):
            inst_j = instructions[j]
            if inst_j.operation.name in ("h", "sdg", "s", "rx", "ry"):
                ctrl_basis = inst_j.operation.name
                break

        if ctrl_basis == "h":
            return "XZX"
        elif ctrl_basis == "sdg":
            return "YZY"
        elif ctrl_basis is None:
            # No basis change → Z-type
            return "ZZ"
        return "unknown"

    def _find_outer_cx_chain(
        self,
        instructions: list,
        cx_start: int,
        qc: QuantumCircuit,
    ) -> List[int]:
        """Walk back from the inner CX to find the outermost qubit in the cascade."""
        # The CNOT cascade for an n-qubit Pauli exponential is:
        #   CX(q0,q1), CX(q1,q2), ..., CX(q_{n-2}, q_{n-1})
        # Walk backwards from cx_start to collect all chained CX pairs.
        qubits = list(_inst_qubit_indices(qc, instructions[cx_start]))
        j = cx_start - 1
        while j >= 0:
            inst_j = instructions[j]
            if inst_j.operation.name != "cx":
                break
            qs = _inst_qubit_indices(qc, inst_j)
            if qs[1] == qubits[0]:  # extends the chain leftward
                qubits.insert(0, qs[0])
            else:
                break
            j -= 1
        return qubits


# ═══════════════════════════════════════════════════════════════════════════════
# Core optimization: Rz merge + CNOT cancellation on instruction list
# ═══════════════════════════════════════════════════════════════════════════════

def _optimize_instruction_list(
    qc: QuantumCircuit,
) -> Tuple[QuantumCircuit, int, int]:
    """Apply Rz-merge and CNOT-cancellation directly on the instruction list.

    Both passes operate in a single left-to-right scan, tracking per-qubit
    "last live instruction" state.  Complexity: O(n_gates).

    Returns:
        (optimized_circuit, rz_merges_count, cnot_cancellations_count)
    """
    instructions = list(qc.data)
    n = len(instructions)

    # result[i] is either a CircuitInstruction or None (cancelled).
    result: List[Optional[CircuitInstruction]] = list(instructions)

    # last_live[q] = index into `result` of the most recent non-cancelled
    # instruction that touched qubit q.  -1 if none.
    last_live: Dict[int, int] = {}

    rz_merges = 0
    cnot_cancels = 0

    for i, inst in enumerate(instructions):
        name = inst.operation.name
        q_idxs = _inst_qubit_indices(qc, inst)

        if name == "rz" and len(q_idxs) == 1:
            q = q_idxs[0]
            prev = last_live.get(q, -1)
            if prev >= 0 and result[prev] is not None:
                prev_inst = result[prev]
                if prev_inst.operation.name == "rz":
                    # Merge: θ_new = θ_prev + θ_curr
                    theta_prev = float(prev_inst.operation.params[0])
                    theta_curr = float(inst.operation.params[0])
                    theta_new  = theta_prev + theta_curr

                    # Cancel entirely if the merged angle is negligibly small
                    if abs(_normalize_angle(theta_new)) < _ANGLE_TOL:
                        result[prev] = None
                        result[i]    = None
                        last_live[q] = -1
                        rz_merges += 1
                        continue
                    else:
                        # Replace previous Rz with merged Rz; cancel current slot
                        from qiskit.circuit.library import RZGate
                        merged_op = RZGate(theta_new)
                        result[prev] = CircuitInstruction(
                            operation=merged_op,
                            qubits=prev_inst.qubits,
                            clbits=prev_inst.clbits,
                        )
                        result[i] = None
                        # last_live[q] stays at prev (the merged instruction)
                        rz_merges += 1
                        continue

            last_live[q] = i

        elif name == "cx" and len(q_idxs) == 2:
            ctrl, tgt = q_idxs
            prev_ctrl = last_live.get(ctrl, -1)
            prev_tgt  = last_live.get(tgt,  -1)

            # Both qubits must have the same previous instruction, and it must
            # be a CX on the same (ctrl, tgt) pair.
            if (prev_ctrl == prev_tgt
                    and prev_ctrl >= 0
                    and result[prev_ctrl] is not None):
                prev_inst = result[prev_ctrl]
                if prev_inst.operation.name == "cx":
                    prev_q = _inst_qubit_indices(qc, prev_inst)
                    if prev_q[0] == ctrl and prev_q[1] == tgt:
                        # CX² = I  →  cancel both
                        result[prev_ctrl] = None
                        result[i]         = None
                        last_live[ctrl]   = -1
                        last_live[tgt]    = -1
                        cnot_cancels += 1
                        continue

            last_live[ctrl] = i
            last_live[tgt]  = i

        else:
            for q in q_idxs:
                last_live[q] = i

    # Rebuild circuit from surviving instructions
    optimized = QuantumCircuit(qc.num_qubits, qc.num_clbits, name=qc.name + "_fused")
    for inst in result:
        if inst is not None:
            optimized.append(inst)

    return optimized, rz_merges, cnot_cancels


# ═══════════════════════════════════════════════════════════════════════════════
# TrotterStepFusion — Qiskit TransformationPass
# ═══════════════════════════════════════════════════════════════════════════════

class TrotterStepFusion(TransformationPass):
    """Qiskit TransformationPass that fuses redundant gates in Trotter circuits.

    Two optimizations (both unitary-preserving):

    1. Rz merge — consecutive Rz(θ₁) · Rz(θ₂) on the same qubit with no
       intervening gate on that qubit → Rz(θ₁+θ₂).  If |θ₁+θ₂| < ε the
       pair is eliminated entirely.

    2. CNOT cancel — consecutive CX(a,b) · CX(a,b) with no intervening gate
       on a or b → identity (both removed).

    These two patterns arise naturally at Trotter block boundaries:
      - Adjacent XZX and YZY blocks share boundary CNOT pairs that can cancel.
      - The on-site Rz terms from n_{↑}n_{↓} decomposition produce adjacent
        single-qubit rotations that can be merged.

    Attributes:
        rz_merges:        Count of Rz pairs merged in the last run().
        cnot_cancellations: Count of CX pairs cancelled in the last run().
    """

    def __init__(self):
        super().__init__()
        self.rz_merges = 0
        self.cnot_cancellations = 0

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on a DAGCircuit.

        Converts DAG → circuit, applies _optimize_instruction_list, converts back.

        Args:
            dag: Input DAGCircuit.

        Returns:
            Optimized DAGCircuit.
        """
        # Convert DAG to a plain circuit for instruction-list manipulation
        qc = dag_to_circuit(dag)

        optimized_qc, rz_merges, cnot_cancels = _optimize_instruction_list(qc)

        self.rz_merges = rz_merges
        self.cnot_cancellations = cnot_cancels

        return circuit_to_dag(optimized_qc)


# ═══════════════════════════════════════════════════════════════════════════════
# compile_trotter — end-to-end Trotter-aware compilation
# ═══════════════════════════════════════════════════════════════════════════════

def compile_trotter(
    circuit: QuantumCircuit,
    circuit_type: str = "fermi_hubbard",
    dt: float = 0.1,
    order: int = 1,
) -> dict:
    """Compile a Trotterized circuit with domain-aware optimizations.

    Pipeline:
        1. Run TrotterBlockDetector on the abstract circuit.
        2a. If Trotter blocks found:
              Apply TrotterStepFusion (Rz merge + CNOT cancel),
              then Qiskit transpile(optimization_level=3) for hardware mapping.
        2b. If no blocks found:
              Fall back to standard quantumopt.compile().
        3. Return a stats dict.

    Args:
        circuit:      QuantumCircuit to compile (abstract gates, not yet
                      decomposed to hardware basis).
        circuit_type: Hint for the optimizer ("fermi_hubbard" or "generic").
        dt:           Trotter time step (informational; not used by the pass).
        order:        Trotter order (informational; not used by the pass).

    Returns:
        Dict with keys:
            before_gates, after_gates, gate_reduction_pct,
            before_depth, after_depth, depth_reduction_pct,
            trotter_blocks_detected, rz_merges, cnot_cancellations,
            method_used  ("trotter_fusion" | "standard")
    """
    from qiskit import transpile as qiskit_transpile

    before_gates = sum(circuit.count_ops().values())
    before_depth = circuit.depth()

    # ── Step 1: Detect Trotter blocks ─────────────────────────────────────────
    detector = TrotterBlockDetector()
    blocks = detector.detect(circuit)
    n_blocks = len(blocks)

    # ── Step 2a: Trotter-aware path ────────────────────────────────────────────
    if n_blocks > 0:
        # Apply fusion pass at abstract level
        fusion_pass = TrotterStepFusion()
        fused_dag   = fusion_pass.run(circuit_to_dag(circuit))

        rz_merges          = fusion_pass.rz_merges
        cnot_cancellations = fusion_pass.cnot_cancellations

        # If the pass made no changes, transpile the original circuit directly
        # to avoid DAG round-trip noise affecting routing decisions.
        if rz_merges == 0 and cnot_cancellations == 0:
            circuit_to_compile = circuit
        else:
            circuit_to_compile = dag_to_circuit(fused_dag)

        # Hardware transpilation after fusion (avoids H/Sdg inflation)
        try:
            from qiskit.providers.fake_provider import GenericBackendV2
            backend = GenericBackendV2(num_qubits=max(27, circuit.num_qubits + 4))
        except Exception:
            backend = None

        compiled_qc = qiskit_transpile(
            circuit_to_compile,
            backend=backend,
            optimization_level=3,
            seed_transpiler=42,
        )
        method = "trotter_fusion"

    # ── Step 2b: Standard quantumopt fallback ─────────────────────────────────
    else:
        rz_merges          = 0
        cnot_cancellations = 0

        try:
            import quantumopt
            result = quantumopt.compile(circuit, hardware="ibm_brisbane", explain=False)
            compiled_qc = result.optimized_circuit
        except Exception:
            # Last-resort: plain Qiskit transpile
            compiled_qc = qiskit_transpile(
                circuit, optimization_level=3, seed_transpiler=42
            )
        method = "standard"

    # ── Step 3: Compute stats ─────────────────────────────────────────────────
    after_ops   = compiled_qc.count_ops()
    after_gates = sum(after_ops.values())
    after_depth = compiled_qc.depth()

    gate_reduction_pct  = (before_gates - after_gates) / max(before_gates, 1) * 100
    depth_reduction_pct = (before_depth - after_depth) / max(before_depth, 1) * 100

    return {
        "before_gates":           before_gates,
        "after_gates":            after_gates,
        "gate_reduction_pct":     round(gate_reduction_pct, 2),
        "before_depth":           before_depth,
        "after_depth":            after_depth,
        "depth_reduction_pct":    round(depth_reduction_pct, 2),
        "trotter_blocks_detected": n_blocks,
        "rz_merges":              rz_merges,
        "cnot_cancellations":     cnot_cancellations,
        "method_used":            method,
    }
