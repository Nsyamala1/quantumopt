#!/usr/bin/env python3
"""
validate_dataset.py — Validate and summarize the generated quantum circuit dataset.

Checks:
  - JSON loads correctly
  - Required fields present in every record
  - Value ranges are sane
  - Prints per-algorithm stats and overall summary

Usage:
    python validate_dataset.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

DATASET_FILE = Path("dataset.json")

REQUIRED_FIELDS = {
    "algorithm": str,
    "num_qubits": int,
    "original_qasm": str,
    "original_depth": int,
    "original_gates": int,
    "optimized_depth": int,
    "optimized_gates": int,
    "improvement_ratio": (int, float),
    "cx_count_original": int,
    "cx_count_optimized": int,
}


def validate():
    if not DATASET_FILE.exists():
        print(f"❌ {DATASET_FILE} not found. Run generate_dataset.py first.")
        sys.exit(1)

    print(f"📂 Loading {DATASET_FILE}...")
    with open(DATASET_FILE) as f:
        dataset = json.load(f)

    total = len(dataset)
    print(f"📊 Total records: {total}\n")

    if total == 0:
        print("❌ Dataset is empty!")
        sys.exit(1)

    # ── Field validation ───────────────────────────────────────────────
    errors = []
    warnings = []
    stats = defaultdict(lambda: {
        "count": 0,
        "qubits": [],
        "improvements": [],
        "depths_original": [],
        "depths_optimized": [],
        "cx_original": [],
        "cx_optimized": [],
    })

    for i, record in enumerate(dataset):
        # Check required fields
        for field, expected_type in REQUIRED_FIELDS.items():
            if field not in record:
                errors.append(f"Record {i}: missing field '{field}'")
            elif not isinstance(record[field], expected_type):
                errors.append(
                    f"Record {i}: '{field}' expected {expected_type}, "
                    f"got {type(record[field]).__name__}"
                )

        if errors and len(errors) > 20:
            errors.append(f"... (stopping after 20 errors)")
            break

        # Sanity checks
        algo = record.get("algorithm", "unknown")
        nq = record.get("num_qubits", 0)
        imp = record.get("improvement_ratio", 0)
        od = record.get("original_depth", 0)
        optd = record.get("optimized_depth", 0)

        if nq < 1 or nq > 200:
            warnings.append(f"Record {i}: unusual qubit count {nq}")
        if imp < -1.0 or imp > 1.0:
            warnings.append(f"Record {i}: improvement_ratio {imp} out of [-1, 1]")
        if od < 0 or optd < 0:
            warnings.append(f"Record {i}: negative depth values")

        # Accumulate stats
        s = stats[algo]
        s["count"] += 1
        s["qubits"].append(nq)
        s["improvements"].append(imp)
        s["depths_original"].append(od)
        s["depths_optimized"].append(optd)
        s["cx_original"].append(record.get("cx_count_original", 0))
        s["cx_optimized"].append(record.get("cx_count_optimized", 0))

    # ── Report errors ──────────────────────────────────────────────────
    if errors:
        print(f"❌ {len(errors)} validation errors:")
        for e in errors[:20]:
            print(f"   • {e}")
        print()
    else:
        print("✅ All records have valid structure.\n")

    if warnings:
        print(f"⚠️  {len(warnings)} warnings:")
        for w in warnings[:10]:
            print(f"   • {w}")
        print()

    # ── Per-algorithm summary ──────────────────────────────────────────
    sorted_algos = sorted(stats.items(), key=lambda x: -x[1]["count"])

    print("=" * 80)
    print(f"{'Algorithm':<22} {'Count':>7} {'Qubits':>12} "
          f"{'Avg Depth':>11} {'Avg Opt':>9} {'Avg Imp':>9} {'Avg CX→':>10}")
    print("-" * 80)

    for algo, s in sorted_algos:
        n = s["count"]
        q_min, q_max = min(s["qubits"]), max(s["qubits"])
        avg_depth = sum(s["depths_original"]) / n
        avg_opt = sum(s["depths_optimized"]) / n
        avg_imp = sum(s["improvements"]) / n
        cx_orig = sum(s["cx_original"]) / n
        cx_opt = sum(s["cx_optimized"]) / n

        print(
            f"  {algo:<20} {n:>7} {q_min:>4}–{q_max:<4} "
            f"{avg_depth:>11.1f} {avg_opt:>9.1f} {avg_imp:>8.1%} "
            f"{cx_orig:.0f}→{cx_opt:.0f}"
        )

    print("-" * 80)

    # Overall
    all_imp = [r.get("improvement_ratio", 0) for r in dataset]
    avg_all = sum(all_imp) / len(all_imp)
    all_q = [r.get("num_qubits", 0) for r in dataset]
    print(
        f"  {'TOTAL':<20} {total:>7} {min(all_q):>4}–{max(all_q):<4} "
        f"{'':>11} {'':>9} {avg_all:>8.1%}"
    )
    print("=" * 80)

    # ── Final verdict ──────────────────────────────────────────────────
    print()
    if total >= 15_000:
        print(f"🎉 Target reached! {total} ≥ 15,000 circuits.")
    else:
        print(f"⚠️  Below target: {total} < 15,000 circuits.")
        print(f"   Consider increasing RANDOM_COUNT in generate_dataset.py.")

    if not errors:
        print("✅ Dataset is valid and ready for GNN training.")
    else:
        print("❌ Fix validation errors before training.")

    # ── QASM spot-check ────────────────────────────────────────────────
    print(f"\n📝 QASM spot-check (first record):")
    qasm = dataset[0].get("original_qasm", "")
    preview = qasm[:200] + ("..." if len(qasm) > 200 else "")
    print(f"   {preview}")


if __name__ == "__main__":
    validate()
