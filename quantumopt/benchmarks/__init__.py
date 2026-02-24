"""
quantumopt.benchmarks — Benchmarking module.

Compares QuantumOpt compilation against Qiskit's built-in transpiler
to measure depth reduction, gate count reduction, and runtime.
"""

from quantumopt.benchmarks.compare import run_benchmark

__all__ = ["run_benchmark"]
