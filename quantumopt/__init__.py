"""
QuantumOpt — AI-Driven Quantum Compiler
========================================

A pip-installable Python package that uses Graph Neural Networks (GNN)
and Claude LLM API to optimize quantum circuits for IBM Quantum hardware.

Target users: University quantum research labs running VQE, QAOA, QFT,
and Grover circuits on IBM Quantum hardware.

Usage:
    from quantumopt import compile

    result = compile(
        circuit=my_vqe_circuit,
        hardware="ibm_brisbane",
        priority="fidelity"
    )

    print(result.optimized_circuit)
    print(result.depth_reduction)
    print(result.explanation)
    print(result.benchmark)
"""

__version__ = "0.1.0"
__author__ = "QuantumOpt Team"

from quantumopt.compiler import compile, CompileResult

__all__ = ["compile", "CompileResult", "__version__"]
