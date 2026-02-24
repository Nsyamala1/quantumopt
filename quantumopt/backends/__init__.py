"""
quantumopt.backends — Quantum hardware backend integration.

Provides compilation and optimization for specific quantum hardware
platforms, starting with IBM Quantum.
"""

from quantumopt.backends.ibm_backend import compile_for_ibm

__all__ = ["compile_for_ibm"]
