# QuantumOpt — AI-Driven Quantum Circuit Compiler

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**QuantumOpt** uses Graph Neural Networks (GNN) and Claude LLM to optimize quantum circuits for IBM Quantum hardware — outperforming Qiskit's built-in transpiler.

Built for **university quantum research labs** running VQE, QAOA, QFT, and Grover circuits daily.

---

## Quick Start

```bash
pip install -e .
```

```python
from quantumopt import compile

result = compile(
    circuit=my_vqe_circuit,
    hardware="ibm_brisbane",
    priority="fidelity"
)

print(result.optimized_circuit)    # hardware-ready optimized circuit
print(result.depth_reduction)      # e.g. "31%"
print(result.explanation)          # Claude plain-English report
print(result.benchmark)            # comparison vs Qiskit transpiler
```

## Setup

1. **Clone and install**:
   ```bash
   git clone <repo-url>
   cd quantumopt
   pip install -e .
   ```

2. **Configure API keys** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your keys
   ```

   - `ANTHROPIC_API_KEY` — Required for Claude explanations (compile still works without it)
   - `IBM_QUANTUM_TOKEN` — Optional, for real IBM hardware (uses FakeBrisbane simulator otherwise)

## Architecture

```
Circuit → DAG → PyG Graph → GNN Prediction → IBM Backend Compilation → Claude Explanation
```

| Component | Purpose |
|-----------|---------|
| `graph/encoder.py` | Circuit → DAG → PyG graph conversion |
| `graph/features.py` | 20-dim gate feature vectors |
| `models/gnn.py` | 3-layer GCN for optimization scoring |
| `backends/ibm_backend.py` | Qiskit transpiler + optimization passes |
| `llm/explainer.py` | Claude API for plain-English explanations |
| `compiler.py` | Orchestrates the full pipeline |

## Supported Circuit Types

- **VQE** — Variational Quantum Eigensolver (quantum chemistry)
- **QAOA** — Quantum Approximate Optimization Algorithm
- **QFT** — Quantum Fourier Transform
- **Grover** — Quantum search algorithm

## Running Tests

```bash
pytest tests/ -v
```

## Training the GNN

```bash
# Generate training dataset
python -c "from quantumopt.data import generate_dataset; generate_dataset()"

# Train the model
python train.py
```

## License

MIT License — see [LICENSE](LICENSE) for details.
