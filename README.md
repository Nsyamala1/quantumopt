# quantumopt

> AI-driven quantum circuit compiler using Graph Neural 
> Networks + Claude LLM — tested on real research circuits

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-purple.svg)](https://qiskit.org)

## What It Does

quantumopt takes your quantum circuit and returns an 
optimized version that runs more accurately on IBM 
quantum hardware — with a plain English explanation 
you can cite in your paper.

## Installation

```bash
pip install quantumopt
```

## Usage

```python
from qiskit import QuantumCircuit
from quantumopt import compile

# Your research circuit
qc = QuantumCircuit(5)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.ry(0.5, 0)
qc.rz(0.3, 1)

# Compile and optimize
result = compile(qc, hardware="ibm_brisbane")

# Results
print(result.depth_reduction)    # "31%"
print(result.gate_reduction)     # "32%"
print(result.explanation)        # Claude-generated report
print(result.optimized_circuit)  # Ready to run on IBM
```

## Benchmark Results

Tested on 41 real circuits from QASMbench 
(published research circuits):

| Metric                  | quantumopt | Baseline |
|-------------------------|-----------|---------|
| Avg depth reduction     | 13.2%     | 0%      |
| Avg gate reduction      | 15.2%     | 0%      |
| Circuits improved       | 34/41     | N/A     |
| Circuits made worse     | 0/41      | N/A     |
| Best result             | 89%       | N/A     |

Tested on 10,240 synthetic circuits:

| Metric                          | Result |
|---------------------------------|--------|
| GNN prediction accuracy (±10%)  | 82%    |
| GNN prediction accuracy (±20%)  | 100%   |
| Avg predicted improvement       | 64.5%  |

## Example Explanation Output

When `ANTHROPIC_API_KEY` is set, quantumopt generates 
research-quality explanations:

> "Transpilation of the target circuit for IBM Brisbane 
> hardware yielded a 31.6% reduction in circuit depth 
> (128 → 88 layers) and a 31.6% reduction in total gate 
> count (326 → 223 gates). Among the applied optimization 
> passes, merge_rotations contributed most substantially, 
> as consecutive single-qubit rotation gates collapse into 
> single parametrized operations. The resulting reduction 
> in two-qubit gate exposure is particularly consequential 
> for hardware execution, as each eliminated layer directly 
> reduces coherence-time consumption against Brisbane's 
> median T₂ timescales (~100–200 µs)."

## How It Works

1. Your circuit is encoded as a Directed Acyclic Graph
2. A trained GNN (82% accuracy) predicts optimization 
   potential and recommends optimization actions
3. Qiskit transpiler optimizes for target hardware
4. Claude generates a hardware-specific explanation 
   you can cite in your paper

## Supported Algorithms

VQE, QAOA, QFT, Grover, GHZ, Bernstein-Vazirani, 
Deutsch-Jozsa, Amplitude Estimation, Phase Estimation

## Supported Hardware

- IBM Brisbane (default)
- More backends coming

## Configuration

```bash
# Required for AI explanations
export ANTHROPIC_API_KEY=your-key-here

# Optional — for real IBM hardware execution
export IBM_QUANTUM_TOKEN=your-token-here
```

## Requirements

- Python 3.10+
- Qiskit >= 1.0.0
- PyTorch >= 2.0.0
- torch-geometric >= 2.4.0
- anthropic >= 0.20.0 (optional, for explanations)

## Citation

If you use quantumopt in your research please cite:

```
Syamala, N. (2025). quantumopt: An AI-driven quantum 
circuit compiler using Graph Neural Networks and Large 
Language Models. GitHub.
https://github.com/nsyamala1/quantumopt
```

## License

MIT License — free for research and commercial use.

## Contact

Built by Naveen Syamala
GitHub: [github.com/nsyamala1](https://github.com/nsyamala1)
Issues: [github.com/nsyamala1/quantumopt/issues](https://github.com/nsyamala1/quantumopt/issues)
