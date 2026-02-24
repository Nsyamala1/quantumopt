import json
import random
from qiskit import QuantumCircuit
from quantumopt import compile
import warnings

# Filter out Qiskit 2.1 deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("\n🚀 Running QuantumOpt Smoke Test (Real Dataset Circuit)...\n")

# Load a real circuit from our training dataset
with open('dataset_clean.json') as f:
    dataset = json.load(f)

# Find a circuit with good optimization potential
# Pick one where improvement_ratio > 0.5 and depth > 30
good_circuits = [
    c for c in dataset 
    if c['improvement_ratio'] > 0.5 
    and c['original_depth'] > 100
    and c['num_qubits'] >= 5
    and c['num_qubits'] <= 12
]

# Pick a random good circuit
sample = random.choice(good_circuits)
print(f"Algorithm: {sample['algorithm']}")
print(f"Qubits: {sample['num_qubits']}")
print(f"Expected improvement (from fake transpiler): {sample['improvement_ratio']*100:.1f}%")

# Load circuit from QASM
qc = QuantumCircuit.from_qasm_str(sample['original_qasm'])
print(f"Circuit depth: {qc.depth()}, Gates: {qc.count_ops().get('cx', 0) + qc.count_ops().get('h', 0) + sum(qc.count_ops().values())} (logical count)")
print("Compiling for ibm_brisbane...\n")

# Run through our compiler
result = compile(qc, hardware="ibm_brisbane")

print(f"\nCompilation Complete!")
print(f"Original depth:   {result.original_stats['depth']}")
print(f"Optimized depth:  {result.optimized_stats['depth']}")
print(f"Depth reduction:  {result.depth_reduction}")
print(f"Gate reduction:   {result.gate_reduction}")
print(f"GNN Predicted:    {result.gnn_prediction:.1%}")
print(f"Explanation:      {result.explanation}")
print(f"Execution Time:   {result.compile_time:.2f}s\n")
