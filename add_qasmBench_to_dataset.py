import os
import glob
import json
import warnings
import multiprocessing
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2

# Filter out Qiskit 2.1 deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def compile_circuit_worker(file_path, queue):
    """Parses, transpiles, and labels a single circuit."""
    try:
        circuit = QuantumCircuit.from_qasm_file(file_path)
        
        # Skip circuits that are too large
        if circuit.num_qubits > 20 or circuit.depth() == 0 or circuit.num_qubits == 0:
            queue.put(None)
            return
            
        backend = GenericBackendV2(num_qubits=27)
        
        # Step 1: Base unoptimized routing mapping
        level_0 = transpile(
            circuit, 
            backend=backend,
            optimization_level=0, 
            seed_transpiler=42
        )
        
        # Step 2: Full optimization
        level_3 = transpile(
            circuit, 
            backend=backend,
            optimization_level=3,
            seed_transpiler=42
        )
        
        # Calculate dataset-compatible metrics
        improvement_ratio = (level_0.depth() - level_3.depth()) / max(level_0.depth(), 1)
        
        # Apply the explicit user filter
        if improvement_ratio <= 0:
            queue.put(None)
            return
            
        filename = os.path.basename(file_path).replace(".qasm", "")
        
        from qiskit import qasm2
        # Save in the REQUIRED dataset JSON schema
        res_dict = {
            "algorithm": filename,
            "num_qubits": circuit.num_qubits,
            "original_qasm": qasm2.dumps(level_0),
            "original_depth": level_0.depth(),
            "original_gates": sum(level_0.count_ops().values()),
            "optimized_depth": level_3.depth(),
            "optimized_gates": sum(level_3.count_ops().values()),
            "improvement_ratio": improvement_ratio,
            "cx_count_original": level_0.count_ops().get('cx', 0),
            "cx_count_optimized": level_3.count_ops().get('cx', 0)
        }
        
        queue.put(res_dict)
    except Exception as e:
        filename = os.path.basename(file_path)
        queue.put({"error": f"❌ {filename}: {type(e).__name__}: {e}"})

def main():
    print("Loading existing dataset (dataset_clean.json)...")
    try:
        with open("dataset_clean.json", "r") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Error loading existing dataset: {e}")
        return
        
    initial_dataset_size = len(dataset)
    print(f"Loaded {initial_dataset_size} circuits from dataset_clean.json.")
    
    # Discover small and medium circuits
    qasm_files = glob.glob("QASMbench/small/**/*.qasm", recursive=True) + glob.glob("QASMbench/medium/**/*.qasm", recursive=True)
    
    if not qasm_files:
        print("No QASM files found in QASMbench folders.")
        return
        
    print(f"Found {len(qasm_files)} QASM files to evaluate.")
    
    new_circuits = []
    failed_skipped = 0
    timeout_seconds = 30
    
    print("-" * 50)
    for file_path in qasm_files:
        filename = os.path.basename(file_path)
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=compile_circuit_worker, args=(file_path, queue))
        p.start()
        p.join(timeout_seconds)
        
        if p.is_alive():
            print(f"Skipping {filename}: TIMEOUT (>30s)")
            p.terminate()
            p.join()
            failed_skipped += 1
        else:
            if not queue.empty():
                res_dict = queue.get()
                if res_dict and "error" in res_dict:
                    print(res_dict['error'])
                    failed_skipped += 1
                elif res_dict:
                    new_circuits.append(res_dict)
                    print(f"Added {filename} | {res_dict['num_qubits']}q | Improvement: {res_dict['improvement_ratio']:.2f}")
                else:
                    failed_skipped += 1
            else:
                failed_skipped += 1
                
    dataset.extend(new_circuits)
    
    print("\nSaving dataset_v2.json...")
    with open("dataset_v2.json", "w") as f:
        json.dump(dataset, f)
        
    # Calculate new average improvement ratio
    avg_improvement = sum(c.get("improvement_ratio", 0) for c in dataset) / max(len(dataset), 1)
    
    print("\n" + "=" * 50)
    print("DATASET AUGMENTATION SUMMARY")
    print("=" * 50)
    print(f"Original dataset size:      {initial_dataset_size}")
    print(f"QASMbench circuits added:   {len(new_circuits)}")
    print(f"Failed/skipped circuits:    {failed_skipped}")
    print(f"New total size:             {len(dataset)}")
    print(f"New average improvement:    {avg_improvement:.2%}")

if __name__ == "__main__":
    main()
