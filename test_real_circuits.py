import os
import glob
import json
import warnings
import multiprocessing
from qiskit import QuantumCircuit
from quantumopt import compile

# Filter out Qiskit 2.1 deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def compile_circuit_worker(file_path, queue):
    """Compiles a single circuit, intended to be run in a separate process."""
    try:
        qc = QuantumCircuit.from_qasm_file(file_path)
        
        # Skip empty circuits
        if qc.depth() == 0 or qc.num_qubits == 0:
            queue.put(None)
            return
            
        # Compile bypassing the LLM explanation for speed
        result = compile(qc, hardware="ibm_brisbane", explain=False)
        
        orig_depth = result.original_stats.get('depth', 0)
        opt_depth = result.optimized_stats.get('depth', 0)
        
        # Calculate numerical reductions for summary
        depth_red_float = (orig_depth - opt_depth) / max(orig_depth, 1) * 100
        
        orig_gates = result.original_stats.get('gate_count', 0)
        opt_gates = result.optimized_stats.get('gate_count', 0)
        gate_red_float = (orig_gates - opt_gates) / max(orig_gates, 1) * 100
        
        queue.put({
            "filename": os.path.basename(file_path),
            "num_qubits": qc.num_qubits,
            "original_depth": orig_depth,
            "optimized_depth": opt_depth,
            "depth_reduction": result.depth_reduction,
            "depth_reduction_float": depth_red_float,
            "gate_reduction": result.gate_reduction,
            "gate_reduction_float": gate_red_float,
        })
    except Exception:
        queue.put(None)

def main():
    qasm_files = glob.glob("QASMbench/medium/**/*.qasm", recursive=True)
    
    if not qasm_files:
        print("No QASM files found in QASMbench/medium/")
        return
        
    print(f"Found {len(qasm_files)} circuits in QASMbench/medium/")
    print(f"{'Filename':<30} | {'Qubits':<8} | {'Orig Depth':<12} | {'Opt Depth':<11} | {'Depth Red.':<12} | {'Gate Red.':<11}")
    print("-" * 96)
    
    results = []
    timeout_seconds = 60
    
    for file_path in qasm_files:
        filename = os.path.basename(file_path)
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=compile_circuit_worker, args=(file_path, queue))
        p.start()
        p.join(timeout_seconds)
        
        if p.is_alive():
            print(f"{filename:<30} | {'TIMEOUT (>60s)':<63}")
            p.terminate()
            p.join()
        else:
            if not queue.empty():
                res_dict = queue.get()
                if res_dict:
                    results.append(res_dict)
                    print(f"{filename:<30} | {res_dict['num_qubits']:<8} | {res_dict['original_depth']:<12} | {res_dict['optimized_depth']:<11} | {res_dict['depth_reduction']:<12} | {res_dict['gate_reduction']:<11}")
                else:
                    print(f"{filename:<30} | {'FAILED':<63}")
            else:
                print(f"{filename:<30} | {'FAILED (Crash)':<63}")
            
    if not results:
        print("No circuits were successfully compiled.")
        return
        
    # Summary stats
    avg_depth_reduction = sum(r['depth_reduction_float'] for r in results) / len(results)
    avg_gate_reduction = sum(r['gate_reduction_float'] for r in results) / len(results)
    
    improved_count = sum(1 for r in results if r['depth_reduction_float'] > 0)
    worse_count = sum(1 for r in results if r['depth_reduction_float'] < 0)
    neutral_count = sum(1 for r in results if r['depth_reduction_float'] == 0)
    
    best_circuit = max(results, key=lambda x: x['depth_reduction_float'])
    
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Total Successful Circuits: {len(results)}")
    print(f"Average Depth Reduction:   {avg_depth_reduction:.1f}%")
    print(f"Average Gate Reduction:    {avg_gate_reduction:.1f}%")
    print(f"Circuits Improved:         {improved_count}")
    print(f"Circuits Got Worse:        {worse_count}")
    print(f"Circuits Neutral:          {neutral_count}")
    print(f"Best Performing Circuit:   {best_circuit['filename']} ({best_circuit['depth_reduction']} depth reduction)")
    
    # Save to JSON
    out_file = "benchmark_real_circuits.json"
    with open(out_file, "w") as f:
        json.dump({
            "summary": {
                "total_circuits": len(results),
                "avg_depth_reduction": avg_depth_reduction,
                "avg_gate_reduction": avg_gate_reduction,
                "improved_count": improved_count,
                "worse_count": worse_count,
                "neutral_count": neutral_count,
                "best_circuit": best_circuit['filename']
            },
            "results": results
        }, f, indent=2)
        
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
