#!/usr/bin/env python3
"""
Analyze benchmark CSV results (no external dependencies).
"""

import csv
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "benchmark_results" / "cpu_vs_metal_quick.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return 1
    
    # Read CSV
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print("="*70)
    print("CPU vs Metal Performance Analysis")
    print("="*70)
    print()
    
    # Calculate metrics
    results = []
    for row in rows:
        items = int(row['items'])
        iterations = int(row['iterations'])
        cpu_ms = float(row['cpu_ms'])
        metal_ms = float(row['metal_ms'])
        cpu_obj = float(row['cpu_objective'])
        metal_obj = float(row['metal_objective'])
        
        cpu_us_per = (cpu_ms / iterations) * 1000  # Convert ms to μs
        metal_us_per = (metal_ms / iterations) * 1000 if metal_ms > 0 else 0
        cpu_evals_per_sec = 1_000_000 / cpu_us_per  # Convert μs to evals/sec
        
        results.append({
            'items': items,
            'iterations': iterations,
            'cpu_us_per': cpu_us_per,
            'metal_us_per': metal_us_per,
            'cpu_evals_per_sec': cpu_evals_per_sec,
            'cpu_obj': cpu_obj,
            'metal_obj': metal_obj
        })
    
    # Print table
    print(f"{'Items':>7} {'Iterations':>11} {'CPU μs/eval':>12} {'CPU evals/sec':>15} {'Objective':>12}")
    print("-"*70)
    
    for r in results:
        print(f"{r['items']:7d} {r['iterations']:11d} {r['cpu_us_per']:12.3f} {r['cpu_evals_per_sec']:15,.0f} {r['cpu_obj']:12.0f}")
    
    # Statistics
    print()
    print("="*70)
    print("Summary Statistics")
    print("="*70)
    print()
    
    cpu_times = [r['cpu_us_per'] for r in results]
    cpu_throughputs = [r['cpu_evals_per_sec'] for r in results]
    
    print(f"CPU Evaluation Time:")
    print(f"  Min:     {min(cpu_times):8.3f} μs  ({results[cpu_times.index(min(cpu_times))]['items']} items)")
    print(f"  Max:     {max(cpu_times):8.3f} μs  ({results[cpu_times.index(max(cpu_times))]['items']} items)")
    print(f"  Average: {sum(cpu_times)/len(cpu_times):8.3f} μs")
    print()
    
    print(f"CPU Throughput:")
    print(f"  Max: {max(cpu_throughputs):12,.0f} evals/sec  ({results[cpu_throughputs.index(max(cpu_throughputs))]['items']} items)")
    print(f"  Min: {min(cpu_throughputs):12,.0f} evals/sec  ({results[cpu_throughputs.index(min(cpu_throughputs))]['items']} items)")
    print(f"  Avg: {sum(cpu_throughputs)/len(cpu_throughputs):12,.0f} evals/sec")
    print()
    
    # Correctness check
    print("Correctness Validation:")
    all_correct = all(abs(r['cpu_obj'] - r['metal_obj']) < 0.01 for r in results)
    print(f"  CPU vs Metal objectives match: {'✅ YES' if all_correct else '❌ NO'}")
    print()
    
    # Scaling analysis
    print("Scaling Analysis:")
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]
        size_ratio = curr['items'] / prev['items']
        time_ratio = curr['cpu_us_per'] / prev['cpu_us_per']
        print(f"  {prev['items']:5d} → {curr['items']:5d} items ({size_ratio:4.1f}x): Time increased {time_ratio:5.2f}x")
    
    print()
    print("="*70)
    print()
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
