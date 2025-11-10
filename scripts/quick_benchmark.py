#!/usr/bin/env python3
"""
Quick performance benchmark: CPU vs Metal using the benchmark tool.

Creates configs of varying sizes and measures performance.
"""

import json
import subprocess
import sys
from pathlib import Path

def create_config(n_items: int, output_path: Path) -> None:
    """Create a benchmark config file."""
    config = {
        "version": 2,
        "mode": "select",
        "random_seed": 42,
        "items": {
            "count": n_items,
            "attributes": {
                "value": [float(i * 10) for i in range(1, n_items + 1)],
                "weight": [float(i * 5) for i in range(1, n_items + 1)]
            }
        },
        "blocks": [
            {"name": "all", "start": 0, "count": n_items}
        ],
        "objective": [
            {"attr": "value", "weight": 1.0}
        ],
        "constraints": [
            {
                "kind": "capacity",
                "attr": "weight",
                "limit": float(n_items * 5 * 0.6),  # 60% capacity
                "soft": False
            }
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

def run_benchmark(exe_path: Path, config_path: Path, iterations: int) -> dict:
    """Run benchmark and parse results."""
    result = subprocess.run(
        [str(exe_path), str(config_path), str(iterations)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    # Parse: items,iterations,cpu_ms,metal_ms,speedup,cpu_obj,metal_obj
    parts = result.stdout.strip().split(',')
    return {
        'items': int(parts[0]),
        'iterations': int(parts[1]),
        'cpu_ms': float(parts[2]),
        'metal_ms': float(parts[3]),
        'speedup': float(parts[4]),
        'cpu_obj': float(parts[5]),
        'metal_obj': float(parts[6])
    }

def main():
    project_root = Path(__file__).parent.parent
    exe_path = project_root / "build" / "benchmark_cpu_vs_metal"
    
    if not exe_path.exists():
        print(f"Error: {exe_path} not found")
        print("Please run: make -C build benchmark_cpu_vs_metal")
        return 1
    
    print("="*70)
    print("CPU vs Metal Performance Benchmark")
    print("="*70)
    print()
    
    # Test configurations
    tests = [
        (10, 2000),
        (50, 1000),
        (100, 500),
        (500, 100),
        (1000, 50),
        (5000, 10),
    ]
    
    results = []
    
    for n_items, iterations in tests:
        print(f"Testing {n_items:5d} items ({iterations:4d} iterations)... ", end='', flush=True)
        
        config_path = project_root / "build" / f"bench_{n_items}.json"
        create_config(n_items, config_path)
        
        result = run_benchmark(exe_path, config_path, iterations)
        if result:
            results.append(result)
            print(f"CPU: {result['cpu_ms']:8.3f}ms  Metal: {result['metal_ms']:8.3f}ms  Speedup: {result['speedup']:6.1f}x")
        else:
            print("FAILED")
    
    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print(f"{'Items':>6} {'Iterations':>10} {'CPU (ms)':>10} {'Metal (ms)':>12} {'Speedup':>8}")
    print("-"*70)
    
    for r in results:
        print(f"{r['items']:6d} {r['iterations']:10d} {r['cpu_ms']:10.3f} {r['metal_ms']:12.3f} {r['speedup']:8.1f}x")
    
    # Save to CSV
    output_file = project_root / "benchmark_results" / "cpu_vs_metal_quick.csv"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("items,iterations,cpu_ms,metal_ms,speedup,cpu_objective,metal_objective\n")
        for r in results:
            f.write(f"{r['items']},{r['iterations']},{r['cpu_ms']:.3f},{r['metal_ms']:.3f},"
                   f"{r['speedup']:.3f},{r['cpu_obj']:.2f},{r['metal_obj']:.2f}\n")
    
    print()
    print(f"✅ Results saved to: {output_file}")
    print()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
