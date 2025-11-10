#!/usr/bin/env python3
"""
Generate performance charts from benchmark CSV.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "benchmark_results" / "cpu_vs_metal_quick.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        print("Please run: python3 scripts/quick_benchmark.py")
        return 1
    
    # Read data
    df = pd.read_csv(csv_path)
    
    # Calculate per-iteration times
    df['cpu_us_per_iter'] = (df['cpu_ms'] / df['iterations']) * 1000
    df['metal_us_per_iter'] = (df['metal_ms'] / df['iterations']) * 1000
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Time per iteration vs problem size
    ax = axes[0]
    ax.plot(df['items'], df['cpu_us_per_iter'], 'o-', label='CPU', linewidth=2, markersize=8)
    ax.plot(df['items'], df['metal_us_per_iter'], 's-', label='Metal (placeholder)', linewidth=2, markersize=8)
    ax.set_xlabel('Problem Size (items)', fontsize=12)
    ax.set_ylabel('Time per Evaluation (μs)', fontsize=12)
    ax.set_title('CPU vs Metal Evaluation Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot 2: Total throughput (evaluations/second)
    ax = axes[1]
    df['cpu_evals_per_sec'] = 1e6 / df['cpu_us_per_iter']  # Convert μs to evals/sec
    df['metal_evals_per_sec'] = 1e6 / df['metal_us_per_iter']
    
    ax.bar(df['items'].astype(str), df['cpu_evals_per_sec'], alpha=0.7, label='CPU')
    ax.set_xlabel('Problem Size (items)', fontsize=12)
    ax.set_ylabel('Throughput (evaluations/second)', fontsize=12)
    ax.set_title('CPU Evaluation Throughput', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # Add value labels on bars
    for i, (items, evals) in enumerate(zip(df['items'], df['cpu_evals_per_sec'])):
        ax.text(i, evals, f'{evals:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    output_path = project_root / "benchmark_results" / "performance_chart.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Chart saved to: {output_path}")
    
    # Print statistics
    print()
    print("="*70)
    print("CPU Performance Statistics")
    print("="*70)
    print()
    print(f"{'Problem Size':>15} {'μs/eval':>12} {'Evals/sec':>15}")
    print("-"*70)
    for _, row in df.iterrows():
        print(f"{row['items']:15.0f} {row['cpu_us_per_iter']:12.3f} {row['cpu_evals_per_sec']:15,.0f}")
    
    print()
    print(f"Best CPU throughput:  {df['cpu_evals_per_sec'].max():,.0f} evals/sec ({df.loc[df['cpu_evals_per_sec'].idxmax(), 'items']:.0f} items)")
    print(f"Worst CPU throughput: {df['cpu_evals_per_sec'].min():,.0f} evals/sec ({df.loc[df['cpu_evals_per_sec'].idxmin(), 'items']:.0f} items)")
    print()
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
