#!/usr/bin/env python3
"""
Convert OR-Library Multiple Knapsack Problem (MKP) format to v2 Config JSON.

OR-Library MKP Format:
    Line 1: n m (number of items, number of knapsacks)
    Lines 2 to n+1: profit values (one per line)
    Lines n+2 to n+m+1: knapsack capacities (one per line)
    Lines n+m+2 onward: resource consumption matrix (m rows of n values each)

Example usage:
    python convert_orlib_mkp.py data/benchmarks/or-library/mknap1.txt > config.json
    
    # Or extract just one problem from a file with multiple problems
    python convert_orlib_mkp.py data/benchmarks/or-library/mknap1.txt --problem 1 > problem1.json
"""

import json
import sys
import argparse
from typing import Dict, List, Any


def parse_orlib_mkp_file(filename: str, problem_index: int = 0) -> Dict[str, Any]:
    """
    Parse OR-Library MKP format file.
    
    Args:
        filename: Path to OR-Library format file
        problem_index: Which problem to extract (0-based, for files with multiple problems)
    
    Returns:
        v2 Config dictionary
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    # Parse problem(s) from file
    problems = []
    idx = 0
    
    while idx < len(lines):
        # First line: n m
        parts = lines[idx].split()
        if len(parts) < 2:
            idx += 1
            continue
            
        n = int(parts[0])  # number of items
        m = int(parts[1])  # number of knapsacks
        idx += 1
        
        # Next n values: profits
        profits = []
        for i in range(n):
            if idx >= len(lines):
                break
            profits.extend([float(x) for x in lines[idx].split()])
            idx += 1
        
        # Next m values: capacities
        capacities = []
        for i in range(m):
            if idx >= len(lines):
                break
            capacities.extend([float(x) for x in lines[idx].split()])
            idx += 1
        
        # Next m*n values: resource consumption matrix (weights)
        # Matrix is m rows x n columns (one row per knapsack)
        weight_matrix = []
        for i in range(m):
            if idx >= len(lines):
                break
            row = []
            while len(row) < n and idx < len(lines):
                row.extend([float(x) for x in lines[idx].split()])
                idx += 1
            weight_matrix.append(row[:n])
        
        # For single knapsack representation, we use the first knapsack's weights
        # For assign mode, we need per-knapsack constraints
        problems.append({
            'n': n,
            'm': m,
            'profits': profits[:n],
            'capacities': capacities[:m],
            'weight_matrix': weight_matrix
        })
    
    if problem_index >= len(problems):
        raise ValueError(f"Problem index {problem_index} out of range (found {len(problems)} problems)")
    
    prob = problems[problem_index]
    return create_v2_config(prob)


def create_v2_config(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert parsed problem to v2::Config format.
    
    For Multiple Knapsack Problem, we use assign mode where each item
    is assigned to exactly one knapsack (or not selected).
    """
    n = problem['n']
    m = problem['m']
    profits = problem['profits']
    capacities = problem['capacities']
    weight_matrix = problem['weight_matrix']
    
    # For multi-knapsack, each item has m different weight values
    # (one for each knapsack it could be assigned to)
    # We'll encode this as a single "weight" attribute and handle
    # capacity constraints per knapsack in assign mode
    
    # Average weight across knapsacks (or use first knapsack's weights)
    weights = weight_matrix[0] if weight_matrix else [0.0] * n
    
    config = {
        "mode": "assign",
        "items": {
            "count": n,
            "attributes": {
                "value": profits,
                "weight": weights
            }
        },
        "knapsack": {
            "K": m,
            "capacities": capacities,
            "capacity_attr": "weight"
        },
        "constraints": [],
        "objective": [
            {
                "attr": "value",
                "weight": 1.0
            }
        ],
        "solver": {
            "beam_width": 32,
            "max_iterations": 3,
            "seed": 42
        }
    }
    
    # Add capacity constraint for each knapsack
    # Note: In true MKP, each knapsack has different resource requirements per item
    # This is a simplified encoding - for full MKP support, you may need
    # to extend the v2 format or use item-specific attributes per knapsack
    for k in range(m):
        config["constraints"].append({
            "kind": "capacity",
            "attr": "weight",
            "limit": capacities[k],
            "soft": False
        })
    
    return config


def create_select_mode_config(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Alternative: Convert to select mode (single knapsack).
    Uses the first knapsack's capacity and weights.
    """
    n = problem['n']
    profits = problem['profits']
    capacity = problem['capacities'][0]
    weights = problem['weight_matrix'][0] if problem['weight_matrix'] else [0.0] * n
    
    config = {
        "mode": "select",
        "items": {
            "count": n,
            "attributes": {
                "value": profits,
                "weight": weights
            }
        },
        "constraints": [
            {
                "kind": "capacity",
                "attr": "weight",
                "limit": capacity,
                "soft": False
            }
        ],
        "objective": [
            {
                "attr": "value",
                "weight": 1.0
            }
        ],
        "solver": {
            "beam_width": 32,
            "max_iterations": 3,
            "seed": 42
        }
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Convert OR-Library MKP format to v2 Config JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input_file', help='OR-Library format file')
    parser.add_argument('--problem', type=int, default=0,
                       help='Problem index to extract (0-based, default: 0)')
    parser.add_argument('--mode', choices=['assign', 'select'], default='assign',
                       help='Config mode (assign=multi-knapsack, select=single knapsack)')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--pretty', action='store_true',
                       help='Pretty-print JSON output')
    
    args = parser.parse_args()
    
    try:
        problem = parse_orlib_mkp_file(args.input_file, args.problem)[0]
        
        if args.mode == 'select':
            config = create_select_mode_config(problem)
        else:
            config = create_v2_config(problem)
        
        # Output
        indent = 2 if args.pretty else None
        json_str = json.dumps(config, indent=indent)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_str)
                f.write('\n')
            print(f"Wrote config to {args.output}", file=sys.stderr)
        else:
            print(json_str)
        
        # Print summary to stderr
        print(f"\nConverted problem {args.problem}:", file=sys.stderr)
        print(f"  Items: {config['items']['count']}", file=sys.stderr)
        if args.mode == 'assign':
            print(f"  Knapsacks: {config['knapsack']['K']}", file=sys.stderr)
            print(f"  Capacities: {config['knapsack']['capacities']}", file=sys.stderr)
        else:
            print(f"  Capacity: {config['constraints'][0]['limit']}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
