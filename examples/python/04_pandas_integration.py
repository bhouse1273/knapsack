#!/usr/bin/env python3
"""
Example 4: Pandas Integration

Demonstrates loading data from CSV, DataFrame manipulation, and exporting results.
Shows integration with existing data pipelines.
"""

import sys
import os

# Add build directory to path for development
build_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build')
if os.path.exists(build_dir):
    sys.path.insert(0, os.path.abspath(build_dir))

import knapsack

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas not available. Install with: pip install pandas")
    print("    This example will generate synthetic data instead.\n")


def create_sample_dataframe(num_items=100):
    """Create a sample DataFrame for demonstration"""
    
    import random
    random.seed(42)
    
    data = {
        'item_id': [f'ITEM-{i+1:04d}' for i in range(num_items)],
        'item_name': [f'Product {i+1}' for i in range(num_items)],
        'value': [random.randint(10, 1000) for _ in range(num_items)],
        'weight': [random.randint(1, 50) for _ in range(num_items)],
        'category': [random.choice(['Electronics', 'Clothing', 'Food', 'Furniture', 'Sports'])
                    for _ in range(num_items)],
        'priority': [random.choice(['High', 'Medium', 'Low']) for _ in range(num_items)],
        'quantity_available': [random.randint(1, 100) for _ in range(num_items)],
        'unit_cost': [random.randint(5, 500) for _ in range(num_items)],
    }
    
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(data)
        # Add computed columns
        df['value_weight_ratio'] = df['value'] / df['weight']
        df['margin'] = df['value'] - df['unit_cost']
        df['roi'] = (df['value'] - df['unit_cost']) / df['unit_cost']
        return df
    else:
        return data


def example_1_basic_pandas_workflow():
    """Basic workflow: load, solve, export"""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Pandas Workflow")
    print("="*80)
    
    # Create sample data
    df = create_sample_dataframe(num_items=50)
    
    if PANDAS_AVAILABLE:
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Solve
        capacity = df['weight'].sum() * 0.4  # 40% of total
        solution = knapsack.solve(
            values=df['value'].tolist(),
            weights=df['weight'].tolist(),
            capacity=capacity,
            config={"beam_width": 32, "iters": 5}
        )
        
        # Add selection column
        df['selected'] = False
        df.loc[solution.selected_indices, 'selected'] = True
        
        # Statistics
        selected_df = df[df['selected']]
        print(f"\n{'SOLUTION SUMMARY':-^80}")
        print(f"Total items: {len(df)}")
        print(f"Selected items: {len(selected_df)}")
        print(f"Total value: ${df['value'].sum():,.0f}")
        print(f"Selected value: ${selected_df['value'].sum():,.0f}")
        print(f"Total weight: {df['weight'].sum():.1f}")
        print(f"Selected weight: {selected_df['weight'].sum():.1f} / {capacity:.1f}")
        print(f"Solve time: {solution.solve_time_ms:.2f} ms")
        
        # Category breakdown
        print(f"\n{'SELECTION BY CATEGORY':-^80}")
        category_summary = df.groupby('category').agg({
            'item_id': 'count',
            'selected': 'sum',
            'value': 'sum',
            'weight': 'sum'
        }).rename(columns={'item_id': 'total_items', 'selected': 'selected_items'})
        
        category_summary['selection_rate'] = (category_summary['selected_items'] / 
                                              category_summary['total_items'] * 100)
        print(category_summary.to_string())
        
        # Export results
        output_file = '/tmp/knapsack_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Results exported to: {output_file}")
        
    else:
        print("(Pandas not available - skipping example)")


def example_2_feature_engineering():
    """Feature engineering with pandas"""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Feature Engineering with Pandas")
    print("="*80)
    
    if not PANDAS_AVAILABLE:
        print("(Pandas not available - skipping example)")
        return
    
    df = create_sample_dataframe(num_items=100)
    
    print("\nOriginal features:")
    print(f"  {list(df.columns)}")
    
    # Engineer new features
    df['profit_margin'] = (df['value'] - df['unit_cost']) / df['value']
    df['efficiency'] = df['value'] / (df['weight'] * df['unit_cost'])
    
    # Priority scoring
    priority_scores = {'High': 1.5, 'Medium': 1.0, 'Low': 0.7}
    df['priority_factor'] = df['priority'].map(priority_scores)
    
    # Adjusted value
    df['adjusted_value'] = df['value'] * df['priority_factor'] * (1 + df['profit_margin'])
    
    print("\nEngineered features:")
    print(f"  profit_margin: (value - cost) / value")
    print(f"  efficiency: value / (weight × cost)")
    print(f"  priority_factor: multiplier based on priority level")
    print(f"  adjusted_value: value × priority × (1 + margin)")
    
    # Solve with adjusted values
    capacity = df['weight'].sum() * 0.4
    solution = knapsack.solve(
        values=df['adjusted_value'].tolist(),
        weights=df['weight'].tolist(),
        capacity=capacity,
        config={"beam_width": 32, "iters": 5}
    )
    
    df['selected'] = False
    df.loc[solution.selected_indices, 'selected'] = True
    
    selected_df = df[df['selected']]
    
    print(f"\n{'SOLUTION WITH ADJUSTED VALUES':-^80}")
    print(f"Selected: {len(selected_df)} items")
    print(f"Avg profit margin: {selected_df['profit_margin'].mean():.1%}")
    print(f"Avg efficiency: {selected_df['efficiency'].mean():.2f}")
    
    # Priority breakdown
    priority_dist = selected_df['priority'].value_counts()
    print(f"\nPriority distribution:")
    for priority, count in priority_dist.items():
        pct = count / len(selected_df) * 100
        print(f"  {priority}: {count} ({pct:.1f}%)")


def example_3_filtering_and_preprocessing():
    """Data filtering and preprocessing"""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Filtering and Preprocessing")
    print("="*80)
    
    if not PANDAS_AVAILABLE:
        print("(Pandas not available - skipping example)")
        return
    
    df = create_sample_dataframe(num_items=100)
    
    print(f"\nOriginal dataset: {len(df)} items")
    
    # Apply filters
    print("\nApplying filters:")
    
    # Filter 1: Minimum value threshold
    min_value = 100
    df_filtered = df[df['value'] >= min_value]
    print(f"  1. Value >= ${min_value}: {len(df_filtered)} items remain")
    
    # Filter 2: Exclude low margin items
    df_filtered = df_filtered[df_filtered['margin'] > 0]
    print(f"  2. Positive margin: {len(df_filtered)} items remain")
    
    # Filter 3: Category filter
    allowed_categories = ['Electronics', 'Sports']
    df_filtered = df_filtered[df_filtered['category'].isin(allowed_categories)]
    print(f"  3. Categories {allowed_categories}: {len(df_filtered)} items remain")
    
    # Solve on filtered dataset
    capacity = df_filtered['weight'].sum() * 0.5
    solution = knapsack.solve(
        values=df_filtered['value'].tolist(),
        weights=df_filtered['weight'].tolist(),
        capacity=capacity,
        config={"beam_width": 32, "iters": 5}
    )
    
    print(f"\n{'FILTERED SOLUTION':-^80}")
    print(f"Capacity: {capacity:.1f}")
    print(f"Selected from filtered set: {len(solution.selected_indices)}")
    print(f"Best value: ${solution.best_value:,.0f}")
    
    # Compare with unfiltered
    solution_full = knapsack.solve(
        values=df['value'].tolist(),
        weights=df['weight'].tolist(),
        capacity=capacity,
        config={"beam_width": 32, "iters": 5}
    )
    
    print(f"\nComparison:")
    print(f"  Filtered universe: ${solution.best_value:,.0f}")
    print(f"  Full universe: ${solution_full.best_value:,.0f}")
    print(f"  Difference: ${(solution_full.best_value - solution.best_value):,.0f} "
          f"({(solution_full.best_value - solution.best_value)/solution.best_value*100:.1f}%)")


def example_4_multi_scenario_analysis():
    """Run multiple scenarios and compare"""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Multi-Scenario Analysis")
    print("="*80)
    
    if not PANDAS_AVAILABLE:
        print("(Pandas not available - skipping example)")
        return
    
    df = create_sample_dataframe(num_items=50)
    
    # Define scenarios
    scenarios = [
        {"name": "Conservative", "capacity_pct": 0.3, "beam_width": 16},
        {"name": "Balanced", "capacity_pct": 0.4, "beam_width": 32},
        {"name": "Aggressive", "capacity_pct": 0.5, "beam_width": 64},
    ]
    
    results = []
    
    for scenario in scenarios:
        capacity = df['weight'].sum() * scenario['capacity_pct']
        solution = knapsack.solve(
            values=df['value'].tolist(),
            weights=df['weight'].tolist(),
            capacity=capacity,
            config={"beam_width": scenario['beam_width'], "iters": 5}
        )
        
        selected_weight = sum(df.loc[solution.selected_indices, 'weight'])
        
        results.append({
            'scenario': scenario['name'],
            'capacity': capacity,
            'items_selected': len(solution.selected_indices),
            'total_value': solution.best_value,
            'total_weight': selected_weight,
            'utilization': selected_weight / capacity * 100,
            'value_per_weight': solution.best_value / selected_weight if selected_weight > 0 else 0,
            'solve_time_ms': solution.solve_time_ms
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'SCENARIO COMPARISON':-^80}")
    print(results_df.to_string(index=False))
    
    # Export comparison
    output_file = '/tmp/scenario_comparison.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Scenario comparison exported to: {output_file}")


def example_5_data_quality_checks():
    """Data quality validation before solving"""
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Data Quality Checks")
    print("="*80)
    
    if not PANDAS_AVAILABLE:
        print("(Pandas not available - skipping example)")
        return
    
    df = create_sample_dataframe(num_items=50)
    
    print("\nData Quality Checks:")
    
    # Check 1: Missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"  ✗ Missing values found:")
        print(f"    {missing[missing > 0].to_dict()}")
    else:
        print(f"  ✓ No missing values")
    
    # Check 2: Negative values
    negative_values = (df['value'] < 0).sum()
    negative_weights = (df['weight'] <= 0).sum()
    if negative_values > 0 or negative_weights > 0:
        print(f"  ✗ Invalid values: {negative_values} negative values, {negative_weights} non-positive weights")
    else:
        print(f"  ✓ All values positive")
    
    # Check 3: Duplicates
    duplicates = df.duplicated(subset=['item_id']).sum()
    if duplicates > 0:
        print(f"  ✗ Duplicate item_ids: {duplicates}")
    else:
        print(f"  ✓ No duplicate item_ids")
    
    # Check 4: Data types
    print(f"  ✓ Data types:")
    print(f"    value: {df['value'].dtype}")
    print(f"    weight: {df['weight'].dtype}")
    
    # Check 5: Value ranges
    print(f"  ✓ Value ranges:")
    print(f"    value: {df['value'].min():.2f} to {df['value'].max():.2f}")
    print(f"    weight: {df['weight'].min():.2f} to {df['weight'].max():.2f}")
    
    # Check 6: Correlations
    corr = df[['value', 'weight']].corr().iloc[0, 1]
    print(f"  ℹ  Value-weight correlation: {corr:.3f}")
    
    print("\n✅ Data quality checks complete!")


def main():
    """Run all pandas integration examples"""
    
    print("╔" + "═"*78 + "╗")
    print("║" + " PANDAS INTEGRATION EXAMPLES ".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    
    if not PANDAS_AVAILABLE:
        print("\n⚠️  pandas is not installed. Some examples will be skipped.")
        print("   Install with: pip install pandas\n")
    
    example_1_basic_pandas_workflow()
    example_2_feature_engineering()
    example_3_filtering_and_preprocessing()
    example_4_multi_scenario_analysis()
    example_5_data_quality_checks()
    
    print("\n" + "="*80)
    print("✅ All pandas integration examples completed!")
    print("="*80)
    
    if PANDAS_AVAILABLE:
        print("\nGenerated files:")
        print("  • /tmp/knapsack_results.csv")
        print("  • /tmp/scenario_comparison.csv")


if __name__ == "__main__":
    main()
