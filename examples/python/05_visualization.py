#!/usr/bin/env python3
"""
Example 5: Visualization and Analysis

Demonstrates plotting results using matplotlib for visual analysis.
Includes scatter plots, Pareto frontiers, constraint charts, and more.
"""

import sys
import os

# Add build directory to path for development
build_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build')
if os.path.exists(build_dir):
    sys.path.insert(0, os.path.abspath(build_dir))

import knapsack
import random

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available. Install with: pip install matplotlib")
    print("    This example requires matplotlib for visualization.\n")
    sys.exit(1)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def generate_test_data(num_items=50, seed=42):
    """Generate test dataset for visualization"""
    random.seed(seed)
    
    items = []
    for i in range(num_items):
        value = random.randint(10, 1000)
        weight = random.randint(1, 50)
        category = random.choice(['A', 'B', 'C', 'D'])
        
        items.append({
            'id': i,
            'value': value,
            'weight': weight,
            'ratio': value / weight,
            'category': category
        })
    
    return items


def example_1_basic_scatter_plot(items, capacity):
    """Basic scatter plot showing selected vs. rejected items"""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Scatter Plot (Selected vs. Rejected)")
    print("="*80)
    
    values = [item['value'] for item in items]
    weights = [item['weight'] for item in items]
    
    solution = knapsack.solve(values, weights, capacity, {"beam_width": 32, "iters": 5})
    
    selected_set = set(solution.selected_indices)
    
    # Separate selected and rejected
    selected_values = [items[i]['value'] for i in solution.selected_indices]
    selected_weights = [items[i]['weight'] for i in solution.selected_indices]
    
    rejected_values = [item['value'] for i, item in enumerate(items) if i not in selected_set]
    rejected_weights = [item['weight'] for i, item in enumerate(items) if i not in selected_set]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(rejected_weights, rejected_values, c='lightgray', s=100, alpha=0.6, 
               label='Rejected', edgecolors='gray', linewidth=1)
    ax.scatter(selected_weights, selected_values, c='#2ecc71', s=100, alpha=0.8,
               label='Selected', edgecolors='darkgreen', linewidth=2)
    
    ax.set_xlabel('Weight', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Knapsack Solution: {len(solution.selected_indices)}/{len(items)} items selected\n'
                 f'Total Value: {solution.best_value:.0f}, Capacity: {capacity}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = '/tmp/knapsack_scatter.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_file}")
    plt.close()


def example_2_value_weight_ratio_analysis(items, capacity):
    """Analyze value/weight ratio and selection"""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Value/Weight Ratio Analysis")
    print("="*80)
    
    values = [item['value'] for item in items]
    weights = [item['weight'] for item in items]
    
    solution = knapsack.solve(values, weights, capacity, {"beam_width": 32, "iters": 5})
    selected_set = set(solution.selected_indices)
    
    # Calculate ratios
    ratios = [v/w for v, w in zip(values, weights)]
    selected_ratios = [ratios[i] for i in solution.selected_indices]
    rejected_ratios = [ratios[i] for i in range(len(items)) if i not in selected_set]
    
    # Create histogram comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(rejected_ratios, bins=20, alpha=0.5, color='lightcoral', label='Rejected', edgecolor='black')
    ax1.hist(selected_ratios, bins=20, alpha=0.7, color='#3498db', label='Selected', edgecolor='black')
    ax1.set_xlabel('Value/Weight Ratio', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Value/Weight Ratios', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Box plot
    ax2.boxplot([rejected_ratios, selected_ratios], labels=['Rejected', 'Selected'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.5),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Value/Weight Ratio', fontsize=11, fontweight='bold')
    ax2.set_title('Ratio Comparison: Selected vs Rejected', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = '/tmp/knapsack_ratio_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_file}")
    plt.close()


def example_3_capacity_sensitivity(items):
    """Show how solution changes with different capacities"""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Capacity Sensitivity Analysis")
    print("="*80)
    
    values = [item['value'] for item in items]
    weights = [item['weight'] for item in items]
    total_weight = sum(weights)
    
    # Test different capacity levels
    capacity_levels = [total_weight * pct for pct in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]
    
    results = []
    for cap in capacity_levels:
        solution = knapsack.solve(values, weights, cap, {"beam_width": 32, "iters": 5})
        selected_weight = sum(weights[i] for i in solution.selected_indices)
        
        results.append({
            'capacity': cap,
            'capacity_pct': cap / total_weight * 100,
            'items_selected': len(solution.selected_indices),
            'total_value': solution.best_value,
            'utilization': selected_weight / cap * 100 if cap > 0 else 0
        })
    
    # Create multi-panel plot
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Items selected vs capacity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot([r['capacity'] for r in results], [r['items_selected'] for r in results],
             'o-', linewidth=2, markersize=8, color='#3498db')
    ax1.set_xlabel('Capacity', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Items Selected', fontsize=11, fontweight='bold')
    ax1.set_title('Items vs. Capacity', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Total value vs capacity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot([r['capacity'] for r in results], [r['total_value'] for r in results],
             'o-', linewidth=2, markersize=8, color='#2ecc71')
    ax2.set_xlabel('Capacity', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Total Value', fontsize=11, fontweight='bold')
    ax2.set_title('Value vs. Capacity', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Capacity utilization
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(range(len(results)), [r['utilization'] for r in results],
            color='#e74c3c', alpha=0.7, edgecolor='black')
    ax3.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100% utilization')
    ax3.set_xlabel('Capacity Level', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Utilization (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Capacity Utilization', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(results)))
    ax3.set_xticklabels([f"{r['capacity_pct']:.0f}%" for r in results], rotation=45)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Value per item
    ax4 = fig.add_subplot(gs[1, 1])
    value_per_item = [r['total_value'] / r['items_selected'] if r['items_selected'] > 0 else 0
                      for r in results]
    ax4.plot([r['capacity'] for r in results], value_per_item,
             'o-', linewidth=2, markersize=8, color='#9b59b6')
    ax4.set_xlabel('Capacity', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Avg Value per Item', fontsize=11, fontweight='bold')
    ax4.set_title('Average Value per Selected Item', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle('Capacity Sensitivity Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    output_file = '/tmp/knapsack_capacity_sensitivity.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_file}")
    plt.close()


def example_4_pareto_frontier(items, capacity):
    """Visualize Pareto frontier of solutions"""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Pareto Frontier Visualization")
    print("="*80)
    
    values = [item['value'] for item in items]
    weights = [item['weight'] for item in items]
    
    # Generate multiple solutions with different beam widths
    configs = [
        {"beam_width": 8, "iters": 3, "label": "Fast (beam=8)"},
        {"beam_width": 16, "iters": 3, "label": "Default (beam=16)"},
        {"beam_width": 32, "iters": 5, "label": "Balanced (beam=32)"},
        {"beam_width": 64, "iters": 5, "label": "Quality (beam=64)"},
        {"beam_width": 128, "iters": 10, "label": "Maximum (beam=128)"},
    ]
    
    solutions = []
    for config in configs:
        sol = knapsack.solve(values, weights, capacity, config)
        selected_weight = sum(weights[i] for i in sol.selected_indices)
        solutions.append({
            'label': config['label'],
            'value': sol.best_value,
            'weight': selected_weight,
            'time': sol.solve_time_ms,
            'items': len(sol.selected_indices)
        })
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Value vs Weight (Pareto frontier)
    colors = plt.cm.viridis(np.linspace(0, 1, len(solutions)))
    for i, sol in enumerate(solutions):
        ax1.scatter(sol['weight'], sol['value'], s=200, c=[colors[i]], 
                   alpha=0.7, edgecolors='black', linewidth=2, label=sol['label'])
    
    # Draw capacity line
    ax1.axvline(x=capacity, color='red', linestyle='--', linewidth=2, label=f'Capacity ({capacity})')
    
    ax1.set_xlabel('Total Weight', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Value', fontsize=12, fontweight='bold')
    ax1.set_title('Value vs. Weight Trade-off', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Value vs Time
    for i, sol in enumerate(solutions):
        ax2.scatter(sol['time'], sol['value'], s=200, c=[colors[i]],
                   alpha=0.7, edgecolors='black', linewidth=2, label=sol['label'])
    
    ax2.set_xlabel('Solve Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Value', fontsize=12, fontweight='bold')
    ax2.set_title('Value vs. Computation Time', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_file = '/tmp/knapsack_pareto_frontier.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_file}")
    plt.close()


def example_5_scout_mode_frequency_heatmap(items, capacity):
    """Visualize item selection frequency from scout mode"""
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Scout Mode - Item Selection Frequency")
    print("="*80)
    
    values = [item['value'] for item in items]
    weights = [item['weight'] for item in items]
    
    # Run scout mode
    scout_result = knapsack.solve_scout(
        values, weights, capacity,
        {"beam_width": 32, "iters": 5, "scout_threshold": 0.3, "scout_top_k": 10}
    )
    
    # Sort items by frequency
    item_freq = [(i, scout_result.item_frequency[i], items[i]) 
                 for i in range(len(items))]
    item_freq.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 20 for visualization
    top_items = item_freq[:20]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(top_items))
    frequencies = [item[1] for item in top_items]
    labels = [f"Item {item[0]} (V:{item[2]['value']}, W:{item[2]['weight']})" 
              for item in top_items]
    
    # Color by frequency
    colors = plt.cm.RdYlGn(frequencies)
    
    bars = ax.barh(y_pos, frequencies, color=colors, edgecolor='black', linewidth=1)
    
    # Highlight active items
    active_set = set(scout_result.active_items)
    for i, (idx, freq, _) in enumerate(top_items):
        if idx in active_set:
            bars[i].set_edgecolor('blue')
            bars[i].set_linewidth(3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Selection Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Scout Mode: Top 20 Items by Selection Frequency\n'
                 f'Active Set: {scout_result.active_item_count}/{scout_result.original_item_count} items '
                 f'({100*(1-scout_result.active_item_count/scout_result.original_item_count):.1f}% reduction)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    blue_patch = mpatches.Patch(edgecolor='blue', facecolor='white', linewidth=3, label='Active Item')
    ax.legend(handles=[blue_patch], fontsize=10, loc='lower right')
    
    plt.tight_layout()
    output_file = '/tmp/knapsack_scout_frequency.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_file}")
    plt.close()


def example_6_category_breakdown(items, capacity):
    """Visualize selection by category"""
    
    print("\n" + "="*80)
    print("EXAMPLE 6: Category Breakdown Analysis")
    print("="*80)
    
    values = [item['value'] for item in items]
    weights = [item['weight'] for item in items]
    
    solution = knapsack.solve(values, weights, capacity, {"beam_width": 32, "iters": 5})
    selected_set = set(solution.selected_indices)
    
    # Count by category
    categories = {}
    for i, item in enumerate(items):
        cat = item['category']
        if cat not in categories:
            categories[cat] = {'total': 0, 'selected': 0, 'total_value': 0, 'selected_value': 0}
        
        categories[cat]['total'] += 1
        categories[cat]['total_value'] += item['value']
        
        if i in selected_set:
            categories[cat]['selected'] += 1
            categories[cat]['selected_value'] += item['value']
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    cat_names = list(categories.keys())
    
    # Plot 1: Selection rate by category
    selection_rates = [categories[cat]['selected'] / categories[cat]['total'] * 100
                       for cat in cat_names]
    ax1.bar(cat_names, selection_rates, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Selection Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Selection Rate by Category', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Count comparison
    x = np.arange(len(cat_names))
    width = 0.35
    ax2.bar(x - width/2, [categories[cat]['total'] for cat in cat_names],
            width, label='Total', color='lightgray', edgecolor='black')
    ax2.bar(x + width/2, [categories[cat]['selected'] for cat in cat_names],
            width, label='Selected', color='#2ecc71', edgecolor='black')
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title('Items: Total vs Selected', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_names)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Value distribution (pie chart)
    selected_values = [categories[cat]['selected_value'] for cat in cat_names]
    colors_pie = plt.cm.Set3(np.arange(len(cat_names)))
    ax3.pie(selected_values, labels=cat_names, autopct='%1.1f%%',
            colors=colors_pie, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax3.set_title('Value Distribution by Category', fontsize=12, fontweight='bold')
    
    # Plot 4: Value efficiency
    value_efficiency = [categories[cat]['selected_value'] / categories[cat]['selected']
                       if categories[cat]['selected'] > 0 else 0
                       for cat in cat_names]
    ax4.bar(cat_names, value_efficiency, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Avg Value per Item', fontsize=11, fontweight='bold')
    ax4.set_title('Average Value per Selected Item', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Category Breakdown Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = '/tmp/knapsack_category_breakdown.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_file}")
    plt.close()


def main():
    """Run all visualization examples"""
    
    print("╔" + "═"*78 + "╗")
    print("║" + " VISUALIZATION AND ANALYSIS EXAMPLES ".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    
    if not MATPLOTLIB_AVAILABLE:
        print("\n❌ matplotlib is required for this example")
        print("   Install with: pip install matplotlib")
        return 1
    
    print("\n📊 Generating test data...")
    items = generate_test_data(num_items=50, seed=42)
    capacity = sum(item['weight'] for item in items) * 0.4
    
    print(f"Dataset: {len(items)} items, capacity: {capacity:.1f}")
    
    example_1_basic_scatter_plot(items, capacity)
    example_2_value_weight_ratio_analysis(items, capacity)
    example_3_capacity_sensitivity(items)
    example_4_pareto_frontier(items, capacity)
    example_5_scout_mode_frequency_heatmap(items, capacity)
    example_6_category_breakdown(items, capacity)
    
    print("\n" + "="*80)
    print("✅ All visualization examples completed!")
    print("="*80)
    print("\nGenerated files:")
    print("  • /tmp/knapsack_scatter.png")
    print("  • /tmp/knapsack_ratio_analysis.png")
    print("  • /tmp/knapsack_capacity_sensitivity.png")
    print("  • /tmp/knapsack_pareto_frontier.png")
    print("  • /tmp/knapsack_scout_frequency.png")
    print("  • /tmp/knapsack_category_breakdown.png")
    print("\nKey Insights:")
    print("  • Scatter plots reveal selection patterns")
    print("  • Ratio analysis shows greedy heuristic limitations")
    print("  • Capacity sensitivity helps tuning resource allocation")
    print("  • Pareto frontiers show quality vs. speed trade-offs")
    print("  • Scout mode frequencies identify core items")


if __name__ == "__main__":
    sys.exit(main() or 0)
