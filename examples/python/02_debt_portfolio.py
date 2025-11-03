#!/usr/bin/env python3
"""
Example 2: Debt Portfolio Selection

Optimize debt collection priorities with limited resources.
Models realistic debt collection scenarios with probability, urgency, and effort.
"""

import sys
import os
import random

# Add build directory to path for development
build_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build')
if os.path.exists(build_dir):
    sys.path.insert(0, os.path.abspath(build_dir))

import knapsack


def generate_debt_portfolio(num_debts=100, seed=42):
    """Generate realistic synthetic debt portfolio"""
    
    random.seed(seed)
    debts = []
    
    for i in range(num_debts):
        # Debt characteristics
        balance = random.randint(500, 50000)
        age_months = random.randint(1, 60)
        credit_score = random.randint(300, 850)
        
        # Collection probability (decreases with age, increases with credit score)
        base_prob = 0.8
        age_penalty = min(0.5, age_months * 0.01)
        credit_factor = (credit_score - 300) / 550
        collection_prob = max(0.1, base_prob - age_penalty) * (0.5 + 0.5 * credit_factor)
        
        # Urgency factor (time-sensitive priority)
        if age_months < 6:
            urgency = 1.5  # Fresh debts
        elif age_months < 12:
            urgency = 1.2
        elif age_months < 24:
            urgency = 1.0
        else:
            urgency = 0.7  # Old debts
        
        # Expected recovery value
        expected_value = balance * collection_prob * urgency
        
        # Effort required (staff hours + legal costs)
        staff_hours = 2 + (balance / 10000) * 5
        legal_cost = max(0, balance * 0.05) if credit_score < 500 else 0
        total_effort = staff_hours + (legal_cost / 100)
        
        debts.append({
            'id': f"DEBT-{i+1:04d}",
            'balance': balance,
            'age_months': age_months,
            'credit_score': credit_score,
            'collection_prob': collection_prob,
            'urgency': urgency,
            'expected_value': expected_value,
            'staff_hours': staff_hours,
            'legal_cost': legal_cost,
            'total_effort': total_effort
        })
    
    return debts


def example_1_basic_debt_selection(debts, max_staff_hours=300):
    """Basic debt portfolio optimization"""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Debt Portfolio Selection")
    print("="*80)
    
    values = [d['expected_value'] for d in debts]
    efforts = [d['total_effort'] for d in debts]
    
    total_balance = sum(d['balance'] for d in debts)
    total_expected = sum(values)
    total_effort = sum(efforts)
    
    print(f"\nPortfolio Overview:")
    print(f"  Total debts: {len(debts)}")
    print(f"  Outstanding balance: ${total_balance:,.2f}")
    print(f"  Expected recovery (if all pursued): ${total_expected:,.2f}")
    print(f"  Total effort required: {total_effort:.1f} staff hours")
    print(f"  Available capacity: {max_staff_hours} staff hours")
    print(f"  Capacity utilization: {max_staff_hours/total_effort*100:.1f}%")
    
    # Solve
    solution = knapsack.solve(
        values=values,
        weights=efforts,
        capacity=max_staff_hours,
        config={"beam_width": 32, "iters": 5}
    )
    
    # Analysis
    selected_debts = [debts[i] for i in solution.selected_indices]
    selected_balance = sum(d['balance'] for d in selected_debts)
    selected_effort = sum(d['total_effort'] for d in selected_debts)
    avg_prob = sum(d['collection_prob'] for d in selected_debts) / len(selected_debts)
    avg_age = sum(d['age_months'] for d in selected_debts) / len(selected_debts)
    
    print(f"\n{'SOLUTION':-^80}")
    print(f"Selected: {len(selected_debts)} / {len(debts)} debts ({len(selected_debts)/len(debts)*100:.1f}%)")
    print(f"Total balance to pursue: ${selected_balance:,.2f}")
    print(f"Expected recovery: ${solution.best_value:,.2f}")
    print(f"Recovery rate: {solution.best_value/selected_balance*100:.1f}%")
    print(f"Staff hours allocated: {selected_effort:.1f} / {max_staff_hours} ({selected_effort/max_staff_hours*100:.1f}%)")
    print(f"Average collection probability: {avg_prob:.1%}")
    print(f"Average debt age: {avg_age:.1f} months")
    print(f"ROI: ${solution.best_value/selected_effort:.2f} per staff hour")
    print(f"Solve time: {solution.solve_time_ms:.2f} ms")
    
    # Top selected debts
    print(f"\n{'TOP 10 SELECTED DEBTS':-^80}")
    print(f"{'ID':<12} {'Balance':<12} {'Age':<6} {'Credit':<8} {'Prob':<8} {'Expected':<12} {'Effort'}")
    print("-" * 80)
    
    sorted_selected = sorted(selected_debts, key=lambda d: d['expected_value'], reverse=True)[:10]
    for debt in sorted_selected:
        print(f"{debt['id']:<12} ${debt['balance']:>10,.0f} "
              f"{debt['age_months']:>4}m {debt['credit_score']:>6} "
              f"{debt['collection_prob']:>6.1%} ${debt['expected_value']:>10,.0f} "
              f"{debt['total_effort']:>6.1f}h")
    
    return solution, selected_debts


def example_2_age_stratification(debts):
    """Analyze results by debt age"""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Age Stratification Analysis")
    print("="*80)
    
    # Solve with moderate capacity
    values = [d['expected_value'] for d in debts]
    efforts = [d['total_effort'] for d in debts]
    capacity = sum(efforts) * 0.4  # 40% of total effort
    
    solution = knapsack.solve(values, efforts, capacity)
    selected_indices = set(solution.selected_indices)
    
    # Stratify by age
    age_buckets = [
        ("0-6 months", 0, 6),
        ("6-12 months", 6, 12),
        ("12-24 months", 12, 24),
        ("24+ months", 24, 999)
    ]
    
    print(f"\nCapacity: {capacity:.1f} staff hours")
    print(f"Selected: {len(solution.selected_indices)} / {len(debts)} debts")
    
    print(f"\n{'Age Stratification':-^80}")
    print(f"{'Age Range':<15} {'Total':<8} {'Selected':<10} {'%':<8} {'Avg Balance':<15} {'Avg Prob'}")
    print("-" * 80)
    
    for label, min_age, max_age in age_buckets:
        bucket_debts = [d for d in debts if min_age <= d['age_months'] < max_age]
        bucket_selected = [d for i, d in enumerate(debts) 
                          if min_age <= d['age_months'] < max_age and i in selected_indices]
        
        if bucket_debts:
            pct = len(bucket_selected) / len(bucket_debts) * 100
            avg_balance = sum(d['balance'] for d in bucket_selected) / len(bucket_selected) if bucket_selected else 0
            avg_prob = sum(d['collection_prob'] for d in bucket_selected) / len(bucket_selected) if bucket_selected else 0
            
            print(f"{label:<15} {len(bucket_debts):<8} {len(bucket_selected):<10} "
                  f"{pct:<7.1f}% ${avg_balance:>12,.0f} {avg_prob:>8.1%}")


def example_3_credit_score_analysis(debts):
    """Analyze selection by credit score"""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Credit Score Analysis")
    print("="*80)
    
    values = [d['expected_value'] for d in debts]
    efforts = [d['total_effort'] for d in debts]
    capacity = sum(efforts) * 0.4
    
    solution = knapsack.solve(values, efforts, capacity)
    selected_indices = set(solution.selected_indices)
    
    # Stratify by credit score
    score_buckets = [
        ("Poor (300-579)", 300, 580),
        ("Fair (580-669)", 580, 670),
        ("Good (670-739)", 670, 740),
        ("Very Good (740-799)", 740, 800),
        ("Excellent (800+)", 800, 900)
    ]
    
    print(f"\n{'Credit Score Distribution':-^80}")
    print(f"{'Score Range':<25} {'Total':<8} {'Selected':<10} {'%':<8} {'Avg Value'}")
    print("-" * 80)
    
    for label, min_score, max_score in score_buckets:
        bucket_debts = [d for d in debts if min_score <= d['credit_score'] < max_score]
        bucket_selected = [d for i, d in enumerate(debts)
                          if min_score <= d['credit_score'] < max_score and i in selected_indices]
        
        if bucket_debts:
            pct = len(bucket_selected) / len(bucket_debts) * 100
            avg_value = sum(d['expected_value'] for d in bucket_selected) / len(bucket_selected) if bucket_selected else 0
            
            print(f"{label:<25} {len(bucket_debts):<8} {len(bucket_selected):<10} "
                  f"{pct:<7.1f}% ${avg_value:>12,.0f}")


def example_4_capacity_sensitivity(debts):
    """Show how selection changes with different capacity levels"""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Capacity Sensitivity Analysis")
    print("="*80)
    
    values = [d['expected_value'] for d in debts]
    efforts = [d['total_effort'] for d in debts]
    total_effort = sum(efforts)
    
    capacities = [total_effort * pct for pct in [0.2, 0.3, 0.4, 0.5, 0.6]]
    
    print(f"\n{'Capacity vs. Returns':-^80}")
    print(f"{'Capacity':<12} {'% of Total':<12} {'Debts':<8} {'Expected':<15} {'ROI':<12} {'Utilization'}")
    print("-" * 80)
    
    for capacity in capacities:
        solution = knapsack.solve(values, efforts, capacity)
        selected_effort = sum(efforts[i] for i in solution.selected_indices)
        roi = solution.best_value / selected_effort if selected_effort > 0 else 0
        utilization = selected_effort / capacity * 100
        
        print(f"{capacity:<12.1f} {capacity/total_effort*100:<12.1f} "
              f"{len(solution.selected_indices):<8} ${solution.best_value:>13,.0f} "
              f"${roi:>10.2f}/h {utilization:>10.1f}%")


def example_5_scout_mode_for_exact_solver(debts):
    """Use scout mode to identify core portfolio"""
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Scout Mode for Exact Solver Integration")
    print("="*80)
    
    values = [d['expected_value'] for d in debts]
    efforts = [d['total_effort'] for d in debts]
    capacity = sum(efforts) * 0.4
    
    print(f"\nPhase 1: Beam search scout")
    print(f"  Portfolio size: {len(debts)} debts")
    print(f"  Capacity: {capacity:.1f} staff hours")
    
    # Run scout mode
    scout_result = knapsack.solve_scout(
        values=values,
        weights=efforts,
        capacity=capacity,
        config={
            "beam_width": 32,
            "iters": 5,
            "scout_threshold": 0.5,
            "scout_top_k": 8
        }
    )
    
    reduction_pct = 100.0 * (1.0 - scout_result.active_item_count / scout_result.original_item_count)
    
    print(f"\nScout Results:")
    print(f"  Original debts: {scout_result.original_item_count}")
    print(f"  Active debts: {scout_result.active_item_count}")
    print(f"  Reduction: {reduction_pct:.1f}%")
    print(f"  Solve time: {scout_result.solve_time_ms:.1f} ms")
    
    # Analyze active portfolio
    active_debts = [debts[i] for i in scout_result.active_items]
    active_balance = sum(d['balance'] for d in active_debts)
    active_expected = sum(d['expected_value'] for d in active_debts)
    active_effort = sum(d['total_effort'] for d in active_debts)
    
    print(f"\nActive Portfolio (for exact solver):")
    print(f"  Debts: {len(active_debts)}")
    print(f"  Total balance: ${active_balance:,.2f}")
    print(f"  Expected recovery: ${active_expected:,.2f}")
    print(f"  Total effort: {active_effort:.1f} hours")
    
    print(f"\n{'TOP 10 ACTIVE DEBTS (by frequency)':-^80}")
    print(f"{'ID':<12} {'Balance':<12} {'Expected':<12} {'Effort':<8} {'Frequency'}")
    print("-" * 80)
    
    # Sort by frequency
    active_with_freq = [(debts[i], scout_result.item_frequency[i]) 
                        for i in scout_result.active_items]
    active_with_freq.sort(key=lambda x: x[1], reverse=True)
    
    for debt, freq in active_with_freq[:10]:
        print(f"{debt['id']:<12} ${debt['balance']:>10,.0f} "
              f"${debt['expected_value']:>10,.0f} {debt['total_effort']:>6.1f}h "
              f"{freq:>8.1%}")
    
    print(f"\n{'NEXT STEPS':-^80}")
    print("  1. Export active_items to CSV: active_debts.csv")
    print("  2. Pass to Gurobi/CPLEX/SCIP for proven optimal solution")
    print("  3. Use beam solution as warm start (MIP start)")
    print("  4. Expect 10-20× speedup vs. solving full problem optimally")
    
    return scout_result, active_debts


def main():
    """Run all debt portfolio examples"""
    
    print("╔" + "═"*78 + "╗")
    print("║" + " DEBT PORTFOLIO SELECTION EXAMPLES ".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    
    print("\n📊 Generating synthetic debt portfolio...")
    debts = generate_debt_portfolio(num_debts=100, seed=42)
    
    example_1_basic_debt_selection(debts, max_staff_hours=300)
    example_2_age_stratification(debts)
    example_3_credit_score_analysis(debts)
    example_4_capacity_sensitivity(debts)
    example_5_scout_mode_for_exact_solver(debts)
    
    print("\n" + "="*80)
    print("✅ All debt portfolio examples completed!")
    print("="*80)
    print("\nKey Takeaways:")
    print("  • Younger debts are prioritized (higher urgency factor)")
    print("  • Higher credit scores improve collection probability")
    print("  • Scout mode can reduce problem size by 50-80%")
    print("  • ROI analysis helps justify resource allocation decisions")


if __name__ == "__main__":
    main()
