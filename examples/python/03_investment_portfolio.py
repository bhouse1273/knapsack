#!/usr/bin/env python3
"""
Example 3: Investment Portfolio Selection

Optimize investment allocation with risk-adjusted returns, diversification,
and capital constraints. Demonstrates financial portfolio optimization.
"""

import sys
import os
import random
import math

# Add build directory to path for development
build_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build')
if os.path.exists(build_dir):
    sys.path.insert(0, os.path.abspath(build_dir))

import knapsack


def generate_investment_universe(num_investments=50, seed=42):
    """Generate realistic investment opportunities"""
    
    random.seed(seed)
    investments = []
    
    sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer", "Industrial", "Real Estate"]
    risk_levels = ["Low", "Medium", "High"]
    
    for i in range(num_investments):
        sector = random.choice(sectors)
        risk_level = random.choice(risk_levels)
        
        # Expected return varies by risk
        if risk_level == "Low":
            expected_return = random.uniform(0.05, 0.10)  # 5-10%
            volatility = random.uniform(0.05, 0.15)
        elif risk_level == "Medium":
            expected_return = random.uniform(0.08, 0.15)  # 8-15%
            volatility = random.uniform(0.12, 0.25)
        else:  # High
            expected_return = random.uniform(0.12, 0.25)  # 12-25%
            volatility = random.uniform(0.20, 0.40)
        
        # Sharpe ratio (risk-adjusted return)
        risk_free_rate = 0.03  # 3% risk-free rate
        sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Investment amount (minimum check size)
        min_investment = random.choice([10000, 25000, 50000, 100000, 250000])
        
        # ESG score (Environmental, Social, Governance)
        esg_score = random.randint(1, 100)
        
        # Liquidity (days to exit)
        liquidity_days = random.choice([1, 7, 30, 90, 180, 365])
        
        # Value metric: expected return weighted by Sharpe ratio and ESG
        esg_factor = 0.5 + (esg_score / 200)  # 0.5 to 1.0
        value = min_investment * expected_return * sharpe_ratio * esg_factor
        
        investments.append({
            'id': f"INV-{i+1:03d}",
            'name': f"{sector} {risk_level} {i+1}",
            'sector': sector,
            'risk_level': risk_level,
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'min_investment': min_investment,
            'esg_score': esg_score,
            'liquidity_days': liquidity_days,
            'value': value
        })
    
    return investments


def example_1_basic_portfolio(investments, total_capital=2000000):
    """Basic portfolio allocation maximizing risk-adjusted returns"""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Portfolio Allocation")
    print("="*80)
    
    values = [inv['value'] for inv in investments]
    capitals = [inv['min_investment'] for inv in investments]
    
    total_opportunity = sum(capitals)
    total_potential_value = sum(values)
    
    print(f"\nInvestment Universe:")
    print(f"  Opportunities: {len(investments)}")
    print(f"  Total capital required (all): ${total_opportunity:,.0f}")
    print(f"  Available capital: ${total_capital:,.0f}")
    print(f"  Capital utilization: {total_capital/total_opportunity*100:.1f}%")
    print(f"  Total potential value: ${total_potential_value:,.0f}")
    
    # Solve
    solution = knapsack.solve(
        values=values,
        weights=capitals,
        capacity=total_capital,
        config={"beam_width": 32, "iters": 5}
    )
    
    # Analysis
    selected_investments = [investments[i] for i in solution.selected_indices]
    total_invested = sum(inv['min_investment'] for inv in selected_investments)
    avg_return = sum(inv['expected_return'] for inv in selected_investments) / len(selected_investments)
    avg_sharpe = sum(inv['sharpe_ratio'] for inv in selected_investments) / len(selected_investments)
    avg_esg = sum(inv['esg_score'] for inv in selected_investments) / len(selected_investments)
    
    print(f"\n{'PORTFOLIO SOLUTION':-^80}")
    print(f"Selected investments: {len(selected_investments)} / {len(investments)}")
    print(f"Capital allocated: ${total_invested:,.0f} / ${total_capital:,.0f} ({total_invested/total_capital*100:.1f}%)")
    print(f"Portfolio value score: ${solution.best_value:,.0f}")
    print(f"Average expected return: {avg_return:.2%}")
    print(f"Average Sharpe ratio: {avg_sharpe:.2f}")
    print(f"Average ESG score: {avg_esg:.1f}/100")
    print(f"Solve time: {solution.solve_time_ms:.2f} ms")
    
    # Sector diversification
    sectors = {}
    for inv in selected_investments:
        sectors[inv['sector']] = sectors.get(inv['sector'], 0) + inv['min_investment']
    
    print(f"\n{'SECTOR ALLOCATION':-^80}")
    for sector, amount in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
        pct = amount / total_invested * 100
        print(f"  {sector:<15} ${amount:>12,.0f} ({pct:>5.1f}%)")
    
    # Risk profile
    risk_profile = {"Low": 0, "Medium": 0, "High": 0}
    for inv in selected_investments:
        risk_profile[inv['risk_level']] += inv['min_investment']
    
    print(f"\n{'RISK PROFILE':-^80}")
    for risk, amount in sorted(risk_profile.items(), key=lambda x: ["Low", "Medium", "High"].index(x[0])):
        pct = amount / total_invested * 100 if total_invested > 0 else 0
        print(f"  {risk:<15} ${amount:>12,.0f} ({pct:>5.1f}%)")
    
    return solution, selected_investments


def example_2_esg_focused_portfolio(investments, total_capital=2000000, min_esg_score=60):
    """ESG-focused portfolio with sustainability constraints"""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: ESG-Focused Portfolio")
    print("="*80)
    
    # Filter by ESG score
    esg_investments = [inv for inv in investments if inv['esg_score'] >= min_esg_score]
    
    print(f"\nESG Filtering:")
    print(f"  Minimum ESG score: {min_esg_score}/100")
    print(f"  Original universe: {len(investments)} investments")
    print(f"  ESG-qualified: {len(esg_investments)} investments ({len(esg_investments)/len(investments)*100:.1f}%)")
    
    values = [inv['value'] for inv in esg_investments]
    capitals = [inv['min_investment'] for inv in esg_investments]
    
    solution = knapsack.solve(values, capitals, total_capital, {"beam_width": 32, "iters": 5})
    
    selected = [esg_investments[i] for i in solution.selected_indices]
    total_invested = sum(inv['min_investment'] for inv in selected)
    avg_esg = sum(inv['esg_score'] for inv in selected) / len(selected)
    avg_return = sum(inv['expected_return'] for inv in selected) / len(selected)
    
    print(f"\n{'ESG PORTFOLIO':-^80}")
    print(f"Selected: {len(selected)} investments")
    print(f"Capital allocated: ${total_invested:,.0f}")
    print(f"Portfolio avg ESG: {avg_esg:.1f}/100")
    print(f"Expected return: {avg_return:.2%}")
    
    # Top ESG investments
    print(f"\n{'TOP 5 ESG INVESTMENTS':-^80}")
    print(f"{'ID':<10} {'Sector':<15} {'Return':<10} {'ESG':<6} {'Capital'}")
    print("-" * 80)
    
    sorted_esg = sorted(selected, key=lambda x: x['esg_score'], reverse=True)[:5]
    for inv in sorted_esg:
        print(f"{inv['id']:<10} {inv['sector']:<15} {inv['expected_return']:>8.1%} "
              f"{inv['esg_score']:>4}/100 ${inv['min_investment']:>12,.0f}")


def example_3_risk_adjusted_optimization(investments, total_capital=2000000):
    """Optimize for maximum Sharpe ratio (risk-adjusted returns)"""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Risk-Adjusted Optimization (Sharpe Ratio)")
    print("="*80)
    
    # Use Sharpe ratio * capital as value (risk-adjusted value)
    values = [inv['sharpe_ratio'] * inv['min_investment'] for inv in investments]
    capitals = [inv['min_investment'] for inv in investments]
    
    solution = knapsack.solve(values, capitals, total_capital, {"beam_width": 32, "iters": 5})
    
    selected = [investments[i] for i in solution.selected_indices]
    total_invested = sum(inv['min_investment'] for inv in selected)
    
    # Calculate portfolio statistics
    portfolio_return = sum(inv['expected_return'] * inv['min_investment'] for inv in selected) / total_invested
    portfolio_risk = math.sqrt(sum((inv['volatility'] * inv['min_investment'])**2 for inv in selected)) / total_invested
    portfolio_sharpe = (portfolio_return - 0.03) / portfolio_risk if portfolio_risk > 0 else 0
    
    print(f"\n{'RISK-ADJUSTED PORTFOLIO':-^80}")
    print(f"Selected: {len(selected)} investments")
    print(f"Capital allocated: ${total_invested:,.0f}")
    print(f"Portfolio expected return: {portfolio_return:.2%}")
    print(f"Portfolio volatility: {portfolio_risk:.2%}")
    print(f"Portfolio Sharpe ratio: {portfolio_sharpe:.2f}")
    
    print(f"\n{'TOP SHARPE RATIO INVESTMENTS':-^80}")
    print(f"{'ID':<10} {'Sector':<15} {'Return':<10} {'Volatility':<12} {'Sharpe':<8} {'Capital'}")
    print("-" * 80)
    
    sorted_sharpe = sorted(selected, key=lambda x: x['sharpe_ratio'], reverse=True)[:5]
    for inv in sorted_sharpe:
        print(f"{inv['id']:<10} {inv['sector']:<15} {inv['expected_return']:>8.1%} "
              f"{inv['volatility']:>10.1%} {inv['sharpe_ratio']:>6.2f} ${inv['min_investment']:>12,.0f}")


def example_4_liquidity_constrained(investments, total_capital=2000000, max_illiquid_pct=0.3):
    """Portfolio with liquidity constraints"""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Liquidity-Constrained Portfolio")
    print("="*80)
    
    print(f"\nLiquidity Constraint:")
    print(f"  Maximum illiquid allocation: {max_illiquid_pct*100:.0f}% of portfolio")
    print(f"  Illiquid defined as: >30 days to exit")
    
    values = [inv['value'] for inv in investments]
    capitals = [inv['min_investment'] for inv in investments]
    
    solution = knapsack.solve(values, capitals, total_capital, {"beam_width": 32, "iters": 5})
    
    selected = [investments[i] for i in solution.selected_indices]
    total_invested = sum(inv['min_investment'] for inv in selected)
    
    # Liquidity analysis
    liquid = sum(inv['min_investment'] for inv in selected if inv['liquidity_days'] <= 30)
    illiquid = total_invested - liquid
    illiquid_pct = illiquid / total_invested if total_invested > 0 else 0
    
    print(f"\n{'LIQUIDITY ANALYSIS':-^80}")
    print(f"Total invested: ${total_invested:,.0f}")
    print(f"Liquid (<= 30 days): ${liquid:,.0f} ({liquid/total_invested*100:.1f}%)")
    print(f"Illiquid (> 30 days): ${illiquid:,.0f} ({illiquid_pct*100:.1f}%)")
    print(f"Constraint satisfied: {'✓ Yes' if illiquid_pct <= max_illiquid_pct else '✗ No'}")
    
    # Liquidity profile
    liquidity_buckets = [
        ("1 day", 1),
        ("1 week", 7),
        ("1 month", 30),
        ("3 months", 90),
        ("6 months", 180),
        ("1 year", 365)
    ]
    
    print(f"\n{'LIQUIDITY PROFILE':-^80}")
    for label, days in liquidity_buckets:
        amount = sum(inv['min_investment'] for inv in selected if inv['liquidity_days'] == days)
        if amount > 0:
            pct = amount / total_invested * 100
            print(f"  {label:<12} ${amount:>12,.0f} ({pct:>5.1f}%)")


def example_5_scout_mode_for_portfolio(investments, total_capital=2000000):
    """Use scout mode to identify core holdings"""
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Scout Mode - Identify Core Holdings")
    print("="*80)
    
    values = [inv['value'] for inv in investments]
    capitals = [inv['min_investment'] for inv in investments]
    
    print(f"\nPhase 1: Beam search scout")
    print(f"  Investment universe: {len(investments)}")
    print(f"  Available capital: ${total_capital:,.0f}")
    
    scout_result = knapsack.solve_scout(
        values=values,
        weights=capitals,
        capacity=total_capital,
        config={
            "beam_width": 32,
            "iters": 5,
            "scout_threshold": 0.5,
            "scout_top_k": 8
        }
    )
    
    reduction_pct = 100.0 * (1.0 - scout_result.active_item_count / scout_result.original_item_count)
    
    print(f"\nScout Results:")
    print(f"  Original universe: {scout_result.original_item_count} investments")
    print(f"  Core holdings identified: {scout_result.active_item_count}")
    print(f"  Reduction: {reduction_pct:.1f}%")
    print(f"  Solve time: {scout_result.solve_time_ms:.1f} ms")
    
    # Analyze core holdings
    core_investments = [investments[i] for i in scout_result.active_items]
    core_capital = sum(inv['min_investment'] for inv in core_investments)
    avg_sharpe = sum(inv['sharpe_ratio'] for inv in core_investments) / len(core_investments)
    
    print(f"\nCore Holdings:")
    print(f"  Investments: {len(core_investments)}")
    print(f"  Total capital: ${core_capital:,.0f}")
    print(f"  Avg Sharpe ratio: {avg_sharpe:.2f}")
    
    # Sector breakdown of core
    sectors = {}
    for inv in core_investments:
        sectors[inv['sector']] = sectors.get(inv['sector'], 0) + 1
    
    print(f"\n{'CORE HOLDINGS BY SECTOR':-^80}")
    for sector, count in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(core_investments) * 100
        print(f"  {sector:<15} {count:>3} investments ({pct:>5.1f}%)")
    
    print(f"\n{'TOP 10 CORE HOLDINGS (by selection frequency)':-^80}")
    print(f"{'ID':<10} {'Sector':<15} {'Return':<10} {'Sharpe':<8} {'Frequency'}")
    print("-" * 80)
    
    # Sort by frequency
    core_with_freq = [(investments[i], scout_result.item_frequency[i])
                      for i in scout_result.active_items]
    core_with_freq.sort(key=lambda x: x[1], reverse=True)
    
    for inv, freq in core_with_freq[:10]:
        print(f"{inv['id']:<10} {inv['sector']:<15} {inv['expected_return']:>8.1%} "
              f"{inv['sharpe_ratio']:>6.2f} {freq:>8.1%}")
    
    print(f"\n{'RECOMMENDED WORKFLOW':-^80}")
    print("  1. Use core holdings for initial portfolio construction")
    print("  2. Add satellite positions from remaining universe as needed")
    print("  3. Rebalance quarterly using updated scout results")
    print("  4. Monitor frequency scores to identify persistent winners")


def main():
    """Run all investment portfolio examples"""
    
    print("╔" + "═"*78 + "╗")
    print("║" + " INVESTMENT PORTFOLIO OPTIMIZATION EXAMPLES ".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    
    print("\n📈 Generating investment universe...")
    investments = generate_investment_universe(num_investments=50, seed=42)
    
    example_1_basic_portfolio(investments, total_capital=2000000)
    example_2_esg_focused_portfolio(investments, total_capital=2000000, min_esg_score=70)
    example_3_risk_adjusted_optimization(investments, total_capital=2000000)
    example_4_liquidity_constrained(investments, total_capital=2000000, max_illiquid_pct=0.3)
    example_5_scout_mode_for_portfolio(investments, total_capital=2000000)
    
    print("\n" + "="*80)
    print("✅ All investment portfolio examples completed!")
    print("="*80)
    print("\nKey Takeaways:")
    print("  • Risk-adjusted returns (Sharpe ratio) are key to portfolio quality")
    print("  • ESG constraints can be incorporated without major return sacrifice")
    print("  • Liquidity management prevents over-allocation to illiquid assets")
    print("  • Scout mode identifies persistent 'core holdings' across scenarios")


if __name__ == "__main__":
    main()
