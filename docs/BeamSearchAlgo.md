## **Beam Search - Simple Explanation**

**The Problem:**
When solving optimization problems (like knapsack, routing, scheduling), the search space is enormous. For example, with 100 items and deciding yes/no for each, there are 2^100 possible solutions—far too many to check exhaustively.

**The Idea:**
Beam search is a **heuristic search strategy** that explores the solution space by:
1. **Keeping only the K best candidates** at each step (the "beam width")
2. **Expanding each candidate** by making small changes (mutations/moves)
3. **Pruning aggressively** to stay within the beam width
4. **Iterating** until time runs out or quality plateaus

Think of it as a **controlled breadth-first search** where you:
- Don't explore everything (too expensive)
- Don't commit to one path too early (too greedy)
- Keep a diverse set of promising solutions and evolve them

---

## **Concrete Example (Knapsack)**

**Given:**
- 50 items, capacity constraint
- Goal: maximize value without exceeding capacity

**Beam Search Process:**

### **Step 1: Seed the beam (K=16)**
- Generate 16 initial solutions using different strategies:
  - Greedy by value-to-weight ratio
  - Random selections
  - Density-based heuristics
- Keep the 16 best (highest objective - penalty)

### **Step 2: Expand (mutate each candidate)**
For each of the 16 solutions, create variations:
- Flip item 0 (in → out or out → in)
- Flip item 1
- Swap item 5 with item 12
- etc.

This generates ~800 candidates (16 × 50 mutations)

### **Step 3: Evaluate and prune**
- Score all 800 candidates (GPU accelerated)
- Keep only the best 16 for the next iteration

### **Step 4: Repeat**
Run for N iterations or until improvement stops.

**Final result:** The best solution seen across all iterations.

---

## **Why It Works**

✅ **Diverse exploration**: Beam width prevents getting stuck in local optima  
✅ **Scalable**: Computation grows with beam width × candidates, not solution space size  
✅ **GPU-friendly**: Evaluating hundreds of candidates in parallel is exactly what GPUs excel at  
✅ **Anytime algorithm**: Returns best-so-far; more time = better quality  

---

## **Key Parameters**

| Parameter | Effect |
|-----------|--------|
| **Beam width** | More = better quality, slower; 16-64 typical |
| **Iterations** | More = better quality; diminishing returns after convergence |
| **Mutation operators** | Domain-specific (flips, swaps, repairs) |
| **Seed quality** | Better initial solutions → faster convergence |

---

## **Beam Search vs. Alternatives**

| Method | Completeness | Speed | Quality |
|--------|--------------|-------|---------|
| **Exhaustive** | ✅ Optimal | ❌ Exponential | ✅ Best |
| **Greedy** | ❌ No backtrack | ✅ Very fast | ⚠️ Often poor |
| **Genetic Algorithm** | ❌ Heuristic | ⚠️ Medium | ⚠️ Variable |
| **Beam Search** | ❌ Heuristic | ✅ Fast (GPU) | ✅ Good |
| **Branch & Bound** | ✅ Optimal* | ❌ Can be slow | ✅ Best* |

*For problems with tight constraints and exact solvers (like Gurobi)

---

## **Why We Use It**

In our V2 solver:
- **Real-time decisions** needed (100-500ms SLA)
- **Soft constraints** make exact methods hard
- **Block structure** lets us decompose and parallelize
- **GPU acceleration** (Metal/CUDA) makes evaluation cheap
- **Good-enough solutions** beat perfect solutions that arrive too late

**Bottom line:** Beam search gives us high-quality solutions quickly by intelligently exploring a tiny fraction of the search space, keeping the best candidates alive, and leveraging GPU parallelism.

---

## **One-Sentence Summary**

> "Beam search is a controlled parallel search that keeps the K most promising solutions at each step, expands them with small changes, and prunes aggressively—trading completeness for speed while avoiding local optima."

---
