## **Beam Search as a Data Scout**

### **The Idea**

Use beam search as a **fast preprocessing filter** to:
1. **Identify high-value items/constraints** that matter most
2. **Eliminate dominated/low-impact items** safely
3. **Provide warm-starts** (initial feasible solutions)
4. **Reduce problem size** before handing off to an exact solver

This creates a **two-stage pipeline**:
- **Stage 1 (Beam):** Fast heuristic filtering + exploration (100-500ms)
- **Stage 2 (Exact):** Prove optimality on the reduced problem (1-10s)

---

## **How It Works**

### **1. Dominance Filtering **

Solver already has this via `ApplyDominanceFilters`:
```json
{"dom_enable": true, "dom_eps": 1e-9, "dom_surrogate": true}
```

**What it does:**
- Removes items that are strictly worse than others
- Safe transformation: never removes optimal solutions
- Typical reduction: 10-30% of items dropped

**Feed to exact solver:**
```
Original: 500 items → Beam filters → 350 items → Gurobi solves
```

### **2. Active Set Identification**

Run beam search and track:
- **Items that appear frequently** in top-K solutions
- **Items with high marginal value** in mutations
- **Constraints that are tight** across good solutions

**Use case:**
```cpp
// After beam search:
std::set<int> active_items;
for (auto& candidate : top_beam_solutions) {
    for (int i = 0; i < N; ++i) {
        if (candidate.select[i] == 1) {
            active_items.insert(i);
        }
    }
}
// Pass only active_items + their neighbors to Gurobi
```

### **3. Warm Start**

The best beam solution provides:
- **Feasible starting point** for branch-and-bound
- **Objective lower bound** (for minimization) or upper bound (maximization)
- **Reduced search tree** size in exact solver

**Example:**
```cpp
// V2 beam result:
auto beam_sol = SolveKnapsack(configJSON, optsJSON);

// Pass to Gurobi as MIP start:
for (int i = 0; i < N; ++i) {
    x[i].set(GRB_DoubleAttr_Start, beam_sol->Select[i]);
}
model.optimize();
```

### **4. Block Decomposition**

Beam search explores block structure naturally. Use it to:
- **Identify independent subproblems**
- **Solve small blocks exactly** in parallel
- **Coordinate via beam** for inter-block constraints

**Example:**
```
Beam: "Blocks 1,3,7 are saturated; blocks 2,4,5 have slack"
Exact: Solve blocks 2,4,5 to proven optimality (small problems)
Beam: Re-optimize blocks 1,3,7 with exact solutions from 2,4,5
```

---

## **Practical Pipeline**

### **Option A: Sequential (Safe)**
```
1. Beam (500ms) → filtered items + warm start
2. Gurobi (5s) → proven optimal on reduced problem
3. Return: exact solution or beam fallback if timeout
```

### **Option B: Parallel (Fast)**
```
1. Launch beam (GPU) and Gurobi (CPU) simultaneously
2. Beam finishes first → provides warm start to Gurobi mid-solve
3. Return: best of {beam@500ms, Gurobi@5s}
```

### **Option C: Iterative Refinement**
```
1. Beam iteration 1 (100ms) → identify top 20% items
2. Gurobi on reduced problem (2s) → exact solution for subset
3. Beam iteration 2 with exact subset fixed → explore rest
4. Repeat until convergence or timeout
```

---

## **Example: 1000-Item Knapsack**

**Without scouting:**
- Gurobi on 1000 items: 45 seconds (may timeout)
- Beam alone: 0.3s, 95% optimal

**With scouting:**
1. **Dominance filter:** 1000 → 720 items (50ms)
2. **Beam search:** Identify 150 active items (300ms)
3. **Gurobi exact:** Solve 150-item problem (2s) → **proven optimal**
4. **Total time:** 2.35s vs. 45s (19× speedup)

---

## **When to Use Each Approach**

| Scenario | Strategy |
|----------|----------|
| **Tight SLA (< 500ms)** | Beam only |
| **Quality matters, time available** | Beam → filter → Gurobi |
| **Need proof of optimality** | Beam warm-start → Gurobi full |
| **Large sparse problems** | Beam scout → Gurobi on active set |
| **Multiple objectives** | Beam explores Pareto → Gurobi each |

---

## **Code Sketch: Integration**

```cpp
// Stage 1: Beam scout
auto beam_opts = R"({"beam_width":32,"iters":5,"dom_enable":true})";
auto beam_sol = SolveKnapsack(configJSON, beam_opts);

// Stage 2: Extract active set
std::vector<int> active_items;
for (int i = 0; i < beam_sol->NumItems; ++i) {
    if (beam_sol->Select[i] == 1) {
        active_items.push_back(i);
    }
}

// Stage 3: Build reduced Gurobi model
GRBEnv env;
GRBModel model(env);
// ... add only active_items + neighbors as variables ...
// ... set warm start from beam_sol ...
model.set(GRB_DoubleParam_TimeLimit, 5.0);
model.optimize();

// Return best of beam or Gurobi
if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
    return gurobi_solution;  // Proven optimal
} else {
    return beam_sol;  // Heuristic fallback
}
```

---

## **Benefits**

✅ **Reduced problem size** → exact solver runs faster  
✅ **Warm starts** → fewer branch-and-bound nodes  
✅ **Anytime behavior** → always have a good solution  
✅ **Proof when possible** → exact optimality on filtered set  
✅ **Best of both worlds** → speed + quality guarantees  

---

## **Summary**

**Beam search is an excellent data scout!** It:
- Filters dominated items safely
- Identifies active variables for exact solvers
- Provides warm starts that prune search trees
- Decomposes problems into tractable subproblems
- Runs in parallel with exact methods for anytime answers
