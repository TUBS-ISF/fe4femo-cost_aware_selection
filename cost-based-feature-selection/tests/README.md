AI generated: Claude Code (Sonnet 4.6)
# Test Suite — Cost-Based Feature Selection

## Running the tests

```bash
cd cost-based-feature-selection
source venv/bin/activate
python -m pytest tests/ -v
```

45 tests, ~5 seconds.

---

## What is tested

### Cost model (10 tests — `TestGroupCostModel`)
The shared group-based cost function is verified independently of any algorithm:

- Empty subset costs 0
- First feature from a group pays the full group cost
- Additional features from the **same** group have **zero marginal cost**
- Features from different groups have **additive** costs
- Cost is order-independent (feature selection order does not matter)
- Marginal cost properties verified explicitly for first and second selections

### CostBasedCFS (12 tests — `TestCostBasedCFS`)
CFS with a cost-penalty term (λ) and optional hard budget:

- Smoke tests for classification and regression
- `selected_mask_` and `selected_indices_` are consistent
- `n_features_in_` is set correctly
- Higher cost penalty causes the algorithm to prefer cheaper features
- Static budget (`feature_costs` array) is respected
- With λ=0, cost has no effect (pure CFS)
- `total_cost_` equals `cost_fn(selected_indices_)`, not a feature count
- Budget blocks access to expensive groups when `cost_fn` is used
- `cost_fn` overrides `feature_costs` for budget decisions
- `max_features` hard cap is respected

### CostConstrainedGBSelector (12 tests — `TestCostConstrainedGB`)
XGBoost importances ranked by marginal cost with iterative greedy selection:

- Smoke tests for classification and regression
- `selected_mask_` and `selected_indices_` are consistent
- Static budget respected
- `total_cost_` equals `cost_fn(selected_indices_)`
- `total_cost_` reflects group cost, not feature count
- Budget enforced with `cost_fn`
- Budget blocks entry of expensive groups
- **Second feature from an already-selected group has zero marginal cost** (verified directly via the cost function)
- `cost_fn` overrides `feature_costs` for budget enforcement
- Both `gain` and `weight` importance types produce results

### MOPSOFeatureSelector (11 tests — `TestMOPSO`)
Multi-objective PSO with Pareto archive (cost vs. error rate):

- Smoke tests for classification and regression
- `selected_mask_` and `selected_indices_` are consistent
- Pareto front is populated after fitting
- Pareto front entries are sorted by cost (ascending)
- `total_cost_` equals `cost_fn(selected_indices_)`
- `total_cost_` reflects group cost, not feature count
- **Every Pareto front entry has a cost exactly equal to `cost_fn` applied to its selected features**
- Normalised cost objective f1 lies in [0, 1]
- All three selection strategies (`min_cost`, `max_perf`, `knee`) return a result
- **Budget behaviour**: MOPSO budget is a *selection preference* (Pareto filtering), not a hard search constraint. If any feasible Pareto entry exists, the selected solution is within budget. If none exists (possible with few particles/iterations), the algorithm falls back to the full front.

---

## Key design decisions

**`TestGroupCostModel` is separate.** The group cost function underpins all three algorithms. Testing it in isolation means a failing group-cost test points to the fixture, not to an algorithm.

**`cost_fn` vs `feature_costs`.** Every algorithm accepts both. Tests verify that `cost_fn` takes precedence over `feature_costs` for budget decisions — passing expensive static costs alongside a cheap `cost_fn` still produces a result within budget.

**MOPSO budget is soft.** Unlike CFS and GB, MOPSO does not enforce budget as a hard constraint during the particle search. It only filters the Pareto front at selection time. The test acknowledges this and only asserts budget compliance when a feasible Pareto entry actually exists.
