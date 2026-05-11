# AI generated: Claude Code (Sonnet 4.6)
"""
Unit tests for CostBasedCFS, CostConstrainedGBSelector, and MOPSOFeatureSelector.

These tests run locally with synthetic data — no pipeline, Dask, or SLURM involved.

The central property under test is the group-based cost model:
  - The first feature selected from a feature group pays the full group
    extraction cost.
  - Every subsequent feature from the same already-covered group has zero
    marginal cost (the group extractor has already run).

Tests are organised as:
  1. TestGroupCostModel  — the cost function itself
  2. TestCostBasedCFS    — CFS with cost penalty and budget
  3. TestCostConstrainedGB — XGBoost-ranked greedy selector
  4. TestMOPSO           — Multi-objective PSO selector
"""
import numpy as np
import pandas as pd
import pytest

from external.cost_based.cost_cfs import CostBasedCFS
from external.cost_based.cost_constrained_gb import CostConstrainedGBSelector
from external.cost_based.mopso import MOPSOFeatureSelector


# ===========================================================================
# 1. Group cost model (cost_fn behaviour — algorithm-independent)
# ===========================================================================

class TestGroupCostModel:
    """Verify the cost function fixture itself before trusting algorithm tests."""

    def test_empty_subset_costs_zero(self, group_cost_fn):
        cost_fn, _, _ = group_cost_fn
        assert cost_fn([]) == 0.0

    def test_single_feature_pays_full_group_cost(self, group_cost_fn):
        cost_fn, _, group_costs = group_cost_fn
        # Feature 0 is in group A (cost 10)
        assert cost_fn([0]) == group_costs["A"]
        # Feature 4 is in group B (cost 5)
        assert cost_fn([4]) == group_costs["B"]
        # Feature 7 is in group C (cost 1)
        assert cost_fn([7]) == group_costs["C"]

    def test_second_feature_same_group_zero_marginal_cost(self, group_cost_fn):
        cost_fn, _, group_costs = group_cost_fn
        # Both features 0 and 1 are in group A
        assert cost_fn([0]) == cost_fn([0, 1]) == group_costs["A"]
        # All four features of group A cost the same as just one
        assert cost_fn([0, 1, 2, 3]) == group_costs["A"]

    def test_all_features_from_same_group_cost_only_once(self, group_cost_fn):
        cost_fn, _, group_costs = group_cost_fn
        # All group B features: 4, 5, 6
        assert cost_fn([4, 5, 6]) == group_costs["B"]

    def test_cross_group_costs_are_additive(self, group_cost_fn):
        cost_fn, _, group_costs = group_cost_fn
        # One feature from each group
        assert cost_fn([0, 4, 7]) == group_costs["A"] + group_costs["B"] + group_costs["C"]

    def test_all_features_pays_all_group_costs(self, group_cost_fn):
        cost_fn, _, group_costs = group_cost_fn
        total = sum(group_costs.values())
        assert cost_fn(list(range(10))) == total

    def test_order_of_selection_does_not_matter(self, group_cost_fn):
        cost_fn, _, _ = group_cost_fn
        # Reversing order should give the same cost
        assert cost_fn([0, 4, 7]) == cost_fn([7, 4, 0])
        assert cost_fn([1, 2, 5]) == cost_fn([5, 2, 1])

    def test_marginal_cost_of_first_from_group(self, group_cost_fn):
        cost_fn, _, group_costs = group_cost_fn
        # Adding the first feature from group B to an empty selection: marginal = 5
        assert cost_fn([4]) - cost_fn([]) == group_costs["B"]

    def test_marginal_cost_of_second_from_same_group(self, group_cost_fn):
        cost_fn, _, _ = group_cost_fn
        # Adding feature 5 when feature 4 (same group B) is already selected: marginal = 0
        assert cost_fn([4, 5]) - cost_fn([4]) == 0.0

    def test_marginal_cost_of_first_from_new_group(self, group_cost_fn):
        cost_fn, _, group_costs = group_cost_fn
        # Adding feature 7 (group C) when only group A is covered: marginal = 1
        assert cost_fn([0, 7]) - cost_fn([0]) == group_costs["C"]


# ===========================================================================
# 2. CostBasedCFS
# ===========================================================================

class TestCostBasedCFS:

    # --- Basic correctness ---------------------------------------------------

    def test_smoke_classification(self, clf_data):
        X, y = clf_data
        sel = CostBasedCFS()
        sel.fit(X, y)
        out = sel.transform(X)
        assert out.shape[0] == X.shape[0]
        assert 1 <= out.shape[1] <= X.shape[1]

    def test_smoke_regression(self, reg_data):
        X, y = reg_data
        sel = CostBasedCFS()
        sel.fit(X, y)
        out = sel.transform(X)
        assert out.shape[0] == X.shape[0]
        assert 1 <= out.shape[1] <= X.shape[1]

    def test_selected_mask_consistent_with_indices(self, clf_data):
        X, y = clf_data
        sel = CostBasedCFS()
        sel.fit(X, y)
        assert sel.selected_mask_.sum() == len(sel.selected_indices_)
        assert set(np.where(sel.selected_mask_)[0]) == set(sel.selected_indices_)

    def test_n_features_in_set(self, clf_data):
        X, y = clf_data
        sel = CostBasedCFS()
        sel.fit(X, y)
        assert sel.n_features_in_ == X.shape[1]

    # --- Static cost (feature_costs array) -----------------------------------

    def test_high_penalty_prefers_cheap_features(self, clf_data):
        X, y = clf_data
        costs = np.array([100.0] * 5 + [1.0] * 5)
        high = CostBasedCFS(cost_penalty=2.0)
        high.fit(X, y, feature_costs=costs)
        low = CostBasedCFS(cost_penalty=0.0)
        low.fit(X, y, feature_costs=costs)
        cheap = set(range(5, 10))
        assert len(set(high.selected_indices_) & cheap) >= len(set(low.selected_indices_) & cheap)

    def test_budget_respected_static(self, clf_data):
        X, y = clf_data
        costs = np.ones(X.shape[1])
        sel = CostBasedCFS(budget=3.0)
        sel.fit(X, y, feature_costs=costs)
        assert sel.total_cost_ <= 3.0 + 1e-9

    def test_zero_penalty_same_result_regardless_of_costs(self, clf_data):
        X, y = clf_data
        cheap_costs = np.ones(X.shape[1])
        expensive_costs = np.full(X.shape[1], 1000.0)
        sel_cheap = CostBasedCFS(cost_penalty=0.0)
        sel_cheap.fit(X, y, feature_costs=cheap_costs)
        sel_exp = CostBasedCFS(cost_penalty=0.0)
        sel_exp.fit(X, y, feature_costs=expensive_costs)
        # With penalty=0, cost is ignored — same selection expected
        assert sel_cheap.selected_indices_ == sel_exp.selected_indices_

    # --- Group-based cost_fn -------------------------------------------------

    def test_total_cost_equals_cost_fn_of_selected(self, clf_data, group_cost_fn):
        X, y = clf_data
        cost_fn, _, _ = group_cost_fn
        sel = CostBasedCFS(cost_penalty=0.0)
        sel.fit(X, y, cost_fn=cost_fn)
        assert abs(sel.total_cost_ - cost_fn(sel.selected_indices_)) < 1e-9

    def test_total_cost_is_group_cost_not_feature_count(self, clf_data, group_cost_fn):
        """total_cost_ must reflect unique group costs, not the number of features."""
        X, y = clf_data
        cost_fn, feature_to_group, group_costs = group_cost_fn
        sel = CostBasedCFS(cost_penalty=0.0)
        sel.fit(X, y, cost_fn=cost_fn)
        covered_groups = {feature_to_group[i] for i in sel.selected_indices_
                          if i in feature_to_group}
        expected = sum(group_costs[g] for g in covered_groups)
        assert abs(sel.total_cost_ - expected) < 1e-9

    def test_budget_blocks_expensive_groups(self, clf_data, group_cost_fn):
        """With budget=1, only group C (cost=1) is affordable."""
        X, y = clf_data
        cost_fn, feature_to_group, _ = group_cost_fn
        sel = CostBasedCFS(budget=1.0, cost_penalty=0.0)
        sel.fit(X, y, cost_fn=cost_fn)
        assert sel.total_cost_ <= 1.0 + 1e-9
        # All selected features must come from group C
        selected_groups = {feature_to_group.get(i) for i in sel.selected_indices_}
        assert selected_groups <= {"C", None}

    def test_budget_uses_cost_fn_not_feature_costs(self, clf_data, group_cost_fn):
        """When cost_fn is given, budget enforcement ignores feature_costs."""
        X, y = clf_data
        cost_fn, feature_to_group, _ = group_cost_fn
        # Static costs are all 100 — without cost_fn, budget=1 would block everything
        expensive_static = np.full(10, 100.0)
        sel = CostBasedCFS(budget=1.0, cost_penalty=0.0)
        sel.fit(X, y, feature_costs=expensive_static, cost_fn=cost_fn)
        # cost_fn takes over: group C costs 1.0, so features 7-9 are selectable
        assert sel.total_cost_ <= 1.0 + 1e-9
        assert len(sel.selected_indices_) >= 1

    def test_max_features_respected(self, clf_data):
        X, y = clf_data
        sel = CostBasedCFS(max_features=2)
        sel.fit(X, y)
        assert len(sel.selected_indices_) <= 2


# ===========================================================================
# 3. CostConstrainedGBSelector
# ===========================================================================

class TestCostConstrainedGB:

    # --- Basic correctness ---------------------------------------------------

    def test_smoke_classification(self, clf_data):
        X, y = clf_data
        sel = CostConstrainedGBSelector(is_classification=True)
        sel.fit(X, y)
        out = sel.transform(X)
        assert out.shape[0] == X.shape[0]
        assert 1 <= out.shape[1] <= X.shape[1]

    def test_smoke_regression(self, reg_data):
        X, y = reg_data
        sel = CostConstrainedGBSelector(is_classification=False)
        sel.fit(X, y)
        out = sel.transform(X)
        assert out.shape[0] == X.shape[0]
        assert 1 <= out.shape[1] <= X.shape[1]

    def test_selected_mask_consistent_with_indices(self, clf_data):
        X, y = clf_data
        sel = CostConstrainedGBSelector(is_classification=True)
        sel.fit(X, y)
        assert sel.selected_mask_.sum() == len(sel.selected_indices_)
        assert set(np.where(sel.selected_mask_)[0]) == set(sel.selected_indices_)

    # --- Static cost budget --------------------------------------------------

    def test_budget_respected_static(self, clf_data):
        X, y = clf_data
        costs = np.arange(1, 11, dtype=float)
        sel = CostConstrainedGBSelector(is_classification=True, budget=10.0)
        sel.fit(X, y, feature_costs=costs)
        assert sel.total_cost_ <= 10.0 + 1e-9

    # --- Group-based cost_fn -------------------------------------------------

    def test_total_cost_equals_cost_fn_of_selected(self, clf_data, group_cost_fn):
        X, y = clf_data
        cost_fn, _, _ = group_cost_fn
        sel = CostConstrainedGBSelector(is_classification=True)
        sel.fit(X, y, cost_fn=cost_fn)
        assert abs(sel.total_cost_ - cost_fn(sel.selected_indices_)) < 1e-9

    def test_total_cost_is_group_cost_not_feature_count(self, clf_data, group_cost_fn):
        X, y = clf_data
        cost_fn, feature_to_group, group_costs = group_cost_fn
        sel = CostConstrainedGBSelector(is_classification=True)
        sel.fit(X, y, cost_fn=cost_fn)
        covered_groups = {feature_to_group[i] for i in sel.selected_indices_
                          if i in feature_to_group}
        expected = sum(group_costs[g] for g in covered_groups)
        assert abs(sel.total_cost_ - expected) < 1e-9

    def test_budget_respected_with_cost_fn(self, clf_data, group_cost_fn):
        X, y = clf_data
        cost_fn, _, _ = group_cost_fn
        sel = CostConstrainedGBSelector(is_classification=True, budget=11.0)
        sel.fit(X, y, cost_fn=cost_fn)
        assert sel.total_cost_ <= 11.0 + 1e-9

    def test_budget_blocks_expensive_groups(self, clf_data, group_cost_fn):
        """Budget=1 means only group C (cost=1) features are affordable."""
        X, y = clf_data
        cost_fn, feature_to_group, _ = group_cost_fn
        sel = CostConstrainedGBSelector(is_classification=True, budget=1.0)
        sel.fit(X, y, cost_fn=cost_fn)
        assert sel.total_cost_ <= 1.0 + 1e-9
        if sel.selected_indices_:
            selected_groups = {feature_to_group.get(i) for i in sel.selected_indices_}
            assert selected_groups <= {"C", None}

    def test_second_feature_same_group_has_zero_marginal_cost(self, group_cost_fn):
        """Directly verify: once a group is covered, marginal cost of more features is 0."""
        cost_fn, _, _ = group_cost_fn
        # Feature 0 already selected (group A, cost=10)
        selected_so_far = [0]
        current_cost = cost_fn(selected_so_far)  # 10.0
        # Adding feature 1 (also group A)
        marginal = cost_fn(selected_so_far + [1]) - current_cost
        assert marginal == 0.0

    def test_budget_uses_cost_fn_not_feature_costs(self, clf_data, group_cost_fn):
        """cost_fn overrides feature_costs for budget decisions."""
        X, y = clf_data
        cost_fn, _, _ = group_cost_fn
        # Static per-feature costs all 100 → with static budget=1 nothing is affordable
        # With cost_fn, group C (cost=1) is affordable
        expensive_static = np.full(10, 100.0)
        sel = CostConstrainedGBSelector(is_classification=True, budget=1.0)
        sel.fit(X, y, feature_costs=expensive_static, cost_fn=cost_fn)
        assert sel.total_cost_ <= 1.0 + 1e-9
        assert len(sel.selected_indices_) >= 1

    def test_outer_loop_tracks_best_score(self, clf_data):
        """S_best and F_opt are updated across λ iterations."""
        X, y = clf_data
        sel = CostConstrainedGBSelector(is_classification=True, max_iter=5)
        sel.fit(X, y)
        assert hasattr(sel, "best_score_")
        assert hasattr(sel, "best_lambda_")
        assert sel.best_score_ > -np.inf

    def test_max_iter_respected(self, clf_data):
        """With max_iter=1 the loop runs only once and still returns a result."""
        X, y = clf_data
        sel = CostConstrainedGBSelector(is_classification=True, max_iter=1)
        sel.fit(X, y)
        assert len(sel.selected_indices_) >= 1


# ===========================================================================
# 4. MOPSOFeatureSelector
# ===========================================================================

class TestMOPSO:

    # --- Basic correctness ---------------------------------------------------

    def test_smoke_classification(self, clf_data):
        X, y = clf_data
        sel = MOPSOFeatureSelector(n_particles=5, n_iterations=3, random_state=0)
        sel.fit(X, y)
        out = sel.transform(X)
        assert out.shape[0] == X.shape[0]
        assert 1 <= out.shape[1] <= X.shape[1]

    def test_smoke_regression(self, reg_data):
        X, y = reg_data
        sel = MOPSOFeatureSelector(
            n_particles=5, n_iterations=3,
            is_classification=False, random_state=0,
        )
        sel.fit(X, y)
        out = sel.transform(X)
        assert out.shape[0] == X.shape[0]
        assert 1 <= out.shape[1] <= X.shape[1]

    def test_selected_mask_consistent_with_indices(self, clf_data):
        X, y = clf_data
        sel = MOPSOFeatureSelector(n_particles=5, n_iterations=3, random_state=0)
        sel.fit(X, y)
        assert sel.selected_mask_.sum() == len(sel.selected_indices_)
        assert set(np.where(sel.selected_mask_)[0]) == set(sel.selected_indices_)

    def test_pareto_front_populated(self, clf_data):
        X, y = clf_data
        sel = MOPSOFeatureSelector(n_particles=5, n_iterations=3, random_state=0)
        sel.fit(X, y)
        assert len(sel.pareto_front_) >= 1

    def test_pareto_front_sorted_by_cost(self, clf_data):
        X, y = clf_data
        sel = MOPSOFeatureSelector(n_particles=5, n_iterations=3, random_state=0)
        sel.fit(X, y)
        costs_in_front = [entry[1] for entry in sel.pareto_front_]
        assert costs_in_front == sorted(costs_in_front)

    # --- Group-based cost_fn -------------------------------------------------

    def test_total_cost_equals_cost_fn_of_selected(self, clf_data, group_cost_fn):
        X, y = clf_data
        cost_fn, _, _ = group_cost_fn
        sel = MOPSOFeatureSelector(
            n_particles=5, n_iterations=3,
            random_state=0, selection_strategy="min_cost",
        )
        sel.fit(X, y, cost_fn=cost_fn)
        assert abs(sel.total_cost_ - cost_fn(sel.selected_indices_)) < 1e-9

    def test_total_cost_is_group_cost_not_feature_count(self, clf_data, group_cost_fn):
        X, y = clf_data
        cost_fn, feature_to_group, group_costs = group_cost_fn
        sel = MOPSOFeatureSelector(
            n_particles=5, n_iterations=3, random_state=0,
        )
        sel.fit(X, y, cost_fn=cost_fn)
        covered_groups = {feature_to_group.get(i) for i in sel.selected_indices_
                          if i in feature_to_group}
        expected = sum(group_costs[g] for g in covered_groups)
        assert abs(sel.total_cost_ - expected) < 1e-9

    def test_pareto_costs_match_cost_fn(self, clf_data, group_cost_fn):
        """Every entry in pareto_front_ must have cost == cost_fn(its selected features)."""
        X, y = clf_data
        cost_fn, _, _ = group_cost_fn
        sel = MOPSOFeatureSelector(
            n_particles=5, n_iterations=3, random_state=0,
        )
        sel.fit(X, y, cost_fn=cost_fn)
        for perf, cost, mask in sel.pareto_front_:
            indices = list(np.where(mask.astype(bool))[0])
            expected = cost_fn(indices)
            assert abs(cost - expected) < 1e-9, (
                f"Pareto entry has cost={cost}, but cost_fn gives {expected} "
                f"for indices {indices}"
            )

    def test_f1_normalised_within_unit_interval(self, clf_data, group_cost_fn):
        """The normalised cost objective f1 must lie in [0, 1] for all archive entries."""
        X, y = clf_data
        cost_fn, _, _ = group_cost_fn
        max_possible = cost_fn(list(range(10)))  # 10+5+1 = 16
        sel = MOPSOFeatureSelector(
            n_particles=5, n_iterations=3, random_state=0,
        )
        sel.fit(X, y, cost_fn=cost_fn)
        for _, cost, _ in sel.pareto_front_:
            f1 = cost / max_possible
            assert 0.0 <= f1 <= 1.0 + 1e-9

    def test_selection_strategies_all_return_result(self, clf_data):
        X, y = clf_data
        for strategy in ("min_cost", "max_perf", "knee"):
            sel = MOPSOFeatureSelector(
                n_particles=5, n_iterations=3,
                selection_strategy=strategy, random_state=0,
            )
            sel.fit(X, y)
            assert len(sel.selected_indices_) >= 1

    def test_budget_respected_when_feasible_solution_exists(self, clf_data, group_cost_fn):
        """MOPSO budget filters the Pareto front at selection time (not during search).

        If at least one Pareto entry is within budget, the selected solution must
        also be within budget.  If the search found no feasible entry (possible with
        few particles/iterations), the algorithm falls back to the full front — no
        budget guarantee in that case.
        """
        X, y = clf_data
        cost_fn, _, _ = group_cost_fn
        budget = 5.0
        sel = MOPSOFeatureSelector(
            n_particles=5, n_iterations=3,
            budget=budget, selection_strategy="min_cost", random_state=0,
        )
        sel.fit(X, y, cost_fn=cost_fn)
        feasible = [e for e in sel.pareto_front_ if e[1] <= budget]
        if feasible:
            assert sel.total_cost_ <= budget + 1e-9
