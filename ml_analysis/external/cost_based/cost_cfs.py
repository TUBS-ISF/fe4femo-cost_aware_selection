# See paper https://doi.org/10.1016/j.patcog.2014.01.008 for context
from __future__ import annotations

import heapq
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted



def _to_numpy(X):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float), list(X.columns)
    return np.asarray(X, dtype=float), None


def _to_numpy_y(y):
    if isinstance(y, pd.Series):
        return y.to_numpy()
    return np.asarray(y)





class CostBasedCFS(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        budget: float | None = None,
        cost_penalty: float = 0.1,
        max_features: int | None = None,
        random_state: int = 42,
    ):
        self.budget = budget
        self.cost_penalty = cost_penalty
        self.max_features = max_features
        self.random_state = random_state


    def fit(
        self,
        X,
        y,
        feature_costs=None,
        cost_fn: Callable[[list[int]], float] | None = None,
    ) -> "CostBasedCFS":
        X_arr, col_names = _to_numpy(X)
        y_arr = _to_numpy_y(y)
        n_samples, n_features = X_arr.shape

        costs = (
            np.ones(n_features, dtype=np.float64)
            if feature_costs is None
            else np.asarray(feature_costs, dtype=np.float64)
        )
        self._cost_fn = cost_fn

        corr_with_class = np.corrcoef(X_arr.T, y_arr)[-1, :-1]
        r_cf = np.where(np.isnan(corr_with_class), 0.0, np.abs(corr_with_class))

        feat_corr = np.corrcoef(X_arr.T)
        r_ff = np.where(np.isnan(feat_corr), 0.0, np.abs(feat_corr))

        selected = self._best_first_search(r_cf, r_ff, costs, n_features)

        # Store Results
        self.selected_indices_ = selected
        self.selected_mask_ = np.zeros(n_features, dtype=bool)
        if selected:
            self.selected_mask_[selected] = True
        self.feature_names_in_ = col_names
        self.n_features_in_ = n_features
        self.total_cost_ = (
            cost_fn(selected) if cost_fn is not None else float(costs[selected].sum())
        ) if selected else 0.0
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        X_arr, _ = _to_numpy(X)
        out = X_arr[:, self.selected_indices_]
        if isinstance(X, pd.DataFrame):
            cols = [X.columns[i] for i in self.selected_indices_]
            return pd.DataFrame(out, index=X.index, columns=cols)
        return out



    def _best_first_search(
        self,
        r_cf: np.ndarray,
        r_ff: np.ndarray,
        costs: np.ndarray,
        n_features: int,
    ) -> list[int]:
        
        max_k = self.max_features if self.max_features is not None else n_features

        # Min-heap entries: (neg_merit, tiebreak, frozenset_of_indices)
        heap: list = [(0.0, 0, frozenset())]
        visited: set[frozenset] = {frozenset()}
        tiebreak = 1

        best_global_merit = -np.inf
        best_global_subset: list[int] = []
        no_improve_count = 0

        while heap:
            if no_improve_count >= 5:
                break

            _, _, current_frozen = heapq.heappop(heap)
            current_subset = sorted(current_frozen)

            # Cannot expand beyond max_k – skip without counting
            if len(current_subset) >= max_k:
                continue

            remaining = [f for f in range(n_features) if f not in current_frozen]
            expansion_improved = False

            for f in remaining:
                candidate_frozen = current_frozen | {f}
                if candidate_frozen in visited:
                    continue
                visited.add(candidate_frozen)

                candidate = sorted(candidate_frozen)

                candidate_cost = (
                    self._cost_fn(candidate)
                    if self._cost_fn is not None
                    else float(costs[candidate].sum())
                )
                if self.budget is not None and candidate_cost > self.budget:
                    continue

                merit = self._compute_merit(candidate, r_cf, r_ff, costs, candidate_cost)
                heapq.heappush(heap, (-merit, tiebreak, candidate_frozen))
                tiebreak += 1

                if merit > best_global_merit:
                    best_global_merit = merit
                    best_global_subset = candidate
                    expansion_improved = True

            if expansion_improved:
                no_improve_count = 0
            else:
                no_improve_count += 1

        return best_global_subset





    def _compute_merit(
        self,
        subset: list[int],
        r_cf: np.ndarray,
        r_ff: np.ndarray,
        costs: np.ndarray,
        precomputed_cost: float | None = None,
    ) -> float:
        k = len(subset)
        if k == 0:
            return -np.inf

        mean_r_cf = float(r_cf[subset].mean())

        if k == 1:
            mean_r_ff = 0.0
        else:
            pair_vals = [
                r_ff[subset[i], subset[j]]
                for i in range(k)
                for j in range(i + 1, k)
            ]
            mean_r_ff = float(np.mean(pair_vals))

        denominator = np.sqrt(k + k * (k - 1) * mean_r_ff)
        base_merit = (k * mean_r_cf) / denominator if denominator > 0 else 0.0

        if precomputed_cost is not None:
            subset_cost = precomputed_cost
        elif self._cost_fn is not None:
            subset_cost = self._cost_fn(subset)
        else:
            subset_cost = float(costs[subset].sum())
        mean_cost = subset_cost / k
        return base_merit - self.cost_penalty * mean_cost
