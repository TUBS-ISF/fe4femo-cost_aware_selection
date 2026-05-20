# implementation based on paper: https://doi.org/10.1016/j.engappai.2025.112751

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.utils.validation import check_is_fitted


# helpers

def _to_numpy(X):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float), list(X.columns)
    return np.asarray(X, dtype=float), None


def _to_numpy_y(y):
    if isinstance(y, pd.Series):
        return y.to_numpy()
    return np.asarray(y)



class CostConstrainedGBSelector(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        budget: float | None = None,
        max_features: int | None = None,
        is_classification: bool = True,
        gb_params: dict | None = None,
        lambda_0: float = 0.0,
        delta: float | None = None,
        max_iter: int = 50,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        self.budget = budget
        self.max_features = max_features
        self.is_classification = is_classification
        self.gb_params = gb_params
        self.lambda_0 = lambda_0
        self.delta = delta
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs



    def fit(
        self,
        X,
        y,
        feature_costs=None,
        cost_fn: Callable[[list[int]], float] | None = None,
    ) -> "CostConstrainedGBSelector":
        X_arr, col_names = _to_numpy(X)
        y_arr = _to_numpy_y(y)
        n_features = X_arr.shape[1]

        costs = (
            np.ones(n_features, dtype=np.float64)
            if feature_costs is None
            else np.asarray(feature_costs, dtype=np.float64)
        )

        feature_names: list[str] = (
            col_names if col_names is not None
            else [f"f{i}" for i in range(n_features)]
        )
        safe_names = [f"f{i}" for i in range(n_features)]
        X_df = pd.DataFrame(X_arr, columns=safe_names)

        costs_rank = (
            np.array([cost_fn([i]) for i in range(n_features)], dtype=np.float64)
            if cost_fn is not None
            else costs.copy()
        )



        S_best = -np.inf
        F_opt: list[int] = []
        best_lambda = self.lambda_0
        best_model = None
        prev_score: float | None = None
        lambda_val = self.lambda_0
        lambda_max: float = 0.0
        delta: float = 0.0

        for iteration in range(self.max_iter):
            model, total_gain, weight, lambda_max = self._inner_loop(
                X_df, y_arr, lambda_val, costs_rank
            )

            if iteration == 0:
                if self.delta is not None:
                    delta = self.delta
                elif lambda_max > self.lambda_0 and self.max_iter > 1:
                    delta = (lambda_max - self.lambda_0) / (self.max_iter - 1)

            J = total_gain - lambda_val * costs_rank * weight
            rank = np.argsort(J)[::-1]

            selected = self._select(rank, J, costs, cost_fn, n_features)

            score = self._evaluate(X_arr, y_arr, selected)

            if score > S_best:
                S_best = score
                F_opt = selected
                best_lambda = lambda_val
                best_model = model

            converged = prev_score is not None and abs(score - prev_score) < 1e-6
            if lambda_val >= lambda_max - 1e-9 or converged or delta == 0.0:
                break

            prev_score = score
            lambda_val = min(lambda_val + delta, lambda_max)

        # Store Results
        self.selected_indices_ = F_opt
        self.selected_mask_ = np.zeros(n_features, dtype=bool)
        if F_opt:
            self.selected_mask_[F_opt] = True
        self.feature_names_in_ = feature_names
        self.n_features_in_ = n_features
        self.total_cost_ = (
            cost_fn(F_opt) if cost_fn is not None
            else float(costs[F_opt].sum())
        ) if F_opt else 0.0
        self.importance_scores_ = (
            _feature_scores(best_model, "gain")
            if best_model is not None else np.zeros(n_features)
        )
        self.best_score_ = S_best
        self.best_lambda_ = best_lambda
        self.gb_model_ = best_model
        return self


    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        X_arr, _ = _to_numpy(X)
        out = X_arr[:, self.selected_indices_]
        if isinstance(X, pd.DataFrame):
            cols = [X.columns[i] for i in self.selected_indices_]
            return pd.DataFrame(out, index=X.index, columns=cols)
        return out


    # Inner loop is line 3 to 11 from pseudo code given in the paper
    def _inner_loop(
        self,
        X_df: pd.DataFrame,
        y_arr: np.ndarray,
        lambda_val: float,
        costs_rank: np.ndarray,
    ) -> tuple:
        params = dict(self.gb_params or {})
        params["random_state"] = self.random_state
        params["n_jobs"] = self.n_jobs
        params["verbosity"] = -1
        params["cegb_tradeoff"] = lambda_val
        params["cegb_penalty_feature_lazy"] = costs_rank.tolist()

        clf = (
            lgb.LGBMClassifier(**params)
            if self.is_classification
            else lgb.LGBMRegressor(**params)
        )
        clf.fit(X_df, y_arr)

        total_gain = _feature_scores(clf, "gain")
        weight = _feature_scores(clf, "split")
        lambda_max = _compute_lambda_max(total_gain, weight, costs_rank)

        return clf, total_gain, weight, lambda_max

    # Line 13-20, the greedy selection of features
    def _select(
        self,
        rank: np.ndarray,
        J: np.ndarray,
        costs: np.ndarray,
        cost_fn: Callable[[list[int]], float] | None,
        n_features: int,
    ) -> list[int]:
        max_k = self.max_features if self.max_features is not None else n_features
        selected: list[int] = []

        if cost_fn is not None:
            remaining = list(rank)
            current_cost = 0.0
            while remaining and len(selected) < max_k:
                best_idx, best_j = None, -np.inf
                for idx in remaining:
                    marginal = cost_fn(selected + [int(idx)]) - current_cost
                    if self.budget is not None and current_cost + marginal > self.budget:
                        continue
                    if J[idx] > best_j:
                        best_j = J[idx]
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(int(best_idx))
                current_cost = cost_fn(selected)
                remaining.remove(best_idx)
        else:
            cum_cost = 0.0
            for idx in rank:
                if len(selected) >= max_k:
                    break
                if self.budget is not None and cum_cost + costs[idx] > self.budget:
                    break
                selected.append(int(idx))
                cum_cost += float(costs[idx])

        return selected

    # The evaluation
    def _evaluate(
        self,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        selected: list[int],
    ) -> float:
        if not selected:
            return -np.inf
        X_sel = X_arr[:, selected]
        try:
            clf = (
                LogisticRegression(max_iter=1000, random_state=self.random_state)
                if self.is_classification
                else Ridge()
            )
            clf.fit(X_sel, y_arr)
            return float(clf.score(X_sel, y_arr))
        except Exception:
            return -np.inf

# More helpers
def _feature_scores(model, importance_type: str) -> np.ndarray:
    return np.array(
        model.booster_.feature_importance(importance_type=importance_type),
        dtype=np.float64,
    )


def _compute_lambda_max(
    total_gain: np.ndarray,
    weight: np.ndarray,
    costs: np.ndarray,
) -> float:
    weighted_gain = total_gain * weight

    min_cost = costs.min()
    cheapest_mask = np.abs(costs - min_cost) < 1e-10
    if not cheapest_mask.any():
        return 0.0

    best_cheap_idx = weighted_gain[cheapest_mask].argmax()
    gain_e = weighted_gain[cheapest_mask][best_cheap_idx]
    cost_e_w = min_cost * weight[cheapest_mask][best_cheap_idx]

    lambda_max = 0.0
    for k in range(len(costs)):
        if costs[k] <= min_cost + 1e-10:
            continue
        cost_k_w = costs[k] * weight[k]
        denom = cost_k_w - cost_e_w
        if denom > 1e-10:
            threshold = (weighted_gain[k] - gain_e) / denom
            if threshold > lambda_max:
                lambda_max = threshold

    return max(lambda_max, 0.0)
