# implemented after paper: https://doi.org/10.1109/TCBB.2015.2476796
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
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


def _dominates(f1_a: float, f2_a: float, f1_b: float, f2_b: float) -> bool:
    return (f1_a <= f1_b and f2_a <= f2_b) and (f1_a < f1_b or f2_a < f2_b)


def _crowding_distances(f1_vals: list[float], f2_vals: list[float]) -> np.ndarray:
    n = len(f1_vals)
    distances = np.zeros(n)
    if n <= 2:
        distances[:] = np.inf
        return distances
    for obj_vals in (np.asarray(f1_vals, dtype=float), np.asarray(f2_vals, dtype=float)):
        order = np.argsort(obj_vals)
        obj_range = obj_vals[order[-1]] - obj_vals[order[0]]
        if obj_range == 0.0:
            obj_range = 1e-10
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        for k in range(1, n - 1):
            distances[order[k]] += (
                obj_vals[order[k + 1]] - obj_vals[order[k - 1]]
            ) / obj_range
    return distances





class MOPSOFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        n_particles: int = 30,
        n_iterations: int = 100,
        w: float = 0.4,
        v_max: float = 1.0,
        is_classification: bool = True,
        archive_max_size: int = 30,
        budget: float | None = None,
        selection_strategy: str = "knee",
        random_state: int = 42,
        n_jobs: int = 1,
        jump_prob: float = 0.01,
        reinit_ratio: float = 0.10,
    ):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.v_max = v_max
        self.is_classification = is_classification
        self.archive_max_size = archive_max_size
        self.budget = budget
        self.selection_strategy = selection_strategy
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.jump_prob = jump_prob
        self.reinit_ratio = reinit_ratio

    def fit(
        self,
        X,
        y,
        feature_costs=None,
        cost_fn: Callable[[list[int]], float] | None = None,
    ) -> "MOPSOFeatureSelector":
        X_arr, col_names = _to_numpy(X)
        y_arr = _to_numpy_y(y)
        n_features = X_arr.shape[1]
        self.n_features_in_ = n_features

        costs = (
            np.ones(n_features, dtype=np.float64)
            if feature_costs is None
            else np.asarray(feature_costs, dtype=np.float64)
        )

        if cost_fn is not None:
            max_possible_cost = cost_fn(list(range(n_features))) or 1.0
        else:
            max_cost = costs.max()
            u = costs / max_cost if max_cost > 0 else np.ones(n_features)
            sum_u = float(u.sum()) or 1.0

        rng = np.random.default_rng(self.random_state)

        X_eval = X_arr
        y_eval = y_arr

        def _decode(pos: np.ndarray) -> np.ndarray:
            # equation 6
            mask = (pos >= 0.5).astype(np.float64)
            if mask.sum() == 0:
                mask[int(np.argmax(pos))] = 1.0
            return mask

        _eval_cache: dict[tuple, tuple] = {}

        def _evaluate(mask: np.ndarray) -> tuple[float, float, float, float]:
            sol = mask.astype(int)
            if sol.sum() == 0:
                return 0.0, 1.0, 1.0, 0.0

            selected = np.where(sol)[0]
            cache_key = tuple(selected.tolist())
            if cache_key in _eval_cache:
                return _eval_cache[cache_key]

            X_sel = X_eval[:, selected]

            try:
                nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
                nn.fit(X_sel)
                _, idx = nn.kneighbors(X_sel)
                nn_idx = idx[:, 1]
                y_pred = y_eval[nn_idx]
                if self.is_classification:
                    perf = float(np.mean(y_pred == y_eval))
                else:
                    ss_res = float(np.sum((y_eval - y_pred) ** 2))
                    ss_tot = float(np.sum((y_eval - y_eval.mean()) ** 2))
                    perf = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            except Exception:
                perf = 0.0

            f2 = 1.0 - perf

            if cost_fn is not None:
                total_c = cost_fn(selected.tolist())
                f1 = total_c / max_possible_cost
            else:
                total_c = float((sol * costs).sum())
                f1 = float((sol * u).sum()) / sum_u

            result = (perf, f1, f2, total_c)
            _eval_cache[cache_key] = result
            return result


        # Init swarm
        positions = rng.uniform(0.0, 1.0, size=(self.n_particles, n_features))
        velocities = rng.uniform(
            -self.v_max / 2.0, self.v_max / 2.0,
            size=(self.n_particles, n_features),
        )

        pbest_pos = positions.copy()
        pbest_f1 = np.full(self.n_particles, np.inf)
        pbest_f2 = np.full(self.n_particles, np.inf)

        archive: list[tuple] = []


        entries: list[tuple] = []
        for i in range(self.n_particles):
            mask = _decode(positions[i])
            perf, f1, f2, total_c = _evaluate(mask)
            pbest_f1[i] = f1
            pbest_f2[i] = f2
            entries.append((perf, f1, f2, total_c, mask.copy(), positions[i].copy()))

        prev_archive_sig: frozenset | None = None
        stagnant = 0
        for t in range(self.n_iterations):

            archive = self._rebuild_archive(archive + entries)

            archive_sig = frozenset(tuple(e[4].tolist()) for e in archive)
            if archive_sig == prev_archive_sig:
                stagnant += 1
                if stagnant >= 5:
                    break
            else:
                stagnant = 0
                prev_archive_sig = archive_sig

            for i, (perf, f1, f2, total_c, mask, pos) in enumerate(entries):
                if not _dominates(pbest_f1[i], pbest_f2[i], f1, f2):
                    pbest_pos[i] = pos.copy()
                    pbest_f1[i] = f1
                    pbest_f2[i] = f2


            c1 = 2.5 - 2.0 * t / max(self.n_iterations, 1)
            c2 = 0.5 + 2.0 * t / max(self.n_iterations, 1)
            iter_dists = (
                _crowding_distances([e[1] for e in archive], [e[2] for e in archive])
                if len(archive) > 1 else None
            )
            for i in range(self.n_particles):
                gbest_i = self._select_guide(archive, rng, n_features, iter_dists)
                r1 = rng.uniform(0.0, 1.0, size=n_features)
                r2 = rng.uniform(0.0, 1.0, size=n_features)
                velocities[i] = (
                    self.w * velocities[i]
                    + c1 * r1 * (pbest_pos[i] - positions[i])
                    + c2 * r2 * (gbest_i - positions[i])
                )
            velocities = np.clip(velocities, -self.v_max, self.v_max)
            positions = np.clip(positions + velocities, 0.0, 1.0)


            jump_mask = rng.uniform(0.0, 1.0, size=(self.n_particles, n_features)) < self.jump_prob
            positions = np.where(jump_mask, rng.uniform(0.0, 1.0, size=positions.shape), positions)
            # reinit velocities of 10% of particles
            n_reinit = max(1, int(self.reinit_ratio * self.n_particles))
            reinit_idx = rng.choice(self.n_particles, size=n_reinit, replace=False)
            velocities[reinit_idx] = rng.uniform(
                -self.v_max, self.v_max, size=(n_reinit, n_features)
            )

            entries = []
            for i in range(self.n_particles):
                mask = _decode(positions[i])
                perf, f1, f2, total_c = _evaluate(mask)
                entries.append((perf, f1, f2, total_c, mask.copy(), positions[i].copy()))

        self.pareto_front_ = sorted(
            [(e[0], e[3], e[4]) for e in archive],
            key=lambda x: x[1],
        )

        chosen_mask = self._select_solution(self.pareto_front_, self.budget, costs)
        self.selected_mask_ = chosen_mask.astype(bool)
        self.selected_indices_ = list(np.where(self.selected_mask_)[0])
        self.feature_names_in_ = col_names
        self.total_cost_ = (
            cost_fn(self.selected_indices_)
            if cost_fn is not None
            else float((chosen_mask * costs).sum())
        )
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        X_arr, _ = _to_numpy(X)
        out = X_arr[:, self.selected_indices_]
        if isinstance(X, pd.DataFrame):
            cols = [X.columns[i] for i in self.selected_indices_]
            return pd.DataFrame(out, index=X.index, columns=cols)
        return out

    def _rebuild_archive(self, entries: list) -> list:
        n = len(entries)
        if n == 0:
            return []
        dominated = [False] * n
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if _dominates(entries[j][1], entries[j][2], entries[i][1], entries[i][2]):
                    dominated[i] = True
                    break
        non_dom = [e for i, e in enumerate(entries) if not dominated[i]]
        while len(non_dom) > self.archive_max_size:
            dists = _crowding_distances([e[1] for e in non_dom], [e[2] for e in non_dom])
            non_dom.pop(int(np.argmin(dists)))
        return non_dom

    def _select_guide(
        self,
        archive: list,
        rng: np.random.Generator,
        n_features: int,
        dists: np.ndarray | None = None,
    ) -> np.ndarray:
        if not archive:
            return np.full(n_features, 0.5, dtype=np.float64)
        if len(archive) == 1:
            return archive[0][5].astype(np.float64)
        if dists is None:
            dists = _crowding_distances([e[1] for e in archive], [e[2] for e in archive])
        idx_a, idx_b = rng.choice(len(archive), size=2, replace=False)
        winner = idx_a if dists[idx_a] >= dists[idx_b] else idx_b
        return archive[winner][5].astype(np.float64)

    def _select_solution(self, pareto_front: list, budget: float | None, costs: np.ndarray) -> np.ndarray:
        candidates = pareto_front
        if budget is not None:
            feasible = [t for t in pareto_front if t[1] <= budget]
            if feasible:
                candidates = feasible
        if not candidates:
            return np.zeros(self.n_features_in_, dtype=float)

        match self.selection_strategy:
            case "max_perf":
                return max(candidates, key=lambda t: t[0])[2].astype(float)
            case "knee":
                if len(candidates) == 1:
                    return candidates[0][2].astype(float)
                sorted_c = sorted(candidates, key=lambda t: t[1])
                p_arr = np.array([t[0] for t in sorted_c])
                c_arr = np.array([t[1] for t in sorted_c])
                p_range = p_arr.max() - p_arr.min()
                c_range = c_arr.max() - c_arr.min()
                if p_range == 0 or c_range == 0:
                    return sorted_c[0][2].astype(float)
                p_norm = (p_arr - p_arr.min()) / p_range
                c_norm = (c_arr - c_arr.min()) / c_range
                line = np.array([c_norm[-1] - c_norm[0], p_norm[-1] - p_norm[0]])
                line_len = np.linalg.norm(line)
                if line_len == 0:
                    return sorted_c[0][2].astype(float)
                unit = line / line_len
                perp_dists = []
                for i in range(len(sorted_c)):
                    pt = np.array([c_norm[i] - c_norm[0], p_norm[i] - p_norm[0]])
                    perp = pt - np.dot(pt, unit) * unit
                    perp_dists.append(np.linalg.norm(perp))
                return sorted_c[int(np.argmax(perp_dists))][2].astype(float)
            case _:  #min_cost is default
                return min(candidates, key=lambda t: t[1])[2].astype(float)
