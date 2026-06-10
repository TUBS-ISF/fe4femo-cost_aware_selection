from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


@dataclass(frozen=True, eq=True)
class FoldSplit:
    fold_no: int
    train_index: np.ndarray
    test_index: np.ndarray

@dataclass(frozen=True, eq=True)
class FoldResult:
    model_quality: float
    feature_computation_time: float

@dataclass(frozen=True)
class TrialContainer:
    model: BaseEstimator
    x_test: pd.DataFrame
    best_params: dict[str, any]
    time_Feature: float
    time_Model: float
    pareto_front_: list[tuple[float, float]] | None = field(default=None, hash=False, compare=False)