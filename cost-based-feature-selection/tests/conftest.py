# AI generated: Claude Code (Sonnet 4.6)
"""Shared fixtures for cost-based feature selection tests."""
import sys
import os

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

# Algorithms live in ml_analysis/external/cost_based/ (imported as a package)
_ml_analysis_dir = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ml_analysis")
)
sys.path.insert(0, _ml_analysis_dir)


@pytest.fixture
def clf_data():
    """20 samples, 10 features, binary classification."""
    X, y = make_classification(
        n_samples=40, n_features=10, n_informative=5,
        n_redundant=2, random_state=0,
    )
    return X, y


@pytest.fixture
def reg_data():
    """20 samples, 10 features, regression."""
    X, y = make_regression(n_samples=40, n_features=10, n_informative=5, random_state=0)
    return X, y


@pytest.fixture
def group_cost_fn():
    """Cost function for 10 features split into 3 groups.

    Groups:
      group_A: features 0, 1, 2, 3  (cost 10.0)
      group_B: features 4, 5, 6     (cost 5.0)
      group_C: features 7, 8, 9     (cost 1.0)

    First feature from a group pays the full group cost; additional
    features from the same group cost 0.
    """
    feature_to_group = {
        0: "A", 1: "A", 2: "A", 3: "A",
        4: "B", 5: "B", 6: "B",
        7: "C", 8: "C", 9: "C",
    }
    group_costs = {"A": 10.0, "B": 5.0, "C": 1.0}

    def cost_fn(subset: list[int]) -> float:
        covered: set[str] = set()
        total = 0.0
        for i in subset:
            g = feature_to_group.get(i)
            if g is not None and g not in covered:
                covered.add(g)
                total += group_costs[g]
        return total

    return cost_fn, feature_to_group, group_costs
