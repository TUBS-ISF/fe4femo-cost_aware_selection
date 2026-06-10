import itertools
import math
from dataclasses import dataclass
from statistics import mean
from typing import Any

import dask.distributed
import joblib
import numpy as np
import pandas as pd
from distributed import worker_client, Variable
from mrmr import mrmr_classif, mrmr_regression
from optuna import Trial
from functools import partial

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, \
    SelectFromModel, VarianceThreshold, RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import matthews_corrcoef, d2_absolute_error_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
# from zoofs import HarrisHawkOptimization, GeneticOptimization

from external.HFMOEA.main import reduceFeaturesMaxAcc, compute_sol
from external.genetic_parallel import GeneticParallel
from external.harrishawk_parallel import HarrisHawkParallel
from external.multisurf_parallel import MultiSURF_Parallel
from external.skfeature.NDFS import ndfs
from external.skfeature.sparse_learning import feature_ranking
from external.svd_entropy import keep_high_contrib_features
from helper.data_classes import FoldSplit
from helper.load_dataset import filter_SATzilla, filter_SATfeatPy, filter_FMBA, filter_FMChara
from helper.model_training import is_model_classifier
from external.cost_based.cost_cfs import CostBasedCFS
from external.cost_based.cost_constrained_gb import CostConstrainedGBSelector
from external.cost_based.mopso import MOPSOFeatureSelector


def _make_cost_fn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    group_dict: dict[str, list[str]],
    feature_group_times: pd.DataFrame | None,
):
    """
    This code does the calc for feature costs like 
    eval_feature_group_runtime in generate_fold_model.py.
    
    But it works with indices,and I first overlooked the 
    eval_ function and already used this in the algorithms.
    """
    if feature_group_times is None:
        return None

    instances = list(set(X_train.index.values).union(set(X_test.index.values)))
    group_mean_cost: dict[str, float] = {
        g: float(feature_group_times.loc[instances, g].mean())
        for g in feature_group_times.columns
        if g in feature_group_times.columns
    }

    col_names = list(X_train.columns)
    feature_to_group: dict[str, str] = {
        feat: group
        for group, feats in group_dict.items()
        for feat in feats
    }

    def cost_fn(subset: list[int]) -> float:
        covered = {feature_to_group[col_names[i]]
                   for i in subset
                   if col_names[i] in feature_to_group}
        return sum(group_mean_cost.get(g, 0.0) for g in covered)

    return cost_fn


def transform_dict_to_var_dict(dictionary : dict) -> dict:
    ret_dict = {}
    with worker_client() as client:
        for k,v in dictionary.items():
            var = Variable()
            future = client.scatter(v, direct=True)
            var.set(future)
            ret_dict[k] = var
    return ret_dict



def objective_function_zoo(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if is_model_classifier(model):
        return matthews_corrcoef(y_test, y_pred)
    else:
        return d2_absolute_error_score(y_test, y_pred)


def prefilter_features(X_train_in : pd.DataFrame, X_test_in : pd.DataFrame, y_train : pd.Series, threshold : float):
    variance_filter = VarianceThreshold()
    variance_filter.set_output(transform="pandas")
    X_train = variance_filter.fit_transform(X_train_in)
    X_test = variance_filter.transform(X_test_in)

    # https://stackoverflow.com/a/44674459 + own correlation to target
    col_corr = set()
    corr_solution = X_train.corrwith(y_train, axis=0, method="spearman").abs()
    corr_matrix = X_train.corr(method="spearman").abs()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if(corr_matrix.iloc[i, j]) >= threshold:
                if corr_solution.iat[i] > corr_solution.iat[j] and (corr_matrix.columns[i] not in col_corr):
                    col_corr.add(corr_matrix.columns[j])
                if corr_solution.iat[j] > corr_solution.iat[i] and (corr_matrix.columns[j] not in col_corr):
                    col_corr.add(corr_matrix.columns[i])
    to_keep = set(X_train.columns) - set(col_corr)
    return X_train[list(to_keep)], X_test[list(to_keep)]

def impute_and_scale(X_train, X_test):
    imputer = SimpleImputer(keep_empty_features=False, missing_values=pd.NA)
    scaler = StandardScaler()

    imputer.set_output(transform="pandas")
    scaler.set_output(transform="pandas")

    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)

    X_test = imputer.transform(X_test)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def precompute_feature_selection(features: str, isClassification : bool, X_train_orig : pd.DataFrame, X_test_orig : pd.DataFrame, y_train : pd.Series, y_test : pd.Series, model_flatness : pd.Series, threshold : float = .9, parallelism : int = 1, feature_group_times : pd.DataFrame | None = None, ):
    X_train_imputed, X_test_imputed = impute_and_scale(X_train_orig, X_test_orig)
    if features == "all": # do not prefilter for all
        return {
            "X_train": X_train_imputed,
            "X_test": X_test_imputed,
            "y_train": y_train,
            "y_test" : y_test,
        }
    X_train, X_test = prefilter_features(X_train_imputed, X_test_imputed, y_train, threshold)  # todo leaking?
    ret_dict = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test" : y_test,
    }

    if features in ["harris-hawks", "genetic", "HFMOEA"]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        ret_dict["fold_no"] = cv.n_splits
        for i, (train_index, test_index) in enumerate(cv.split(X_train, model_flatness.loc[model_flatness.index.isin(y_train.index)])):
            ret_dict[f"index_{i}"] = FoldSplit(fold_no=i, train_index=train_index, test_index=test_index)

    match features:
        case "SATzilla":
            ret_dict["X_train"] = filter_SATzilla(X_train)
            ret_dict["X_test"] = filter_SATzilla(X_test)
        case "SATfeatPy":
            ret_dict["X_train"] = filter_SATfeatPy(X_train)
            ret_dict["X_test"] = filter_SATfeatPy(X_test)
        case "FMBA":
            ret_dict["X_train"] = filter_FMBA(X_train)
            ret_dict["X_test"] = filter_FMBA(X_test)
        case "FM_Chara":
            ret_dict["X_train"] = filter_FMChara(X_train)
            ret_dict["X_test"] = filter_FMChara(X_test)
        case "prefilter":
            pass
        case "kbest-mutalinfo":
            pass
        case "multisurf":
            with joblib.parallel_config(backend="dask"):
                selector = MultiSURF_Parallel(n_jobs=parallelism)
                selector.fit(X_train.to_numpy(), y_train.to_numpy())
            ret_dict["top_features"] = selector.top_features_
        case "mRMR":
            pass
        case "RFE":
            pass
        case "harris-hawks" | "genetic":
            pass
        case "HFMOEA":
            sol = compute_sol(X_train.to_numpy(), y_train.to_numpy(), isClassification, parallelism)
            ret_dict["sol"] = sol
        case "embedded-tree":
            pass
        case "SVD-entropy":
            with joblib.parallel_config(backend="dask"):
                mask = keep_high_contrib_features(X_train, parallelism)
            ret_dict["mask"] = pd.Series(mask)
        case "NDFS":
            pass
        case "optuna-combined":
            pass
        case "cost-cfs" | "cost-gb" | "mopso":
            pass
        case _:
            raise ValueError("Invalid Feature Subset")
    return ret_dict

def set_njobs_if_possible(estimator, n_jobs:int):
    try:
        estimator.set_params(n_jobs=n_jobs)
    except ValueError: #catch estimators without n_jobs_arg
        pass
    return estimator

def extract_fold_list(precomputed : dict) -> list[Variable]:
    if "fold_no" in precomputed.keys():
        fold_no = precomputed["fold_no"].get().result()
        return [precomputed[f"index_{i}"] for i in range(fold_no)]
    else:
        raise Exception("No folds in precomputed!")

def get_feature_selection(precomputed:dict, features : str, isClassification : bool, selector_args, estimator, group_dict : dict[str, list[str]], parallelism : int = 1, verbose = False, dask_parallel : bool = False, feature_group_times : pd.DataFrame | None = None):
    y_train = precomputed["y_train"].get().result()
    X_train = precomputed["X_train"].get().result()
    X_test = precomputed["X_test"].get().result()
    max_features = X_train.shape[1]
    match features:
        case "all" | "SATzilla" | "SATfeatPy" | "FMBA" | "FM_Chara" | "prefilter":
            return X_train, X_test, None
        case "kbest-mutalinfo":
            score_func = partial(mutual_info_classif, random_state=42, n_jobs=parallelism, n_neighbors=selector_args["n_neighbors"] ) if isClassification else partial(mutual_info_regression, random_state=42, n_jobs=parallelism, n_neighbors=selector_args["n_neighbors"])
            selector = SelectKBest(score_func, k=min(max_features, selector_args["k"]))# limit to max feature count after preprocessing
            selector.set_output(transform="pandas")
            selector.fit(X_train, y_train)
            return selector.transform(X_train), selector.transform(X_test), None
        case "multisurf":
            top_features =  precomputed["top_features"].get().result()
            row_index_to_select = top_features[:min(max_features, selector_args["n_features_to_select"])]# limit to max feature count after preprocessing
            return X_train.iloc[:, row_index_to_select], X_test.iloc[:, row_index_to_select], None
        case "mRMR":
            selector_args["K"] = min(max_features, selector_args["K"]) # limit to max feature count after preprocessing
            selected_feature_names = mrmr_classif(X_train, y_train, **selector_args, n_jobs=parallelism, show_progress=False) if isClassification else mrmr_regression(X_train, y_train, **selector_args, n_jobs=parallelism, show_progress=False)
            return X_train[selected_feature_names], X_test[selected_feature_names], None
        case "RFE":
            set_njobs_if_possible(estimator, parallelism)
            selector_args["n_features_to_select"] = min(max_features, selector_args["n_features_to_select"])  # limit to max feature count after preprocessing
            selector = RFE(estimator, step=selector_args["step"], n_features_to_select=selector_args["n_features_to_select"])
            selector.set_output(transform="pandas")
            selector.fit(X_train, y_train)
            return selector.transform(X_train), selector.transform(X_test), None
        case "harris-hawks":
            #deactivated
            raise NotImplementedError() # if reactivating --> implement seed for pseudo-rng, remove from HPO, change to accept folds like genetic

            selector = HarrisHawkParallel(objective_function_zoo, **selector_args, minimize=False)
            selected_feature_names = set(selector.fit(estimator, precomputed["X_train_i"], precomputed["y_train_i"], precomputed["X_test_i"], precomputed["y_test_i"], verbose=False))
            intersection = list(set(selected_feature_names) & set(X_train.columns.tolist()))
            return X_train[intersection], X_test[intersection], None
        case "genetic":
            selector = GeneticParallel(objective_function_zoo, **selector_args, parallelism=parallelism, minimize=False)
            fold_vars = extract_fold_list(precomputed)
            selected_feature_names = selector.fit_cv(estimator, precomputed["X_train"], precomputed["y_train"], fold_vars, verbose=verbose)
            return X_train[selected_feature_names], X_test[selected_feature_names], None
        case "HFMOEA":
            if "topk" in selector_args.keys():
                selector_args["topk"] = min(max_features, selector_args["topk"])  # limit to max feature count after preprocessing
            fold_vars = extract_fold_list(precomputed)
            feature_mask = reduceFeaturesMaxAcc(precomputed["X_train"], precomputed["y_train"], fold_vars, estimator, **selector_args, n_jobs=parallelism, is_classification=isClassification, sol=precomputed["sol"].get().result(), dask_parallel=dask_parallel, verbose=verbose)
            return  X_train[ feature_mask], X_test[feature_mask], None
        case "embedded-tree":
            forest = RandomForestClassifier(n_estimators=selector_args["e_n_estimators"], max_depth=selector_args["e_max_depth"], n_jobs=parallelism, random_state=42) if isClassification else RandomForestRegressor(n_estimators=selector_args["e_n_estimators"], max_depth=selector_args["e_max_depth"], n_jobs=parallelism, random_state=42)
            forest.fit(X_train, y_train)
            selector_args["e_max_features"] = min(max_features, selector_args["e_max_features"])  # limit to max feature count after preprocessing
            model = SelectFromModel(forest, max_features=selector_args["e_max_features"], prefit=True)
            model.set_output(transform="pandas")
            return model.transform(X_train), model.transform(X_test), None
        case "SVD-entropy":
            boolean_mask = precomputed["mask"].get().result().to_list()
            return X_train.loc[:, boolean_mask], X_test.loc[:, boolean_mask], None
        case "NDFS":
            np_view = X_test.to_numpy()
            W = ndfs(np_view, n_clusters=selector_args["n_clusters"], alpha=selector_args["alpha"], beta=selector_args["beta"])
            ranking = feature_ranking(W)
            selector_args["n_features_to_select"] = min(max_features, selector_args["n_features_to_select"])  # limit to max feature count after preprocessing
            sliced = ranking[:selector_args["n_features_to_select"]]
            return X_train.iloc[:, sliced], X_test.iloc[:, sliced], None
        case "optuna-combined":
            retained_features = set(X_train.columns)
            selected_feature_names_list = [ retained_features & set(v) for k, v in group_dict.items() if selector_args[k]]
            selected_feature_names = list(itertools.chain.from_iterable(selected_feature_names_list))
            return X_train[selected_feature_names], X_test[selected_feature_names], None
        case "cost-cfs":
            cost_fn = _make_cost_fn(X_train, X_test, group_dict, feature_group_times)
            selector = CostBasedCFS(cost_penalty=selector_args.get("cost_penalty", 0.1))
            selector.fit(X_train, y_train, cost_fn=cost_fn)
            return selector.transform(X_train), selector.transform(X_test), None
        case "cost-gb":
            cost_fn = _make_cost_fn(X_train, X_test, group_dict, feature_group_times)
            n_features = X_train.shape[1]
            max_cost = cost_fn(list(range(n_features))) if cost_fn is not None else float(n_features)
            budget = selector_args.get("gb_budget_frac", 1.0) * max_cost
            selector = CostConstrainedGBSelector(
                is_classification=isClassification,
                n_jobs=parallelism,
                max_iter=10,
                delta=None,
                budget=budget,
                gb_params={
                    "n_estimators":  selector_args.get("gb_n_estimators", 100),
                    "max_depth":     selector_args.get("gb_max_depth", 6),
                    "learning_rate": selector_args.get("gb_learning_rate", 0.1),
                },
            )
            selector.fit(X_train, y_train, cost_fn=cost_fn)
            return selector.transform(X_train), selector.transform(X_test), None
        case "mopso":
            cost_fn = _make_cost_fn(X_train, X_test, group_dict, feature_group_times)
            selector = MOPSOFeatureSelector(
                n_particles=selector_args.get("n_particles", 30),
                n_iterations=selector_args.get("n_iterations", 100),
                selection_strategy=selector_args.get("selection_strategy", "knee"),
                is_classification=isClassification,
                n_jobs=parallelism,
            )
            selector.fit(X_train, y_train, cost_fn=cost_fn)
            return selector.transform(X_train), selector.transform(X_test), selector
        case _:
            raise ValueError("Invalid Feature Subset")


def get_selection_HPO_space(features : str, trial : Trial, isClassification : bool, group_dict : dict[str, list[str]], no_features: int) -> dict[str, Any]:
    max_features = math.ceil(no_features / 3.0)
    min_features = min(5, max_features)
    match features:
        case "all" | "SATzilla" | "SATfeatPy" | "FMBA" | "FM_Chara" | "prefilter":
            return {}
        case "kbest-mutalinfo":
            return {
                "k" : trial.suggest_int("k", min_features, max_features),
                "n_neighbors" : trial.suggest_int("n_neighbors", 2, 10)
            }
        case "multisurf":
            return {
                "n_features_to_select" : trial.suggest_int("n_features_to_select", min_features, max_features)
            }
        case "mRMR":
            return {
                "K" : trial.suggest_int("K", min_features, max_features),
                "relevance" : trial.suggest_categorical("relevance", ["f", "rf"]),
                "denominator" : trial.suggest_categorical("denominator", ["max", "mean"]),
            }
        case "RFE":
            return {
                "step" : trial.suggest_float("step", 0.01, 1),
                "n_features_to_select" : trial.suggest_int("n_features_to_select", min_features, max_features)
            }
        case "genetic":
            return {
                "selective_pressure" : trial.suggest_int("selective_pressure", 1, 5),
                "elitism" : trial.suggest_int("elitism", 1, 5),
                "mutation_rate" : trial.suggest_float("mutation_rate", 0.01, 0.3),
                "population_size" : trial.suggest_int("population_size", 10, 30, 2), # only allow even values
                "n_iteration" : 30 #todo n_iterations?
            }
        case "harris-hawks":
            return {
                "population_size" : trial.suggest_int("population_size", 10, 20, 2), # only allow even values
                "beta" : trial.suggest_float("beta", 0.001, 1.999),
                "n_iteration": 20  # todo n_iterations?
            }
        case "HFMOEA":
            return {
                "topk": trial.suggest_int("topk", min_features - 1, max_features - 1),
                "pop_size" : trial.suggest_int("pop_size", 30, 120),
                "max_gen" : 100,
                "mutation_probability" : trial.suggest_float("mutation_probability", 0.01, 0.3)
            }
        case "embedded-tree":
            return {
                "e_n_estimators": trial.suggest_int("e_n_estimators", 10, 1000),
                "e_max_depth": trial.suggest_categorical("e_max_depth", [10, 50, 100, 500]),
                "e_max_features" : trial.suggest_int("e_max_features", min_features, max_features),
            }
        case "SVD-entropy":
            return {}
        case "NDFS":
            return {
                "alpha" : trial.suggest_float("alpha", 1e-2, 10, log=True),
                "beta" : trial.suggest_float("beta", 1e-2, 10, log=True),
                "n_clusters" : trial.suggest_int("n_clusters", 4, 100),
                "n_features_to_select": trial.suggest_int("n_features_to_select", min_features, max_features),
            }
        case "optuna-combined":
            return {
                group_name : trial.suggest_categorical(group_name, [True, False]) for group_name in group_dict.keys()
            }
        case "cost-cfs":
            return {
                "cost_penalty": trial.suggest_float("cost_penalty", 0.0, 2.0),
            }
        case "cost-gb":
            return {
                "gb_n_estimators":  trial.suggest_int("gb_n_estimators", 50, 200),
                "gb_max_depth":     trial.suggest_int("gb_max_depth", 3, 10),
                "gb_learning_rate": trial.suggest_float("gb_learning_rate", 0.01, 0.3, log=True),
                "gb_budget_frac":   trial.suggest_float("gb_budget_frac", 0.05, 1.0),
            }
        case "mopso":
            return {
                "n_particles": trial.suggest_int("n_particles", 5, 20),
                "n_iterations": trial.suggest_int("n_iterations", 10, 40),
                "selection_strategy": trial.suggest_categorical("selection_strategy", ["min_cost", "max_perf", "knee"]),
            }
        case _:
            raise ValueError("Invalid Feature Subset")
