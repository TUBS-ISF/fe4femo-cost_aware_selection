import warnings
from datetime import datetime
import os
import tempfile
import time
import multiprocessing as mp
from multiprocessing import freeze_support

import cloudpickle
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.preprocessing import LabelEncoder

from helper.data_classes import FoldResult, TrialContainer

os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
from pathlib import Path
from statistics import mean

import optuna
from dask.distributed import Client
import dask
from distributed import worker_client, performance_report, get_task_stream
from sklearn.metrics import matthews_corrcoef, r2_score, d2_absolute_error_score
from sklearn.model_selection import StratifiedKFold

from helper.SlurmMemRunner import SLURMMemRunner
from helper.feature_selection import get_selection_HPO_space, get_feature_selection, precompute_feature_selection, \
     transform_dict_to_var_dict
from helper.input_parser import parse_input
from helper.load_dataset import generate_xy_split, get_dataset, get_flat_models, is_task_classification, \
    load_feature_groups, load_feature_group_times
from helper.model_training import get_model_HPO_space, get_model
from helper.optuna_helper import categorical_distance_function

def eval_model_performance(model_instance, X_train_test, precomputed, is_classification):
    y_test = precomputed["y_test"].get().result()
    X_train, X_test = X_train_test
    y_pred = model_instance.predict(X_test)
    if is_classification:
        return matthews_corrcoef(y_test, y_pred)
    else:
        return d2_absolute_error_score(y_test, y_pred)

def train_model(model, is_classification, cores, model_config, X_train_test, precomputed):
    y_train = precomputed["y_train"].get().result()
    X_train, X_test = X_train_test
    model_instance = get_model(model, is_classification, cores, model_config )
    model_instance.fit(X_train, y_train)
    return model_instance

def do_feature_selection(model, is_classification, model_config, precomputed, features, selector_config, feature_groups, cores, verbose = False, feature_group_times = None):
    model_instance_selector = get_model(model, is_classification, 1, model_config )
    return get_feature_selection(precomputed, features, is_classification, selector_config, model_instance_selector, feature_groups, parallelism=cores, verbose=verbose, dask_parallel=False, feature_group_times=feature_group_times)

def eval_feature_group_runtime(X_train_test, feature_groups: dict[str, list[str]], feature_group_times: pd.DataFrame) -> float:
    X_train, X_test = X_train_test
    instances = list(set(X_train.index.values).union(set(X_test.index.values)))
    active_features = set(X_train.columns.values)
    active_groups = [ group for group, feature_list in feature_groups.items() if any( feature in active_features for feature in feature_list ) ]
    return feature_group_times.loc[instances, active_groups].sum(axis=1).mean()

#add further objectives here
def build_fold_result(model_performance: float, feature_computation_time: float) -> FoldResult:
    return FoldResult(
        model_quality=model_performance,
        feature_computation_time=feature_computation_time,
    )

def compute_fold(client, model, features, is_classification, model_config, selector_config, feature_groups, feature_group_times, precomputed, cores : int, verbose = False)  -> float:
    # feature selection + model training
    X_train_test = client.submit(do_feature_selection, model, is_classification, model_config, precomputed, features, selector_config, feature_groups, cores, verbose, feature_group_times, pure=False)
    model_instance = client.submit(train_model, model, is_classification, cores, model_config, X_train_test, precomputed, pure=False)
    model_performance = client.submit(eval_model_performance, model_instance, X_train_test, precomputed, is_classification, pure=False)
    feature_computation_time = client.submit(eval_feature_group_runtime, X_train_test, feature_groups, feature_group_times, pure=False)
    #add further objectives here
    return client.submit(build_fold_result, model_performance, feature_computation_time, pure=False)

def objective(trial: optuna.Trial, folds, features, model, should_modelHPO, is_classification, feature_groups, feature_group_times, feature_count, cores : int, multi_objective : bool) -> float | tuple[float, float]:
    try:
        model_config = get_model_HPO_space(model, trial, is_classification) if should_modelHPO else None
        selector_config = get_selection_HPO_space(features, trial, is_classification, feature_groups, feature_count)
        with worker_client() as client:
            futures = [
                compute_fold(client, model, features, is_classification, model_config, selector_config, feature_groups, feature_group_times, future_precompute, cores)
                for i, future_precompute in folds.items()]
            results = client.gather(futures, direct=True)
        if multi_objective:
            return mean(x.model_quality for x in results), mean(x.feature_computation_time for x in results)
        else:
            return mean(x.model_quality for x in results)
    except Exception as e:
        # fix if further objectives added
        if multi_objective:
            return -100.4242, 42*10e9
        else:
            return -100.4242

def create_run_name(features: str, task: str, model: str, modelHPO: bool, selectorHPO: bool, hpo_its: int, multi_objective : bool, foldNo : int) -> str:
    ret_value = f"{task}#{features}#{model}#{modelHPO}#{selectorHPO}"
    if modelHPO or selectorHPO:
        ret_value += f"#{hpo_its}"
    ret_value += f"#{multi_objective}"
    if foldNo >= 0:
        ret_value += f"#{foldNo}"
    return ret_value

def optimize_optuna(study: optuna.study.Study, objective_function, lock : dask.distributed.Lock, counter : dask.distributed.Variable):
    while True:
        with lock:
            counter_value = counter.get()
            if counter_value <= 0:
                break
            counter.set(counter_value - 1)
        study.optimize(objective_function, n_trials=1)


def compute_final_model(client, model, features, X_train, X_test, y_train, y_test, is_classification, model_config, selector_config, model_flatness, feature_groups, easy_model, cores, verbose, feature_group_times=None):
    model_instance_selector = get_model(model, is_classification, 1, model_config, easy_model=easy_model)
    start_FS = time.time()
    precomputed = client.submit(precompute_feature_selection, features, is_classification, X_train, X_test, y_train,
                                y_test, model_flatness, parallelism=cores, pure=False)
    precomputed = client.submit(transform_dict_to_var_dict, precomputed, pure=False)
    fs_future = client.submit(get_feature_selection, precomputed, features, is_classification, selector_config,
                              model_instance_selector, feature_groups, parallelism=cores, verbose=verbose,
                              dask_parallel=True, feature_group_times=feature_group_times, pure=False)
    X_train, X_test = fs_future.result()
    end_FS = time.time()
    model_instance = get_model(model, is_classification, cores, model_config)
    start_Model = time.time()
    model_instance.fit(X_train, y_train)
    end_Model = time.time()
    model_complete = model_instance
    return model_complete, X_test, end_FS - start_FS, end_Model - start_Model


def main(in_proc_id: int, worker_count : int, pathData: str, pathOutput: str, features: str, task: str, model: str, modelHPO: bool, selectorHPO: bool, hpo_its: int, multi_objective: bool, foldNo : int):
    warnings.simplefilter("ignore", UserWarning)
    cores = int(os.getenv("OMP_NUM_THREADS", "1"))
    scheduler_options = {
        "interface": "ib0",
    }
    worker_options = {
        "local_directory": "$TMPDIR/",
        "nthreads": 1,
        "interface": "ib0",
        "memory_limit": f"{cores * int(os.getenv("SLURM_MEM_PER_CPU", 2000))}MB",
    }
    scheduler_path = Path(os.path.expandvars("$HOME") + "/tmp/scheduler_files")
    scheduler_path.mkdir(parents=True, exist_ok=True)

    with (SLURMMemRunner(scheduler_file=str(scheduler_path)+f"/scheduler-{{job_id}}_{foldNo}.json", in_proc_id=in_proc_id, fold_no=foldNo,
                      worker_options=worker_options, scheduler_options=scheduler_options) as runner):
        scheduler_file = Path(runner.scheduler_file)
        with Client(runner, direct_to_workers=True) as client:
            Path(pathOutput).mkdir(parents=True, exist_ok=True)
            run_config = {
                "name": create_run_name(features, task, model, modelHPO, selectorHPO, hpo_its, multi_objective, foldNo),
                "path_data": pathData,
                "path_output": pathOutput,
                "features": features,
                "task": task,
                "model": model,
                "modelHPO": modelHPO,
                "selectorHPO" : selectorHPO,
                "hpo_its": hpo_its,
                "multiObjective": multi_objective,
                "foldNo": foldNo,
            }

            with (
                performance_report(filename=run_config["path_output"] + "/" + run_config["name"] + ".html"),
                get_task_stream() as task_stream
            ):
                dask.distributed.print(f"{datetime.now()}   Dask dashboard is available at {client.dashboard_link}")
                client.wait_for_workers(worker_count)
                dask.distributed.print(str(datetime.now()) + "  Initialized all workers")

                X, y = get_dataset(pathData, task)

                #remove #SAT features for #SAT prediction
                if task == "value_ssat":
                    X = X.drop(['DyMMer_9/DyMMer/Number_of_valid_configurations_(NVC)','FMBA_18/FMBA/NumberOfValidConfigurationsLog', 'FM_Characterization_1/FM_Characterization/ANALYSIS/Configurations/value', 'FM_Characterization_1/FM_Characterization/ANALYSIS/Partial_variability/value', 'FM_Characterization_1/FM_Characterization/ANALYSIS/Total_variability/value'], axis=1)

                is_classification = is_task_classification(task)
                label_encoder = None
                if is_classification:
                    y_index = y.index
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y)
                    y = pd.Series(y, index=y_index)
                y = pd.to_numeric(y, downcast='float')
                X_train, X_test, y_train, y_test = generate_xy_split(X, y, pathData+"/folds.txt", foldNo)

                feature_count = X_train.shape[1]

                kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                model_flatness = get_flat_models(X_train)
                flatness_future = client.scatter(model_flatness, direct=True)

                feature_groups = load_feature_groups(pathData)
                feature_group_times = load_feature_group_times(pathData)
                feature_group_times_future = client.scatter(feature_group_times, direct=True)


                best_params = {}
                journal_path = "no journal"
                verbose = False
                easy_model = False

                dask.distributed.print(str(datetime.now()) + "  Loaded all Data")
                if selectorHPO:
                    splits = kf.split(X_train, model_flatness)

                    # dask export for better cluster behaviour

                    folds = {}
                    for i, (train_index, test_index) in enumerate(splits):
                        X_train_inner = X_train.iloc[train_index]
                        X_test_inner = X_train.iloc[test_index]
                        y_train_inner = y_train.iloc[train_index]
                        y_test_inner = y_train.iloc[test_index]
                        # feature preprocessing
                        future_pre = client.submit(precompute_feature_selection, features, is_classification,
                                                   X_train_inner, X_test_inner, y_train_inner, y_test_inner,
                                                   flatness_future, 0.9, cores, pure=True)
                        future_pre = client.submit(transform_dict_to_var_dict, future_pre)
                        folds[i] = future_pre
                    dask.distributed.print(str(datetime.now()) + "  Initialized Folds")
                    objective_function = lambda trial: objective(trial, folds, features, model, modelHPO, is_classification, feature_groups, feature_group_times_future, feature_count, cores, multi_objective)

                    journal_path = run_config["path_output"] + "/" + run_config["name"] + ".journal"
                    journal = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(journal_path))
                    storage = optuna.integration.dask.DaskStorage(journal)
                    sampler = TPESampler(seed=None, multivariate=True, group=True, constant_liar=True, categorical_distance_func=categorical_distance_function())
                    study = optuna.create_study(study_name=run_config["name"], storage=storage, directions=["maximize", "minimize"], sampler=sampler) if multi_objective else optuna.create_study(study_name=run_config["name"], storage=storage, direction="maximize", sampler=sampler)

                    n_jobs = 25 #2 less than tasks for scheduler and main-node

                    lock = dask.distributed.Lock("LOCK_COUNTER_VAR")
                    counter = dask.distributed.Variable()
                    counter.set(hpo_its)
                    dask.distributed.print(str(datetime.now()) + "  Initialized optuna")
                    futures = [
                        client.submit(optimize_optuna, study, objective_function, lock, counter, pure=False) for _ in range(n_jobs)
                    ]
                    dask.distributed.print(str(datetime.now()) + "  Started optuna worker")
                    dask.distributed.wait(futures)
                    dask.distributed.print(str(datetime.now()) + "  Optuna optimization completed")

                dask.distributed.print(f"{datetime.now()}  Start training final models")
                if selectorHPO:
                    trial_container = []
                    frozen_trials = study.best_trials if multi_objective else [study.best_trial]
                    for j, frozen_best_trial in enumerate(frozen_trials):
                        dask.distributed.print(f"{datetime.now()}  Start training final model {j}")
                        # feature selection + model training
                        best_params = frozen_best_trial.params
                        model_config = get_model_HPO_space(model, frozen_best_trial, is_classification) if modelHPO else None
                        selector_config = get_selection_HPO_space(features, frozen_best_trial, is_classification, feature_groups, X_train.shape[1])
                        model_complete, X_test_mod, time_feature, time_model = compute_final_model(client, model, features, X_train,
                                                                                       X_test, y_train, y_test,
                                                                                       is_classification, model_config,
                                                                                       selector_config, model_flatness,
                                                                                       feature_groups, easy_model, cores,
                                                                                       verbose, feature_group_times)
                        trial_container.append(TrialContainer(model=model_complete, best_params=best_params, time_Feature=time_feature, time_Model=time_model, x_test=X_test_mod))

                else:
                    model_config = {}
                    selector_config = {}
                    verbose=True
                    easy_model = True

                    model_complete, X_test, time_feature, time_model = compute_final_model(client, model, features, X_train,
                                                                                   X_test, y_train, y_test,
                                                                                   is_classification, model_config,
                                                                                   selector_config, model_flatness,
                                                                                   feature_groups, easy_model, cores,
                                                                                   verbose, feature_group_times)
                    trial_container = [TrialContainer(model=model_complete, best_params=best_params, time_Feature=time_feature, time_Model=time_model, x_test=X_test)]

                # export for later use
                output = {
                    "y_test": y_test,
                    "label_encoder" : label_encoder,
                    "run_config": run_config,
                    "journal_path": journal_path,
                    "trial_container" : trial_container
                }
            output["task_stream"] = task_stream.data
            path = run_config["path_output"] + "/" + run_config["name"] + ".pkl"
            with open(path, "wb") as f:
                cloudpickle.dump(output, f)
            print(f"{datetime.now()}   Exported model at {path}")

    print(f"{datetime.now()}  Shutdown main-client completed")
    scheduler_file.unlink(missing_ok=True)


if __name__ == '__main__':
    freeze_support()
    args = parse_input()
    cpus_per_node = int(os.getenv("SLURM_CPUS_ON_NODE", 128)) // int(os.getenv("OMP_NUM_THREADS", 2))
    no_nodes = int(os.getenv("SLURM_JOB_NUM_NODES", 1))
    worker_count = cpus_per_node*no_nodes - 2
    if worker_count < 25:
        raise ValueError("Not enough worker, needs at least 25")
    print(f"Starting {worker_count} workers with {int(os.getenv("OMP_NUM_THREADS", 2))} cores per worker")

    function_args = (worker_count, os.environ.get("HOME")+"/"+os.path.expandvars(args.pathData), os.environ.get("HOME")+"/"+os.path.expandvars(args.pathOutput), args.features, args.task, args.model, args.modelHPO, args.selectorHPO, args.HPOits, args.multiObjective, args.foldNo)

    ctx = mp.get_context("spawn")
    processes = [ctx.Process(target=main, args=((i,)+function_args), daemon=False) for i in range(cpus_per_node)]
    for p in processes:
        p.start()
    for p in processes:
        exit_value = p.join()
