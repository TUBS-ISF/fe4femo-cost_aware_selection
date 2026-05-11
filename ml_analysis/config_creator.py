import itertools
import math
from datetime import timedelta

from generate_fold_model import create_run_name
from helper.input_parser import get_feature_list, get_model_list, get_task_list

def is_modelHPO(feature: str) -> bool:
    return is_selectorHPO(feature)

def is_selectorHPO(feature: str) -> bool:
    return feature not in ["genetic", "HFMOEA"]

def get_task_count(feature: str) -> int:
    if feature in ["genetic", "HFMOEA"]:
        return 4
    else:
        return 3

def get_runtime(hpoIts: int, feature: str, individual_folds : bool, multi_objective: bool) -> str:
    selector_modifier_m = {
        "all" : 160,
        "prefilter" : 100,
        "SATzilla" : 32,
        "SATfeatPy" : 40,
        "FMBA" : 18,
        "FM_Chara" : 18,
        "kbest-mutalinfo" : 45,
        "multisurf" : 60,
        "mRMR" : 70,
        "RFE" : 6*60,
        "genetic" : 3*60 + 30,
        "HFMOEA" : 50*60 + 30,
        "embedded-tree" : 175,
        "SVD-entropy" : 30,
        "NDFS" : 36,
        "optuna-combined" : 70,
        "cost-cfs" : 60,
        "cost-gb" : 90,
        "mopso" : 210,
    }

    final_modifier = 1.8
    value = selector_modifier_m[feature]
    if feature != "genetic":
        value *= hpoIts / 150
    if individual_folds:
        value /= 8 #less since additional startup for container
    if multi_objective:
        value *= 2
    value *= final_modifier
    value += 20

    value = math.ceil(value)
    value = min(value, 72*60)
    return str(value)

def get_HPO_its(multi_objective: bool) -> int:
    if multi_objective:
        return 300
    else:
        return 150


def check_valid(feature: str, model: str, task: str, multi_objective: bool) -> bool:
    if feature == "harris-hawks":
        return False
    if feature == "RFE" and not (model == "gradboostForest" or model == "randomForest" or model == "adaboost"):
        return False
    if multi_objective and feature not in ["optuna-combined"]:
        return False
    return True

#todo configure
def check_desired(feature: str, model: str, task: str, multi_objective: bool) -> bool:
    return True


if __name__ == '__main__':
    #todo configure
    individual_folds = False



    folds = range(10) if individual_folds else [-1]
    combinations = [
        (fold, feature, model, task, multi_objective) for fold, feature, model, task, multi_objective in itertools.product(folds, get_feature_list(), get_model_list(), get_task_list(), [True, False])
        if check_valid(feature, model, task, multi_objective) and check_desired(feature, model, task, multi_objective)
    ]

    combinations.sort(key=lambda c: int(get_runtime(get_HPO_its(c[4]), c[1], individual_folds, c[4])))

    experiments = [
        f"{i} {create_run_name(feature, task, model, is_modelHPO(feature), is_selectorHPO(feature), get_HPO_its(multi_objective), multi_objective, fold)} {get_task_count(feature)} {get_runtime(get_HPO_its(multi_objective), feature, individual_folds, multi_objective)} {fold} {feature} {task} {model} {get_HPO_its(multi_objective)} {is_modelHPO(feature)} {is_selectorHPO(feature)} {multi_objective}"
        for i, (fold, feature, model, task, multi_objective) in enumerate(combinations)
    ]

    runtime_estimation = 0
    for experiment in experiments:
        runtime_estimation += int(experiment.split(" ")[3])
    print(f"Total estimated sequential runtime: {timedelta(minutes=runtime_estimation)}")

    file_count = 0
    for experiment in experiments:
        fold = int(experiment.split(" ")[4])
        uses_hpo = experiment.split(" ")[9] == "True" or experiment.split(" ")[10]  == "True"
        file_count += 1 # for .out file
        tmp_count = 2
        if uses_hpo:
            tmp_count += 1
        if fold == -1:
            tmp_count *= 10
        file_count += tmp_count
    print(f"Total number of generated files: {file_count}")


    with open("config.txt", "w") as f:
        f.write("NO NAME TASK_COUNT RUNTIME FOLD_NO FEATURE TASK MODEL HPO_ITS MODEL_HPO SELECTOR_HPO MULTI_OBJECTIVE\n")
        f.write("\n".join(experiments))

