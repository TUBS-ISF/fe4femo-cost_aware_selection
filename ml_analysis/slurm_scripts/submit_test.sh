#!/bin/bash

set -euo pipefail

CONFIG="${1:?first argument must be path to config.txt}"
PARTITION="${2:?second argument must be SLURM partition name}"
MAX_SUBMIT="${3:-55}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run.sh"
OUTPUT_PATH="$HOME/GITZ-home/fe4femo/ml_analysis/out/main"
CONTAINER_OUTPUT_PATH="fe4femo/ml_analysis/out/main"
DATA_PATH="GITZ-home/fe4femo/data"

mkdir -p "${OUTPUT_PATH}"

total=$(awk 'NR>1 && $1 ~ /^[0-9]+$/ {c++} END {print c+0}' "${CONFIG}")
to_submit=$(( total < MAX_SUBMIT ? total : MAX_SUBMIT ))

if [[ "${to_submit}" -le 0 ]]; then
    echo "No experiments remaining in ${CONFIG}."
    exit 0
fi

echo "Submitting ${to_submit} of ${total} remaining experiments."

submitted_nos=()
count=0

while IFS= read -r line && [[ "${count}" -lt "${to_submit}" ]]; do
    [[ -z "${line}" || ! "${line}" =~ ^[0-9] ]] && continue

    read -r exp_no name task_count runtime fold_no feature task model hpo_its \
             bool_model_hpo bool_selector_hpo bool_multi_objective <<< "${line}"

    [[ "${bool_model_hpo}"       == "True" ]] && model_hpo="--modelHPO"       || model_hpo="--no-modelHPO"
    [[ "${bool_selector_hpo}"    == "True" ]] && selector_hpo="--selectorHPO"  || selector_hpo="--no-selectorHPO"
    [[ "${bool_multi_objective}" == "True" ]] && multi_obj="--multiObjective"  || multi_obj="--no-multiObjective"

    echo "------------------------------------------------------------"
    echo "Config id : ${exp_no}  (${name})"
    echo "  feature=${feature}  task=${task}  model=${model}"
    echo "  fold=${fold_no}  hpo_its=${hpo_its}  runtime=${runtime} min"
    echo "  tasks=${task_count}  model_hpo=${bool_model_hpo}  selector_hpo=${bool_selector_hpo}"

    time_limit=$((runtime + 10))

    sbatch_cmd=(
        sbatch
        --job-name="${name}"
        --partition="${PARTITION}"
        --ntasks="${task_count}"
        --time="${time_limit}"
        --cpus-per-task=32
        --mem-per-cpu=1950
        --nodes=1
        --output="${OUTPUT_PATH}/${name//#/_}_%j.out"
        --export=ALL,ML_FOLD="${fold_no}"
        "${RUN_SCRIPT}"
        --feature "${feature}"
        --task "${task}"
        --model "${model}"
        --HPOits "${hpo_its}"
        "${model_hpo}" "${selector_hpo}" "${multi_obj}"
        "${DATA_PATH}" "${CONTAINER_OUTPUT_PATH}"
    )

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "  DRY_RUN — would run: ${sbatch_cmd[*]}"
    else
        output="$("${sbatch_cmd[@]}")"
        job_id="$(echo "${output}" | awk '{print $4}')"
        echo "  Submitted → job ${job_id}"
        echo "  Log: ${OUTPUT_PATH}/${name//#/_}_${job_id}.out"
    fi

    submitted_nos+=("${exp_no}")
    count=$((count + 1))
done < "${CONFIG}"

if [[ "${#submitted_nos[@]}" -gt 0 && "${DRY_RUN:-0}" != "1" ]]; then
    ids="${submitted_nos[*]}"
    awk -v ids="${ids}" \
        'BEGIN{split(ids,a); for(i in a) del[a[i]]=1} NR==1 || !($1 in del)' \
        "${CONFIG}" > "${CONFIG}.tmp" && mv "${CONFIG}.tmp" "${CONFIG}"
    remaining=$(awk 'NR>1 && $1 ~ /^[0-9]+$/ {c++} END {print c+0}' "${CONFIG}")
    echo "============================================================"
    echo "Submitted ${#submitted_nos[@]} experiments. ${remaining} remaining in config."
fi

echo "Monitor:  squeue -u \$USER"
echo "Logs:     ls ${OUTPUT_PATH}/*.out"
