#!/bin/bash
# Submit a selection of experiments from config.txt for testing.
#
# Usage:
#   bash submit_test.sh <config> <partition> <id> [id2] [id3] ...
#
# Example — run experiments 0, 5, and 12:
#   bash submit_test.sh slurm_scripts/config.txt multiple_il 0 5 12
#
# Dry-run (print sbatch calls without submitting):
#   DRY_RUN=1 bash submit_test.sh slurm_scripts/config.txt multiple_il 0 5

set -euo pipefail

CONFIG="${1:?first argument must be path to config.txt}"
PARTITION="${2:?second argument must be SLURM partition name}"
shift 2
IDS=("$@")

if [[ "${#IDS[@]}" -eq 0 ]]; then
    echo "Usage: $0 <config> <partition> <id1> [id2] ..." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run.sh"
OUTPUT_PATH="$HOME/fe4femo/ml_analysis/out/main"
DATA_PATH="fe4femo/data"

mkdir -p "${OUTPUT_PATH}"

for id in "${IDS[@]}"; do
    # config.txt: row 1 is the header, data rows start at line 2 = id 0
    line_no=$((id + 2))
    line="$(sed -n "${line_no}p" "${CONFIG}")"

    if [[ -z "${line}" ]]; then
        echo "WARNING: no config row for id ${id}" >&2
        continue
    fi

    read -r _no name task_count runtime fold_no feature task model hpo_its \
             bool_model_hpo bool_selector_hpo bool_multi_objective <<< "${line}"

    [[ "${bool_model_hpo}"       == "True" ]] && model_hpo="--modelHPO"       || model_hpo="--no-modelHPO"
    [[ "${bool_selector_hpo}"    == "True" ]] && selector_hpo="--selectorHPO"  || selector_hpo="--no-selectorHPO"
    [[ "${bool_multi_objective}" == "True" ]] && multi_obj="--multiObjective"  || multi_obj="--no-multiObjective"

    echo "------------------------------------------------------------"
    echo "Config id : ${id}  (${name})"
    echo "  feature=${feature}  task=${task}  model=${model}"
    echo "  fold=${fold_no}  hpo_its=${hpo_its}  runtime=${runtime} min"
    echo "  tasks=${task_count}  model_hpo=${bool_model_hpo}  selector_hpo=${bool_selector_hpo}"

    # Add 10-minute buffer so minor startup variance does not cause timeouts
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
        "${DATA_PATH}" "fe4femo/ml_analysis/out/main"
    )

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "  DRY_RUN — would run: ${sbatch_cmd[*]}"
    else
        output="$("${sbatch_cmd[@]}")"
        job_id="$(echo "${output}" | awk '{print $4}')"
        echo "  Submitted → job ${job_id}"
        echo "  Log: ${OUTPUT_PATH}/${name}_${job_id}.out"
    fi
done

echo "============================================================"
echo "Monitor:  squeue -u \$USER"
echo "Logs:     ls ${OUTPUT_PATH}/*.out"
