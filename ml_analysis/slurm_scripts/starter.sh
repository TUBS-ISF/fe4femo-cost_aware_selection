#!/bin/bash

config="${3:-"$HOME/GITZ-home/fe4femo/ml_analysis/config.txt"}"
output_path="${2:-"GITZ-home/fe4femo/ml_analysis/out/main"}"
script_path="$HOME/GITZ-home/fe4femo/ml_analysis/slurm_scripts/run.sh"
data_path="GITZ-home/fe4femo/data"

mkdir -p $HOME/$output_path

partition="${1:-multiple_il}"
maxIt=$(wc -l < "$config")
((maxIt=maxIt-1))
maxConcurrent=50
maxSubmit="${4:-60}"

submitted=0
submitted_ids=()

while [[ maxIt -ge 0 && submitted -lt maxSubmit ]] ; do
    ((submitted=submitted+1))
    echo "model $maxIt submitted"
    EXPERIMENT_NO=$maxIt

    # parameters

    name=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $2}' $config)
    task_count=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $3}' $config)
    runtime=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $4}' $config)
    out_path=$HOME/${output_path}/${name//#/_}_%j.out
    fold_no=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $5}' $config)
    feature=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $6}' $config)
    task=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $7}' $config)
    model=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $8}' $config)
    HPOits=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $9}' $config)
    bool_modelHPO=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $10}' $config)
    [[ "$bool_modelHPO" == "True" ]] && modelHPO="--modelHPO" || modelHPO="--no-modelHPO"
    bool_selectorHPO=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $11}' $config)
    [[ "$bool_selectorHPO" == "True" ]] && selectorHPO="--selectorHPO" || selectorHPO="--no-selectorHPO"
    bool_multiObjective=$(awk -v ID=$EXPERIMENT_NO '$1==ID {print $12}' $config)
    [[ "$bool_multiObjective" == "True" ]] && multiObjective="--multiObjective" || multiObjective="--no-multiObjective"

    ############

    output=$(sbatch -J "$name" -n "$task_count" --time="$runtime" --output=$out_path  --partition="${partition}" --export=ALL,ML_FOLD="$fold_no" "$script_path" --feature "$feature" --task "$task" --model "$model" --HPOits "$HPOits" "$modelHPO" "$selectorHPO" "$multiObjective" "$data_path" "$output_path" )
    if [[ $output =~ "error" ]]; then
       echo "Error in Submission, trying again!"
       ((submitted=submitted-1))
    else
       submitted_ids+=($EXPERIMENT_NO)
       ((maxIt=maxIt-1))
    fi
done

# Remove submitted experiments from config
if [[ ${#submitted_ids[@]} -gt 0 ]]; then
    awk -v ids="${submitted_ids[*]}" \
        'BEGIN{split(ids,a); for(i in a) del[a[i]]=1} NR==1 || !($1 in del)' \
        "$config" > "${config}.tmp" && mv "${config}.tmp" "$config"
    echo "Removed ${#submitted_ids[@]} experiments from config. $(( $(wc -l < "$config") - 1 )) remaining."
fi

mail -s "Finished Job Submission" "l.gsuck@tu-braunschweig.de" <<< "Submission script finished, but jobs could still be running. Check with command 'squeue' to list all currently running jobs!"
