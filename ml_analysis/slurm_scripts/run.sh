#!/bin/bash
#SBATCH --time=1:0:0
#SBATCH --job-name=eval_model
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=1950
#SBATCH --nodes=2-70
#SBATCH --use-min-nodes

if [[ -z "$ML_FOLD" ]]; then
  echo "Must provide ML_FOLD in environment" 1>&2
  exit 1
fi

export DASK_LOGGING__DISTRIBUTED=WARN

export OMP_NUM_THREADS=2
echo -e "JOB_ID=${SLURM_JOB_ID}"
echo -e "OMP_THREADS=${OMP_NUM_THREADS}"

echo -e "########\nCONTAINER START"

SIF="$HOME/fe4femo/ml_analysis/ml_analysis.sif"

if [ "$ML_FOLD" = "-1" ]; then
  RUN_COMMAND=("/app/slurm_scripts/fold_connector.sh")
else
  RUN_COMMAND=("/app/slurm_scripts/slurm_fork_tracker.sh")
fi

srun apptainer exec \
  --bind /etc/slurm/task_prolog:/etc/slurm/task_prolog \
  --bind /scratch:/scratch \
  --bind "$HOME:$HOME" \
  --bind "$HOME/fe4femo/ml_analysis:/app" \
  --pwd /app \
  "$SIF" "${RUN_COMMAND[@]}" "$@"
