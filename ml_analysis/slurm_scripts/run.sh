#!/bin/bash
#SBATCH --time=1:0:0
#SBATCH --job-name=eval_model
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1950
#SBATCH --nodes=1

if [[ -z "$ML_FOLD" ]]; then
  echo "Must provide ML_FOLD in environment" 1>&2
  exit 1
fi

export DASK_LOGGING__DISTRIBUTED=WARN
export DASK_INTERFACE="${DASK_INTERFACE:-lo}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-${TMPDIR:-/tmp}}"
export TMPDIR="${TMPDIR:-/tmp}"

echo -e "JOB_ID=${SLURM_JOB_ID}"
echo -e "OMP_THREADS=${OMP_NUM_THREADS}"

echo -e "########\nCONTAINER START"

SIF="$HOME/fe4femo/ml_analysis/ml_analysis.sif"

if [[ ! -f "${SIF}" ]]; then
  echo "SIF image not found: ${SIF}" 1>&2
  exit 1
fi

# Copy SIF to node-local scratch to avoid squashfuse failures over NFS
# when multiple tasks read the same file simultaneously.
local_sif="${TMPDIR}/ml_analysis.sif"
echo "Copying SIF to local scratch: ${local_sif}"
cp "${SIF}" "${local_sif}"

if [ "$ML_FOLD" = "-1" ]; then
  RUN_COMMAND=("/app/slurm_scripts/fold_connector.sh")
else
  RUN_COMMAND=("/app/slurm_scripts/slurm_fork_tracker.sh")
fi

if [[ $SLURMD_NODENAME =~ "compute" ]]; then
  apptainer_cmd="$HOME/apptainer-dir/bin/apptainer"
else
  apptainer_cmd="apptainer"
fi
echo -e "NODE=${SLURMD_NODENAME}  APPTAINER=${apptainer_cmd}"

srun --exact "${apptainer_cmd}" exec \
  --tmpdir "${TMPDIR}" \
  --bind "$HOME:$HOME" \
  --bind "$HOME/fe4femo/ml_analysis:/app" \
  --pwd /app \
  "${local_sif}" "${RUN_COMMAND[@]}" "$@"
