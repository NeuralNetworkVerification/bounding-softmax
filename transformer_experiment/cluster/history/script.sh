#!/bin/bash
#SBATCH -e /dev/null
#SBATCH -o /dev/null
#SBATCH -c 2
#SBATCH -a 1-25200
#SBATCH --qos=normal
#SBATCH --partition=octa
#SBATCH -t 00:00:1200
#SBATCH --mem=8000M
#SBATCH -D /barrett/scratch/haozewu/softmax/bounding-softmax/transformer_experiment/cluster/history

set -e -o pipefail

PRIVATE_TMP="/tmp/slurm.${SLURM_JOB_ID:=0}.${SLURM_RESTART_COUNT:=0}"
mkdir "${PRIVATE_TMP}"
export PRIVATE_TMP

out="slurm-${SLURM_ARRAY_TASK_ID}"
log="/barrett/scratch/haozewu/softmax/bounding-softmax/transformer_experiment/cluster/history/${out}.log"
err="/barrett/scratch/haozewu/softmax/bounding-softmax/transformer_experiment/cluster/history/${out}.err"

(
  ARGUMENTS="$(sed ${SLURM_ARRAY_TASK_ID}'q;d' /barrett/scratch/haozewu/softmax/bounding-softmax/transformer_experiment/cluster/history/benchmarks)"
  COMMAND="./run_experiment.sh  $ARGUMENTS"

  echo "c host: $(hostname)"
  echo "c tmpdir: ${PRIVATE_TMP}"
  echo "c start: $(date)"
  echo "c arrayjobid: ${SLURM_ARRAY_JOB_ID}"
  echo "c jobid: ${SLURM_JOB_ID}"
  echo "c command: $COMMAND"
  echo "c arguments: $ARGUMENTS"
  export COMMAND
  # Create private /tmp directory for each job
  unshare -m --map-root-user /bin/bash -c '
  (
    mount --bind "${PRIVATE_TMP}" /tmp
    /barrett/scratch/local/bin/runlim --time-limit=1200 --space-limit=8000 ${COMMAND}
  )'

  [ -d "${PRIVATE_TMP}" ] && rm -rf "${PRIVATE_TMP}"
  echo "c done"

) > "$log" 2> "$err"
