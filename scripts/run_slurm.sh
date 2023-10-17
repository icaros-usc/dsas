#!/bin/bash
# Runs scripts on an HPC with Slurm installed.
#
# This script sources an HPC_CONFIG, which is a bash script with the following
# variables defined:
# - HPC_SLURM_ACCOUNT (the slurm username for running the jobs)
# - HPC_SLURM_TIME (time limit for the jobs, specified as HH:MM:SS)
# - HPC_SLURM_CPUS_PER_NODE (number of CPUs per node; note we always run one
#   task per node)
# - HPC_SLURM_MEM_PER_CPU (memory per CPU)
# - HPC_MASTER_GPU (set to any string to indicate that a GPU should be used on
#   the master node; otherwise leave it out or set it to empty string)
#
# For instance, a file might look like:
#
#   HPC_SLURM_ACCOUNT=account_123
#   HPC_SLURM_TIME=20:00:00
#   HPC_SLURM_CPUS_PER_NODE=12
#   HPC_SLURM_MEM_PER_CPU=2GB
#   HPC_MASTER_GPU=true
#
# Other options:
# - Pass -d to perform a dry run (i.e. don't submit any scripts).
# - Pass -r LOGDIR to reload from an existing logging directory and continue an
#   experiment.
#
# NOTE: If you do not have a /project directory on your cluster, comment out the
# PROJECT_DIR variable below.
#
# Usage:
#   bash scripts/run_slurm.sh QD_CONFIG OPENRAVE_SERVER_CONFIG SEED HPC_CONFIG [-d] [-r LOGDIR]
#
# Example:
#   bash scripts/run_slurm.sh config/foo.gin src/simple_environment/src/simple_environment/search/config/experiment/foo.toml 42 config/hpc/foo.sh
#
#   # Dry run version of the above.
#   bash scripts/run_slurm.sh config/foo.gin src/simple_environment/src/simple_environment/search/config/experiment/foo.toml 42 config/hpc/foo.sh -d
#
#   # Run the above, with reloading from old_dir/
#   bash scripts/run_slurm.sh config/foo.gin src/simple_environment/src/simple_environment/search/config/experiment/foo.toml 42 config/hpc/foo.sh -r old_dir/

print_header() {
  echo
  echo "------------- $1 -------------"
}

#
# Set singularity opts -- comment out PROJECT_DIR if you do not have /project on
# your cluster.
#

PROJECT_DIR="/project"
SINGULARITY_OPTS="--cleanenv"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:/project"
fi
echo "Singularity opts: ${SINGULARITY_OPTS}"

#
# Parse command line flags.
#

QD_CONFIG="$1"
OPENRAVE_SERVER_CONFIG="$2"
SEED="$3"
HPC_CONFIG="$4"
shift 4  # Remove first 4 parameters so getopts does not see them.

if [ -z "$HPC_CONFIG" ]
then
  echo "Usage: bash scripts/run_slurm.sh QD_CONFIG OPENRAVE_SERVER_CONFIG SEED HPC_CONFIG [-d] [-r LOGDIR]"
  exit 1
fi

# For more info on getopts, see https://bytexd.com/using-getopts-in-bash/
DRY_RUN=""
RELOAD_ARG=""
while getopts "dr:" opt; do
  case $opt in
    d)
      echo "Using DRY RUN"
      DRY_RUN="1"
      ;;
    r)
      echo "Using RELOAD: $OPTARG"
      RELOAD_ARG="--reload $OPTARG"
      ;;
  esac
done

#
# Parse HPC config.
#

# Defines HPC_SLURM_ACCOUNT, HPC_SLURM_TIME, HPC_SLURM_NUM_NODES,
# HPC_SLURM_CPUS_PER_NODE, HPC_SLURM_MEM_PER_CPU, and maybe HPC_MASTER_GPU.
source "$HPC_CONFIG"

if [ -z "$HPC_SLURM_ACCOUNT" ] ||
   [ -z "$HPC_SLURM_TIME" ] ||
   [ -z "$HPC_SLURM_CPUS_PER_NODE" ] ||
   [ -z "$HPC_SLURM_MEM_PER_CPU" ]
then
echo "\
HPC_CONFIG must have the following variables defined:
- HPC_SLURM_ACCOUNT
- HPC_SLURM_TIME
- HPC_SLURM_CPUS_PER_NODE
- HPC_SLURM_MEM_PER_CPU"
  exit 1
fi

if [ -z "$HPC_MASTER_GPU" ]; then
  HPC_MASTER_GPU=""  # Make sure HPC_MASTER_GPU is initialized.
fi

set -u  # Uninitialized vars are error.

#
# Build and submit slurm scripts.
#

# Global storage holding all job ids. Newline-delimited string, where each line
# holds name;job_id.
JOB_IDS=""

# Submits a script and records it in JOB_IDS.
submit_script() {
  name="$1"
  slurm_script="$2"
  output=$(sbatch --parsable "$slurm_script")
  IFS=';' read -ra tokens <<< "$output"
  job_id="${tokens[0]}"
  JOB_IDS="${JOB_IDS}${name};${job_id}\n"
  echo "Submitted $job_id ($name)"
}

print_header "Create logging directory"
DATE="$(date +'%Y-%m-%d_%H-%M-%S')"
LOGDIR="${PWD}/slurm_logs/slurm_${DATE}"
echo "SLURM Log directory: ${LOGDIR}"

# Save config.
mkdir -p "$LOGDIR/config"
cp "$HPC_CONFIG" "$LOGDIR/config/"

print_header "Submitting slurm file"
SCRIPT_NAME="${LOGDIR}/main.slurm"
OUTPUT_NAME="${LOGDIR}/main.out"
ROSCORE_OUTPUT_NAME="${LOGDIR}/roscore.out"
OPENRAVE_OUTPUT_NAME="${LOGDIR}/openrave.out"
QD_OUTPUT_NAME="${LOGDIR}/qd.out"
# 1 CPU for scheduler, 1 CPU for main script, and a couple extra workers.
NUM_CPUS=$HPC_SLURM_CPUS_PER_NODE
SERVER_PORT=$((8786 + 10 + $SEED))
echo "Starting experiment from: ${SCRIPT_NAME}"

echo "\
#!/bin/bash
#SBATCH --job-name=${DATE}_main
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=$NUM_CPUS
#SBATCH --mem-per-cpu=$HPC_SLURM_MEM_PER_CPU
#SBATCH --time=$HPC_SLURM_TIME
#SBATCH --account=$HPC_SLURM_ACCOUNT
#SBATCH --output $OUTPUT_NAME
#SBATCH --error $OUTPUT_NAME
$(if [ -n "$HPC_MASTER_GPU" ]; then echo -e "#SBATCH --partition=gpu\n#SBATCH --gres=gpu:1"; fi)

echo
echo \"========== Start ==========\"
date

$(if [ -n "$HPC_MASTER_GPU" ]; then echo "module load cuda/10.2.89"; fi)

# Start roscore.
singularity exec ${SINGULARITY_OPTS} --bind ${PWD}/src:/usr/project/catkin/src openrave_container.sif \\
  rosmaster --core 2>&1 | tee "${ROSCORE_OUTPUT_NAME}" &

# Start the openrave server.
singularity exec ${SINGULARITY_OPTS} \\
  --bind ${PWD}/src:/usr/project/catkin/src \\
  --pwd /usr/project/catkin/src/simple_environment/src/simple_environment \\
  openrave_container.sif \\
  python search/server.py \\
  -c ${OPENRAVE_SERVER_CONFIG} \\
  -p ${SERVER_PORT} \\
  -nc ${NUM_CPUS} 2>&1 | tee "${OPENRAVE_OUTPUT_NAME}" &

sleep 10  # Wait for server to start.

# Start the QD client.
singularity exec ${SINGULARITY_OPTS} \\
  $(if [ -n "$HPC_MASTER_GPU" ]; then echo "--nv"; fi) \\
  --pwd ${PWD}/qd/ \\
  qd/container.sif \\
  python -m src.main \
    --config "${QD_CONFIG}" \
    --seed "${SEED}" \
    --slurm_logdir "$LOGDIR" \
    --address "http://localhost:${SERVER_PORT}" \
    $RELOAD_ARG 2>&1 | tee "${QD_OUTPUT_NAME}"

echo
echo \"========== Done ==========\"
date" > "$SCRIPT_NAME"

# Submit the scheduler script.
if [ -z "$DRY_RUN" ]; then submit_script "main" "$SCRIPT_NAME"; fi

#
# Print monitoring instructions.
#

print_header "Monitoring Instructions"
echo "\
To view output from the scheduler and main script, run:

  tail -f $OUTPUT_NAME
"

#
# Print cancellation instructions.
#

if [ -n "$DRY_RUN" ]
then
  print_header "Skipping cancellation, dashboard, postprocessing instructions"
  exit 0
fi

# Record job ids in logging directory. This can be picked up by
# scripts/slurm_cancel.sh in order to cancel the job.
echo -n -e "$JOB_IDS" > "${LOGDIR}/job_ids.txt"

print_header "Canceling"
echo "\
To cancel this job, run:

  bash scripts/slurm_cancel.sh $LOGDIR
"

#
# Print postprocessing instructions.
#

print_header "Postprocessing"
echo "\
Once this script has terminated, move these Slurm outputs to the experiment's
logging directory using:

  bash scripts/slurm_postprocess.sh $LOGDIR
"
