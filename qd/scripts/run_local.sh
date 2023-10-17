#!/bin/bash
# Runs the QD code. Make sure to start the evaluation server first so that the
# code can query it.
#
# Usage:
#   bash scripts/run_local.sh CONFIG SEED SERVER_ADDRESS [-p PROJECT_DIR] [-r RELOAD_PATH]
# Example:
#   # Configuration config/foo.gin and seed 1, and binding the `/project/`
#   # directory and reloading from old_dir/.
#   bash scripts/run_local.sh \
#     config/foo.gin \
#     1 \
#     http://localhost:5000 \
#     -p /project \
#     -r old_dir

print_header() {
  echo
  echo "------------- $1 -------------"
}

# Prints "=" across an entire line.
print_thick_line() {
  printf "%0.s=" $(seq 1 `tput cols`)
  echo
}

print_header "Create logging directory"
DATE="$(date +'%Y-%m-%d_%H-%M-%S')"
LOGDIR="local_logs/local_${DATE}"
mkdir -p "$LOGDIR"
echo "LOCAL Log directory: ${LOGDIR}"

#
# Parse command line flags.
#
USAGE="Usage: bash scripts/run_local.sh CONFIG SEED SERVER_ADDRESS [-p PROJECT_DIR] [-r RELOAD_PATH]"

CONFIG="$1"
SEED="$2"
SERVER_ADDRESS="$3"
shift 3

while getopts "p:r:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      r) RELOAD_PATH=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done

echo "RELOAD_PATH: ${RELOAD_PATH}"

# An old option we used was: --env MALLOC_TRIM_THRESHOLD_=0
# But the older Singularity 3.0 on our machine does not seem to handle
# environment variables passed in this manner, so we exclude it.
SINGULARITY_OPTS="--cleanenv --no-home --bind $PWD"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:/project"
fi
echo "Singularity opts: ${SINGULARITY_OPTS}"

if [ -z "${SERVER_ADDRESS}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -n "$RELOAD_PATH" ]; then
  RELOAD_ARG="--reload ${RELOAD_PATH}"
else
  RELOAD_ARG=""
fi

set -u  # Uninitialized vars are error.

#
# Run the experiment.
#

print_header "Running experiment"
EXPERIMENT_OUT="$LOGDIR/experiment.out"
echo "(Output goes to $EXPERIMENT_OUT)"
echo
print_thick_line
# shellcheck disable=SC2086
singularity exec ${SINGULARITY_OPTS} --nv container.sif \
  python -m src.main \
    --config "$CONFIG" \
    --seed "$SEED" \
    --local-logdir "$LOGDIR" \
    --address "$SERVER_ADDRESS" \
    $RELOAD_ARG 2>&1 | tee "$EXPERIMENT_OUT"
print_thick_line

#
# Clean Up.
#

print_header "Cleanup"
echo "No cleanup needed"
