#!/bin/bash
# Moves a slurm log dir to its main logging directory after its jobs are complete.
#
# Can also remove reload file(s) (pass -r) and worker logs (pass -w).
#
# Pass -z to zip the main logdir when done.
#
# Pass -e to remove emulation model files.
#
# Usage:
#   bash scripts/slurm_postprocess.sh SLURM_DIR [-r] [-w] [-z] [-e]
#
# Example:
#   bash scripts/slurm_postprocess.sh slurm_logs/slurm_.../

SLURM_DIR="$1"
LOGDIR="qd/$(cat "$SLURM_DIR/logdir")"
shift  # Remove SLURM_DIR from arg list.

echo "-> Cancelling slurm dir in case jobs still running"
bash scripts/slurm_cancel.sh "$SLURM_DIR"

# Refer to https://www.computerhope.com/unix/bash/getopts.htm and
# https://sookocheff.com/post/bash/parsing-bash-script-arguments-with-shopts/
while getopts "wrzt" opt; do
  case "$opt" in
    w)
      echo "-> Removing worker files"
      rm -v $SLURM_DIR/worker*
      ;;
    r)
      echo "-> Removing reload.pkl"
      rm -v "$LOGDIR/reload.pkl"
      ;;
    z)
      ZIP_MAIN="True"
      ;;
    e)
      echo "-> Remove reload_em.pkl and reload_em.pth"
      rm -v "$LOGDIR/reload_em.pkl" "$LOGDIR/reload_em.pth"
      ;;
  esac
done

echo "-> Moving slurm dir to main logdir"
mv -v "$SLURM_DIR" "$LOGDIR"

if [ -n "$ZIP_MAIN" ]; then
  echo "-> Zipping main logdir"
  logdir_dir="$(dirname $LOGDIR)"
  logdir_base="$(basename $LOGDIR)"
  (cd "$logdir_dir" && zip -r "${logdir_base}.zip" "${logdir_base}")
fi
