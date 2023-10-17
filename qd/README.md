# QD

Based on [DSAGE repo](https://github.com/icaros-usc/dsage).

## Contents

* [Manifest](#manifest)
* [Installation](#installation)
* [Instructions](#instructions)
  * [Running the Simulation Server](#running-the-simulation-server)
  * [Running QD Search](#running-qd-search)
    * [Single Run](#single-run)
    * [Running on Slurm](#running-on-slurm)
    * [Reloading](#reloading)
    * [Logging Directory Manifest](#logging-directory-manifest)

## Manifest

- `config/`: [gin](https://github.com/google/gin-config) configuration files.
- `src/`: Python implementation and related tools.
- `scripts/`: Bash scripts.

## Installation

1. **Install Singularity:** All of our code runs in a Singularity container.
   Singularity is a container platform (similar in many ways to Docker). Please
   see the instructions
   [here](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) for
   installing Singularity 3.0. Singularity 3.0 is compatible with Ubuntu 14; the
   latest versions require dependencies that are too new.
1. **Install CPLEX:** CPLEX is used for creating MIPs.
   1. Get the free academic edition
      [here](https://www.ibm.com/products/ilog-cplex-optimization-studio).
   1. Download the installation file for Linux. This file will be named
      something like `IBM ILOG CPLEX Optimization Studio 20.10 for Linux x86-64`
   1. Follow the instructions for installing CPLEX on Linux
      [here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/ilog.odms.studio.help/Optimization_Studio/topics/COS_installing.html).
      Basically:
      ```bash
      chmod u+x INSTALLATION_FILE
      ./INSTALLATION_FILE
      ```
      During installation, set the installation directory to `CPLEX_Studio201/`
      inside this directory.
1. **Build the Container:** Build the Singularity container with
   ```bash
   sudo make container.sif
   ```
1. **Install NVIDIA Drivers and CUDA:** The node where the main script runs
   should have a GPU with NVIDIA drivers and CUDA installed (in the future, we
   may try to put CUDA in the container instead).

## Instructions

In this setup, we connect the QD code in this directory with the simulation
code. The simulation code is turned into a Python 2 Flask server which
handles requests for OpenRave evaluations while this code uses pyribs on Python
3 and queries the server for evaluations.

### Running the Simulation Server

**Execute these from the root directory of the repo (one level above this
directory).**

See the [main README](README.md)

### Running QD Search

#### Single Run

To run one experiment locally, use:

```bash
bash scripts/run_local.sh CONFIG SEED NUM_WORKERS
```

For instance, with 4 workers:

```bash
bash scripts/run_local.sh config/foo.gin 42 4
```

`CONFIG` is the [gin](https://github.com/google/gin-config) experiment config
for `src.main`.

#### Running on Slurm

**Execute these from the root directory of the repo (one level above this
directory).**

Use the following command to run an experiment on an HPC with Slurm (and
Singularity) installed:

```bash
bash scripts/run_slurm.sh CONFIG SEED HPC_CONFIG
```

For example:

```bash
bash scripts/run_slurm.sh config/foo.gin 42 config/hpc/test.sh
```

`CONFIG` is the path (relative to this directory) to the experiment config for `src.main`,
and `HPC_CONFIG` is a shell file that is sourced by the script to provide configuration
for the Slurm cluster. See `scripts/hpc` in the root directory for example files. See
`scripts/run_experiments.sh` for examples on running experiments on Slurm.

#### Reloading

While the experiment is running, its state is saved to `reload.pkl` in the
logging directory. If the experiment fails, e.g. due to memory limits, time
limits, or network connection issues, `reload.pkl` may be used to continue the
experiment. To do so, execute the same command as before, but append the path to
the logging directory of the failed experiment.

```bash
bash scripts/run_slurm.sh config/foo.gin 42 config/hpc/test.sh -r logs/.../
```

The experiment will then run to completion in the same logging directory. This
works with `scripts/run_local.sh` too.

#### Logging Directory Manifest

Regardless of where the script is run, the log files and results are placed in a
logging directory in `logs/`. The directory's name is of the form
`%Y-%m-%d_%H-%M-%S_dashed-name`, e.g. `2020-12-01_15-00-30_experiment-1`. Inside
each directory are the following files:

```text
- config.gin  # All experiment config variables, lumped into one file.
- seed  # Text file containing the seed for the experiment.
- reload.pkl  # Data necessary to reload the experiment if it fails.
- reload_em.pkl  # Pickle data for EmulationModel.
- reload_em.pth  # PyTorch models for EmulationModel.
- metrics.json  # Data for a MetricLogger with info from the entire run, e.g. QD score.
- hpc_config.sh  # Same as the config in the Slurm dir, if Slurm is used.
- archive/  # Snapshots of the full archive, including solutions and metadata,
            # in pickle format.
- archive_history.pkl  # Stores objective values and behavior values necessary
                       # to reconstruct the archive. Solutions and metadata are
                       # excluded to save memory.
- dashboard_status.txt  # Job status which can be picked up by dashboard scripts.
                        # Only used during execution.
- slurm_YYYY-MM-DD_HH-MM-SS/  # Slurm log dir (only exists if using Slurm).
                              # There can be a few of these if there were reloads.
  - config/
    - config.sh  # Possibly has a different name.
  - job_ids.txt  # Job IDs; can be used to cancel job (scripts/slurm_cancel.sh).
  - logdir  # File containing the name of the main logdir.
  - main.slurm  # Slurm script for scheduler and experiment invocation.
  - main.out  # Combined stdout and stderr from running main.slurm.
  - roscore.out  # Roscore output
  - openrave.out  # Simulation code output
  - qd.out  # QD code output
```
