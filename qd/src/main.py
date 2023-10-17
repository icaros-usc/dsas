"""Entry point for running all experiments."""
import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import fire
import gin
import torch
from logdir import LogDir

import src.constraints
import src.sol_postprocessors
from src.client import Client, MockClient
from src.manager import Manager
from src.utils.logging import setup_logging


def load_configs(config_file: str):
    """Loads gin configuration file.

    If the name is of the form X_test, both X and config/test.gin are loaded.
    """
    if config_file.endswith("_test"):
        config_file = config_file[:-len("_test")]
        is_test = True
    else:
        is_test = False

    # TODO: May want to place config/test.gin within src/ in case main.py
    # is not run from root dir of repo.
    gin.parse_config_file(config_file)
    if is_test:
        gin.parse_config_file("config/test.gin")

        # Append " Test" to the experiment name.
        gin.bind_parameter("experiment.name",
                           gin.query_parameter("experiment.name") + " Test")

    eval_configs = [
        "GaussianEmitter.sigma", "GaussianEmitter.bounds", "GaussianEmitter.x0",
        "EvolutionStrategyEmitter.x0", "ws1/EvolutionStrategyEmitter.x0",
        "ws2/EvolutionStrategyEmitter.x0", "GradientArborescenceEmitter.x0"
    ]

    # Note: We use print() below because using logging.warning() or
    # logging.info() seems to kill our logging configuration, such that nothing
    # gets logged afterwards because the default logging behavior is to not show
    # the logging messages.

    for ec in eval_configs:
        try:
            ec_val = gin.query_parameter(ec)
        except ValueError:
            print(f"Config not found for {ec}, ignoring it.")
            continue
        if isinstance(ec_val, str):
            gin.bind_parameter(ec, eval(ec_val))  # pylint: disable = eval-used

    # Bind required functions that were passed as a string in gin.
    func_configs = [
        "EvolutionStrategyEmitter.constraint_func",
        "ScenarioConfig.sol_postprocessing_func"
    ]
    func_modules = [src.constraints, src.sol_postprocessors]
    for fc, fm in zip(func_configs, func_modules):
        try:
            fc_val = gin.query_parameter(fc)
            if isinstance(fc_val, str):
                gin.bind_parameter(fc, getattr(fm, fc_val))
        except ValueError:
            print(f"Unable to bind {fc}, ignoring it.")
        except AttributeError:
            print(f"No matching function for {fc}, ignoring it.")

    gin.finalize()

    return is_test


def check_env():
    """Environment check(s)."""
    assert os.environ['OPENBLAS_NUM_THREADS'] == '1', \
        ("OPENBLAS_NUM_THREADS must be set to 1 so that the numpy in each "
         "worker does not throttle each other. If you are running in the "
         "Singularity container, this should already be set.")


def setup_logdir(seed: int,
                 slurm_logdir: Union[str, Path],
                 local_logdir: Union[str, Path],
                 reload_dir: Optional[str] = None):
    """Creates the logging directory with a LogDir object.

    Args:
        seed: Master seed.
        slurm_logdir: Directory for storing Slurm logs. Pass None if not
            applicable.
        reload_dir: Directory for reloading. If passed in, this directory will
            be reused as the logdir.
    """
    name = gin.query_parameter("experiment.name")

    if reload_dir is not None:
        # Reuse existing logdir.
        reload_dir = Path(reload_dir)
        logdir = LogDir(name, custom_dir=reload_dir)
    else:
        # Create new logdir.
        logdir = LogDir(name, rootdir="./logs", uuid=True)

    # Save configuration options.
    with logdir.pfile("config.gin").open("w") as file:
        file.write(gin.config_str(max_line_length=120))

    # Write a README.
    logdir.readme(git_commit=False, info=[f"Seed: {seed}"])

    # Write the seed.
    with logdir.pfile("seed").open("w") as file:
        file.write(str(seed))

    if slurm_logdir is not None:
        # Write the logging directory to the slurm logdir.
        with (Path(slurm_logdir) / "logdir").open("w") as file:
            file.write(str(logdir.logdir))

        # Copy the hpc config.
        hpc_config = glob.glob(str(Path(slurm_logdir) / "config" / "*.sh"))[0]
        hpc_config_copy = logdir.file("hpc_config.sh")
        shutil.copy(hpc_config, hpc_config_copy)

    if local_logdir is not None:
        # Write the logging directory to the local logdir.
        with (Path(local_logdir) / "logdir").open("w") as file:
            file.write(str(logdir.logdir))

    return logdir


@gin.configurable(denylist=["client", "logdir", "seed", "reload"])
def experiment(client: Client,
               logdir: LogDir,
               seed: int,
               reload: bool = False,
               name: str = gin.REQUIRED):
    """Executes a distributed experiment on Dask.

    Args:
        client: A Dask client for running distributed tasks.
        logdir: A logging directory instance for recording info.
        seed: Master seed for the experiment.
        reload: Whether to reload experiment from logdir.
        name: Name of the experiment.
    """
    logging.info("Experiment Name: %s", name)

    # All further configuration to Manager is handled by gin.
    Manager(
        client=client,
        logdir=logdir,
        seed=seed,
        reload=reload,
    ).execute()


def main(
    config: str,
    seed: int,
    address: str,
    reload: str = None,
    slurm_logdir=None,
    local_logdir=None,
):
    """Parses command line flags and sets up and runs experiment.

    Args:
        config: GIN configuration file. To pass a test config for `X`, pass in
            `X_test`. Then, `X` and `config/test.gin` will be included.
        address: Evaluation server address.
        seed: Master seed.
        reload: Path to previous logging directory for reloading the
            algorithm. New logs are also stored in this directory.
        slurm_logdir: Directory storing slurm output.
        local_logdir: Directory storing local output.
    """
    is_test = load_configs(config)
    check_env()

    logdir = setup_logdir(seed, slurm_logdir, local_logdir, reload)
    client = MockClient(address) if is_test else Client(address)

    setup_logging(on_worker=False)

    logging.info("Master Seed: %d", seed)
    logging.info("Logging Directory: %s", logdir.logdir)
    logging.info("Client pulse check: %s", client.hello())
    logging.info("Server CPUs: %d", client.ncores())
    logging.info("===== Config: =====\n%s", gin.config_str())

    # PyTorch seeding is tricky. However, seeding here should be enough because
    # we only use PyTorch randomness in the initialization of the network. If we
    # use randomness during the iterations, reloading will not be "correct"
    # since we would be re-seeding at a generation other than the first. See
    # here for more info: https://pytorch.org/docs/stable/notes/randomness.html
    # By the way, we add 42 in order to avoid using the same seed as other
    # places.
    torch.manual_seed(seed + 42)

    experiment(
        client=client,
        logdir=logdir,
        seed=seed,
        reload=reload is not None,
    )


if __name__ == "__main__":
    fire.Fire(main)
