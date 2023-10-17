"""Provides a class for running each QD algorithm."""
import dataclasses
import itertools
import logging
import pickle as pkl
from typing import Callable, List, Tuple

import cloudpickle
import gin
import numpy as np
from logdir import LogDir
from matplotlib import pyplot as plt

from ribs.archives import ArchiveBase, Elite
from ribs.visualize import grid_archive_heatmap

# Importing emitters makes them available for gin.
import src.emitters  # pylint: disable = unused-import
from src.archives import GridArchive
from src.client import Client
from src.scenario.scenario_manager import ScenarioManager
from src.schedulers import Scheduler
from src.utils.metric_logger import MetricLogger

# Just to get rid of pylint warning about unused import (adding a comment after
# each line above messes with formatting).
IMPORTS_FOR_GIN = (
    GridArchive,
    ScenarioManager,
)

EMITTERS_WITH_RESTARTS = ()

logger = logging.getLogger(__name__)


@gin.configurable
class Manager:  # pylint: disable = too-many-instance-attributes
    """Runs an (emulation model) QD algorithm on distributed compute.

    If you are trying to understand this code, first refer to how the general
    pyribs loop works (https://pyribs.org). Essentially, the execute() method of
    this class runs this loop but in a more complicated fashion, as we want to
    distribute the solution evaluations, log various performance metrics, save
    various pieces of data, support reloading / checkpoints, etc.

    Main args:
        client: Client for querying the server.
        logdir: Directory for saving all logging info.
        seed: Master seed. The seed is not passed in via gin because it needs to
            be flexible.
        reload: If True, reload the experiment from the given logging directory.
        env_manager_class: This class calls a separate manager based on the
            environment, such as ScenarioManager. Pass this class using this
            argument.

    Algorithm args:
        is_em: Whether this algorithm uses emulation models (EM).
        max_evals: Total number of evaluations of the true objective.
        initial_em_sols: Number of initial solutions to evaluate. Only applicable
            when using emulation models.
        inner_itrs: Number of times to run the inner loop.
        is_dqd: Whether the inner loop algorithm is a DQD algorithm and thus
            should use gradients.
        archive_type: Archive class for both main and emulation archives.
            Intended for gin configuration.
        train_archive_type: Archive class used only during search. The solutions
            in this archive will be added to an instance of archive_type at the
            end of the search. If train_archive_type is unspecified,
            archive_type is used both during search and for the final archive.
            (default: None)
        sol_size: Size of the solution that the emitter should emit and the
            archive should store.
        emitter_types: List of tuples of (class, n); where each tuple indicates
            there should be n emitters with the given class. If is_em, these
            emitters are only used in the inner loop; otherwise, they are
            maintained for the entire run. Intended for gin configuration.
        num_elites_to_eval: Number of elites in the emulation archive to
            evaluate. Pass None to evaluate all elites. (default: None)
        random_sample_em: True if num_elites_to_eval should be selected
            randomly. If num_elites_to_eval is None, this argument is
            ignored. (default: False)
        downsample_em: Whether to downsample the emulation archive.
        downsample_archive_type: Archive type for downsampling. Used for Gin.

    Logging args:
        archive_save_freq: Number of outer itrs to wait before saving the full
            archive (i.e. including solutions and metadata). Set to None to
            never save (the archive will still be available in the reload file).
            Set to -1 to only save on the final iter.
        reload_save_freq: Number of outer itrs to wait before saving
            reload data.
        plot_metrics_freq: Number of outer itrs to wait before displaying text
            plot of metrics. Plotting is not expensive, but the output can be
            pretty large.
        save_separate_em_data: True if surrogate model and the dataset should
            be stored separately for each iteration. By default, only the final
            model/dataset will be saved.
    """

    def __init__(
        self,
        ## Main args ##
        client: Client,
        logdir: LogDir,
        seed: int,
        reload: bool = False,
        env_manager_class: Callable = gin.REQUIRED,
        ## Algorithm args ##
        is_em: bool = gin.REQUIRED,
        max_evals: int = gin.REQUIRED,
        initial_em_sols: int = gin.REQUIRED,
        inner_itrs: int = gin.REQUIRED,
        is_dqd: bool = gin.REQUIRED,
        archive_type: Callable = gin.REQUIRED,
        train_archive_type: Callable = None,
        sol_size: int = gin.REQUIRED,
        emitter_types: List[Tuple] = gin.REQUIRED,
        num_elites_to_eval: int = None,
        random_sample_em: bool = False,
        downsample_em: bool = False,
        downsample_archive_type: Callable = None,
        ## Logging args ##
        archive_save_freq: int = None,
        reload_save_freq: int = 5,
        plot_metrics_freq: int = 5,
        save_separate_em_data: bool = False,
    ):  # pylint: disable = too-many-arguments, too-many-branches

        # Main.
        self.client = client
        self.logdir = logdir

        # Algorithm.
        self.is_em = is_em
        self.max_evals = max_evals
        self.inner_itrs = inner_itrs
        self.initial_em_sols = initial_em_sols
        self.is_dqd = is_dqd
        self.archive_type = archive_type
        self.train_archive_type = train_archive_type
        self.sol_size = sol_size
        self.emitter_types = emitter_types
        self.num_elites_to_eval = num_elites_to_eval
        self.random_sample_em = random_sample_em
        self.downsample_em = downsample_em
        self.downsample_archive_type = downsample_archive_type

        # Logging.
        self.archive_save_freq = archive_save_freq
        self.reload_save_freq = reload_save_freq
        self.plot_metrics_freq = plot_metrics_freq
        self.save_separate_em_data = save_separate_em_data

        # Set up the environment manager.
        self.env_manager = env_manager_class(self.client)
        self.surrogate_data = []

        # The attributes below are either reloaded or created fresh. Attributes
        # added below must be added to the _save_reload_data() method.
        if not reload:
            logger.info("Setting up fresh components")
            self.rng = np.random.default_rng(seed)
            self.outer_itrs_completed = 0
            self.evals_used = 0

            metric_list = [
                ("Total Evals", True),
                ("Mean Evaluation", False),
                ("Actual QD Score", True),
                ("Archive Size", True),
                ("Archive Coverage", True),
                ("Best Objective", False),
                ("Worst Objective", False),
                ("Mean Objective", False),
                ("Overall Min Objective", False),
            ]

            if train_archive_type is not None and not self.is_em:
                extra_metrics = [
                    (f"(Train archive) {e[0]}", e[1]) for e in metric_list
                ]
                metric_list.extend(extra_metrics)

            self.metrics = MetricLogger(metric_list)
            self.total_evals = 0
            self.overall_min_obj = np.inf

            self.metadata_id = 0
            self.cur_best_id = None  # ID of most recent best solution.

            self.failed_levels = []

            if self.is_em:
                logger.info("Setting up emulation model and archive")
                # Archive must be initialized since there is no scheduler.
                self.env_manager.em_init(seed)
                self.archive: ArchiveBase = archive_type(
                    solution_dim=self.sol_size,
                    seed=seed,
                    dtype=np.float32,
                )
                self.log_archive_info(self.archive, "Archive")
                self.train_archive = None
            else:
                logger.info("Setting up scheduler for classic pyribs")
                if train_archive_type is None:
                    train_archive = archive_type(
                        solution_dim=self.sol_size,
                        seed=seed,
                        dtype=np.float32,
                    )
                    final_archive = train_archive
                else:
                    logger.info("Creating a separate archive for search")
                    train_archive = train_archive_type(
                        solution_dim=self.sol_size,
                        seed=seed,
                        dtype=np.float32,
                    )
                    final_archive = archive_type(
                        solution_dim=self.sol_size,
                        seed=seed,
                        dtype=np.float32,
                    )
                _, self.scheduler = self.build_emitters_and_scheduler(
                    train_archive)
                logger.info("Scheduler: %s", self.scheduler)
                # Set self.archive too for ease of reference.
                self.archive = final_archive
                self.train_archive = train_archive
                self.log_archive_info(self.archive, "Archive")
                self.log_archive_info(self.train_archive, "Train Archive")
        else:
            logger.info("Reloading scheduler and other data from logdir")

            with open(self.logdir.pfile("reload.pkl"), "rb") as file:
                data = pkl.load(file)
                self.rng = data["rng"]
                self.outer_itrs_completed = data["outer_itrs_completed"]
                self.total_evals = data["total_evals"]
                self.metrics = data["metrics"]
                self.overall_min_obj = data["overall_min_obj"]
                self.metadata_id = data["metadata_id"]
                self.cur_best_id = data["cur_best_id"]
                self.failed_levels = data["failed_levels"]
                self.surrogate_data = data["surrogate_data"]
                if self.is_em:
                    self.archive = data["archive"]
                    self.log_archive_info(self.archive, "Archive")
                    self.train_archive = None
                else:
                    self.scheduler = data["scheduler"]
                    self.train_archive = self.scheduler.archive
                    self.archive = data["archive"]
                    self.log_archive_info(self.archive, "Archive")
                    self.log_archive_info(self.train_archive, "Train Archive")

            if self.is_em:
                self.env_manager.em_init(seed,
                                         self.logdir.pfile("reload_em.pkl"),
                                         self.logdir.pfile("reload_em.pth"))

            logger.info("Outer itrs already completed: %d",
                        self.outer_itrs_completed)
            logger.info("Execution continues from outer itr %d (1-based)",
                        self.outer_itrs_completed + 1)
            logger.info("Reloaded archive: %s", self.archive)
            logger.info("Reloaded train archive: %s", self.train_archive)

        logger.info("solution_dim: %d", self.archive.solution_dim)

        # Set the rng of the env manager
        self.env_manager.rng = self.rng

    @staticmethod
    def log_archive_info(archive: ArchiveBase, name: str):
        logger.info(
            "%s: %s (threshold_min: %f, learning_rate: %f, measure_dim: %d)",
            name,
            archive,
            archive.threshold_min,
            archive.learning_rate,
            archive.measure_dim,
        )

    def msg_all(self, msg: str):
        """Logs msg on master, on all workers, and in dashboard_status.txt."""
        logger.info(msg)
        self.client.log(msg)
        with self.logdir.pfile("dashboard_status.txt").open("w") as file:
            file.write(msg)

    def finished(self):
        """Whether execution is done."""
        return self.total_evals >= self.max_evals

    def save_reload_data(self):
        """Saves data necessary for a reload.

        Current reload files:
        - reload.pkl
        - reload_em.pkl
        - reload_em.pth

        Since saving may fail due to memory issues, data is first placed in
        reload-tmp.pkl. reload-tmp.pkl then overwrites reload.pkl.

        We use gin to reference emitter classes, and pickle fails when dumping
        things constructed by gin, so we use cloudpickle instead. See
        https://github.com/google/gin-config/issues/8 for more info.
        """
        logger.info("Saving reload data")

        logger.info("Saving reload-tmp.pkl")
        with self.logdir.pfile("reload-tmp.pkl").open("wb") as file:
            reload_data = {
                "rng": self.rng,
                "outer_itrs_completed": self.outer_itrs_completed,
                "total_evals": self.total_evals,
                "metrics": self.metrics,
                "overall_min_obj": self.overall_min_obj,
                "metadata_id": self.metadata_id,
                "cur_best_id": self.cur_best_id,
                "failed_levels": self.failed_levels,
                "surrogate_data": self.surrogate_data,
            }
            if self.is_em:
                reload_data["archive"] = self.archive
            else:
                # Saving archive since it might be different from train_archive
                reload_data["archive"] = self.archive
                reload_data["scheduler"] = self.scheduler

            cloudpickle.dump(reload_data, file)

        if self.is_em:
            logger.info("Saving reload_em-tmp.pkl and reload_em-tmp.pth")
            if self.save_separate_em_data:
                fname = f"reload_em_{self.outer_itrs_completed}"
            else:
                fname = "reload_em"
            self.env_manager.emulation_model.save(
                self.logdir.pfile(f"{fname}-tmp.pkl"),
                self.logdir.pfile(f"{fname}-tmp.pth"))

        logger.info("Renaming tmp reload files")
        self.logdir.pfile("reload-tmp.pkl").rename(
            self.logdir.pfile("reload.pkl"))
        if self.is_em:
            if self.save_separate_em_data:
                fname = f"reload_em_{self.outer_itrs_completed}"
            else:
                fname = "reload_em"
            self.logdir.pfile(f"{fname}-tmp.pkl").rename(
                self.logdir.pfile(f"{fname}.pkl"))
            self.logdir.pfile(f"{fname}-tmp.pth").rename(
                self.logdir.pfile(f"{fname}.pth"))

        logger.info("Finished saving reload data")

    def save_archive(self):
        """Saves dataframes of the archive.

        The archive, including solutions and metadata, is saved to
        logdir/archive/archive_{outer_itr}.pkl

        Note that the archive is saved as an ArchiveDataFrame storing common
        Python objects, so it should be stable (at least, given fixed software
        versions).
        """
        itr = self.outer_itrs_completed
        df = self.archive.as_pandas(include_solutions=True,
                                    include_metadata=True)
        df.to_pickle(self.logdir.file(f"archive/archive_{itr}.pkl"))

        if self.train_archive_type is not None and not self.is_em:
            df = self.train_archive.as_pandas(include_solutions=True,
                                              include_metadata=True)
            df.to_pickle(self.logdir.file(f"archive/train_archive_{itr}.pkl"))

    def save_archive_history(self):
        """Saves the archive's history.

        We are okay with a pickle file here because there are only numpy arrays
        and Python objects, both of which are stable.
        """
        with self.logdir.pfile("archive_history.pkl").open("wb") as file:
            pkl.dump(self.archive.history(), file)

        if self.train_archive_type is not None and not self.is_em:
            with self.logdir.pfile("train_archive_history.pkl").open(
                    "wb") as file:
                pkl.dump(self.train_archive.history(), file)

    def save_data(self):
        """Saves archive, reload data, history, and metrics if necessary.

        This method must be called at the _end_ of each outer itr. Otherwise,
        some things might not be complete. For instance, the metrics may be in
        the middle of an iteration, so when we reload, we get an error because
        we did not end the iteration.
        """
        if self.archive_save_freq is None:
            save_full_archive = False
        elif self.archive_save_freq == -1 and self.finished():
            save_full_archive = True
        elif (self.archive_save_freq > 0 and
              self.outer_itrs_completed % self.archive_save_freq == 0):
            save_full_archive = True
        else:
            save_full_archive = False

        logger.info("Saving metrics")
        self.metrics.to_json(self.logdir.file("metrics.json"))

        logger.info("Saving archive history")
        self.save_archive_history()

        if save_full_archive:
            logger.info("Saving full archive")
            self.save_archive()
        if ((self.outer_itrs_completed % self.reload_save_freq == 0) or
                self.finished()):
            self.save_reload_data()
        if self.finished():
            logger.info("Saving failed levels")
            self.logdir.save_data(self.failed_levels, "failed_levels.pkl")

    def plot_metrics(self):
        """Plots metrics every self.plot_metrics_freq itrs or on final itr."""
        if (self.outer_itrs_completed % self.plot_metrics_freq == 0 or
                self.finished()):
            logger.info("Metrics:\n%s", self.metrics.get_plot_text())

    def add_performance_metrics(self, archive):
        """Calculates various performance metrics at the end of each iter."""
        prefix = ""
        if (self.train_archive_type is not None and not self.is_em and
                archive is self.train_archive):
            prefix = "(Train archive) "
        df = archive.as_pandas(include_solutions=False)
        objs = df.objective_batch()
        stats = archive.stats

        self.metrics.add(
            f"{prefix}Total Evals",
            self.total_evals,
            logger,
        )
        self.metrics.add(
            f"{prefix}Actual QD Score",
            self.env_manager.actual_qd_score(objs),
            logger,
        )
        self.metrics.add(
            f"{prefix}Archive Size",
            stats.num_elites,
            logger,
        )
        self.metrics.add(
            f"{prefix}Archive Coverage",
            stats.coverage,
        )
        self.metrics.add(
            f"{prefix}Best Objective",
            np.max(objs),
            logger,
        )
        self.metrics.add(
            f"{prefix}Worst Objective",
            np.min(objs),
            logger,
        )
        self.metrics.add(
            f"{prefix}Mean Objective",
            np.mean(objs),
            logger,
        )
        self.metrics.add(
            f"{prefix}Overall Min Objective",
            self.overall_min_obj,
            logger,
        )

    def extract_metadata(self, r) -> dict:
        """Constructs metadata object from results of an evaluation."""
        # Creates a shallow copy of the dataclass -- we don't want to recurse
        # into the dataclass fields to try to convert everything to a dict,
        # which is what asdict does -- see
        # https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict
        meta = dict((field.name, getattr(r, field.name))
                    for field in dataclasses.fields(r))

        # Remove unwanted keys.
        none_keys = [key for key in meta if meta[key] is None]
        for key in itertools.chain(none_keys, []):
            try:
                meta.pop(key)
            except KeyError:
                pass

        meta["metadata_id"] = self.metadata_id
        self.metadata_id += 1

        return meta

    def build_emitters_and_scheduler(self, archive):
        """Builds pyribs components with the config params and given archive."""
        emitters = []
        for emitter_class, n_emitters in self.emitter_types:
            emitter_seeds = self.rng.integers(np.iinfo(np.int32).max / 2,
                                              size=n_emitters,
                                              endpoint=True)
            emitters.extend(
                [emitter_class(archive, seed=s) for s in emitter_seeds])
            logger.info("Constructed %d emitters of class %s - seeds %s",
                        n_emitters, emitter_class, emitter_seeds)
        logger.info("Emitters: %s", emitters)

        scheduler = Scheduler(archive, emitters)
        logger.info("Scheduler: %s", scheduler)

        return emitters, scheduler

    def build_emulation_archive(self) -> ArchiveBase:
        """Builds an archive which optimizes the emulation model."""
        logger.info("Setting up pyribs components")
        seed = self.rng.integers(np.iinfo(np.int32).max / 2, endpoint=True)

        if self.train_archive_type is not None:
            archive: ArchiveBase = self.train_archive_type(
                solution_dim=self.sol_size,
                seed=seed,
                dtype=np.float32,
                record_history=False)
        else:
            archive: ArchiveBase = self.archive_type(solution_dim=self.sol_size,
                                                     seed=seed,
                                                     dtype=np.float32,
                                                     record_history=False)
        logger.info("Surrogate archive: %s", archive)

        _, scheduler = self.build_emitters_and_scheduler(archive)

        for inner_itr in range(1, self.inner_itrs + 1):
            self.em_evaluate(scheduler)

            if inner_itr % 1000 == 0 or inner_itr == self.inner_itrs:
                logger.info("Completed inner iteration %d", inner_itr)

        logger.info("Generated emulation archive with %d elites (%f coverage)",
                    archive.stats.num_elites, archive.stats.coverage)

        # In downsampling, we create a smaller archive where the elite in each
        # cell is sampled from a corresponding region of cells in the main
        # archive.
        if self.downsample_em:
            downsample_archive: ArchiveBase = self.downsample_archive_type(
                solution_dim=archive.solution_dim,
                seed=seed,
                dtype=np.float32,
                record_history=False)
            scales = np.array(archive.dims) // np.array(downsample_archive.dims)

            # Iterate through every index in the downsampled archive.
            for downsample_idx in itertools.product(
                    *map(range, downsample_archive.dims)):

                # In each index, retrieve the corresponding elites in the main
                # archive.
                elites = []
                archive_ranges = [
                    range(scales[i] * downsample_idx[i],
                          scales[i] * (downsample_idx[i] + 1))
                    for i in range(archive.measure_dim)
                ]
                for idx in itertools.product(*archive_ranges):
                    int_idx = archive.grid_to_int_index([idx])[0]
                    # pylint: disable = protected-access
                    if archive._occupied_arr[int_idx]:
                        elites.append(
                            Elite(
                                archive._solution_arr[int_idx],
                                archive._objective_arr[int_idx],
                                archive._measures_arr[int_idx],
                                int_idx,
                                archive._metadata_arr[int_idx],
                            ))

                # Choose one of the elites to insert into the archive.
                if len(elites) > 0:
                    sampled_elite = elites[self.rng.integers(len(elites))]
                    downsample_archive.add_single(
                        sampled_elite.solution,
                        sampled_elite.objective,
                        sampled_elite.measures,
                        sampled_elite.metadata,
                    )

            archive = downsample_archive
            logger.info(
                "Downsampled emulation archive has %d elites (%f coverage)",
                archive.stats.num_elites, archive.stats.coverage)

        return archive

    def em_evaluate(self, scheduler):
        """Asks for solutions from the scheduler, evaluates using the emulation
        model, and tells the objective and measures.

        Args:
            scheduler: Scheduler to use
        """
        if self.is_dqd:
            sols = scheduler.ask_dqd()
            objs, measures, jacobians, success_mask = \
                self.env_manager.emulation_pipeline(sols, grad_estimate=True)

            all_objs = np.full(len(sols), np.nan)
            all_measures = np.full((len(sols), measures.shape[1]), np.nan)
            all_jacobians = np.full([len(sols)] + list(jacobians.shape[1:]),
                                    np.nan)
            all_objs[success_mask] = objs
            all_measures[success_mask] = measures
            all_jacobians[success_mask] = jacobians

            scheduler.tell_dqd(all_objs,
                               all_measures,
                               all_jacobians,
                               success_mask=success_mask)

        sols = scheduler.ask()
        objs, measures, success_mask = \
            self.env_manager.emulation_pipeline(sols, grad_estimate=False)

        all_objs = np.full(len(sols), np.nan)
        all_measures = np.full((len(sols), measures.shape[1]), np.nan)
        all_objs[success_mask] = objs
        all_measures[success_mask] = measures

        scheduler.tell(all_objs, all_measures, success_mask=success_mask)

    def evaluate_solutions(self, sols):
        """Evaluates a batch of solutions and adds them to the archive."""
        logger.info("Evaluating solutions")

        skipped_sols = 0
        if self.total_evals + len(sols) > self.max_evals:
            remaining_evals = self.max_evals - self.total_evals
            remaining_sols = remaining_evals
            skipped_sols = len(sols) - remaining_sols
            sols = sols[:remaining_sols]
            logger.info(
                "Unable to evaluate all solutions; will evaluate %d instead",
                remaining_sols,
            )

        logger.info("total_evals (old): %d", self.total_evals)
        self.total_evals += len(sols)
        logger.info("total_evals (new): %d", self.total_evals)

        logger.info("Running evaluations")
        eval_kwargs = {}

        results = self.env_manager.eval_pipeline(sols, eval_kwargs)

        if self.is_em:
            logger.info(
                "Adding solutions to main archive and surrogate dataset")
        else:
            logger.info("Adding solutions to the scheduler")

        objs = []
        measures, metadata, success_mask = [], [], []

        for sol, r in zip(sols, results):
            if not r.failed:
                obj = r.agg_obj
                objs.append(obj)  # Always insert objs.
                meas = r.agg_measures
                meta = self.extract_metadata(r)

                d = r.all_metadata[
                    0]  # all_metadata is a list for multiple repeats
                self.surrogate_data.append({
                    "solution": sol,
                    "goals": d.get("goals", None),
                    "obstacles": d.get("obstacles", None),
                    "human_trajectory": d.get("human_trajectory", None),
                    "robot_trajectory": d.get("robot_trajectory", None),
                    "objective": obj,
                    "measures": meas,
                    "aux_measures": r.extra_measures,
                })

                if self.is_em:
                    self.archive.add_single(sol, r.unreg_obj or obj, meas, meta)
                    self.env_manager.add_experience(sol, r)
                else:
                    if self.train_archive_type is not None:
                        # Add solutions to self.archive to keep the final
                        # archive in sync.
                        self.archive.add_single(sol, r.unreg_obj or obj, meas,
                                                meta)

                        if r.extra_measures is not None:
                            meas = np.concatenate((meas, r.extra_measures))

                    measures.append(meas)
                    metadata.append(meta)
                success_mask.append(True)
            else:
                failed_level_info = self.env_manager.add_failed_info(sol, r)
                self.failed_levels.append(failed_level_info)
                if not self.is_em:
                    # Values do not matter because the success mask is False.
                    objs.append(-1000)
                    measures.append(np.full(self.train_archive.measure_dim,
                                            0.0))
                    metadata.append(None)
                    success_mask.append(False)

        # Tell results to scheduler.
        if not self.is_em:
            logger.info("Filling in null values for skipped sols: %d",
                        skipped_sols)
            for _ in range(skipped_sols):
                objs.append(-1000)
                measures.append(np.full(self.train_archive.measure_dim, 0.0))
                metadata.append(None)
                success_mask.append(False)

            self.scheduler.tell(
                objs,
                measures,
                metadata,
                success_mask=success_mask,
            )

        self.metrics.add(
            "Mean Evaluation",
            np.mean(np.array(objs)[success_mask]),
            logger,
        )
        self.overall_min_obj = min(
            self.overall_min_obj,
            np.min(np.array(objs)[success_mask]),
        )

        if self.train_archive_type is not None and not self.is_em:
            # Only to make the metric logger happy. "Mean Evaluation" doesn't
            # care about the archive.
            self.metrics.add(
                "(Train archive) Mean Evaluation",
                np.mean(np.array(objs)[success_mask]),
                logger,
            )

    def evaluate_initial_emulation_solutions(self):
        logger.info("Evaluating initial solutions")
        initial_solutions = self.env_manager.get_initial_em_sols(
            (self.initial_em_sols, self.sol_size))
        self.evaluate_solutions(initial_solutions)

    def evaluate_emulation_archive(self, emulation_archive: ArchiveBase):
        logger.info("Evaluating solutions in emulation_archive")

        if self.num_elites_to_eval is None:
            sols = [elite.solution for elite in emulation_archive]
            logger.info("%d solutions in emulation_archive", len(sols))
        else:
            num_sols = len(emulation_archive)
            sols = []
            sol_values = []
            rands = self.rng.uniform(0, 1e-8, size=num_sols)  # For tiebreak

            for i, elite in enumerate(emulation_archive):
                sols.append(elite.solution)
                if self.random_sample_em:
                    new_elite = 1
                else:
                    new_elite = int(
                        self.archive.elites_with_measures_single(
                            elite.measures).objective is None)
                sol_values.append(new_elite + rands[i])

            _, sorted_sols = zip(*sorted(
                zip(sol_values, sols), reverse=True, key=lambda x: x[0]))
            sols = sorted_sols[:self.num_elites_to_eval]
            logger.info(
                f"{np.sum(np.array(sol_values) > 1e-6)} solutions predicted to "
                f"improve.")
            logger.info(
                f"Evaluating {len(sols)} out of {num_sols} solutions in "
                f"emulation_archive")

        self.evaluate_solutions(np.array(sols))

    def get_summary(self) -> str:
        """Returns string summary of algorithm performance."""
        qd_score = self.metrics.get_single('Actual QD Score')['y'][-1]
        best = ("N/A" if self.outer_itrs_completed <= 0 else
                f"{self.metrics.get_single('Best Objective')['y'][-1]:.3f}")
        coverage = self.metrics.get_single('Archive Coverage')['y'][-1]
        return f"QD: {qd_score:.3f} | Best: {best} | Cov: {coverage:.3f}"

    def _plot_surrogate_archive_heatmap(self):
        """Run a single inner loop and plot the surrogate archive heatmap."""
        emulation_archive = self.build_emulation_archive()

        downsample_archive: ArchiveBase = self.archive_type(
            solution_dim=emulation_archive.solution_dim,
            dtype=np.float32,
            record_history=False)

        for elite in emulation_archive:
            downsample_archive.add_single(elite.solution, elite.objective,
                                          elite.measures[:2])

        fig, ax = plt.subplots()
        plt.axis("off")
        grid_archive_heatmap(downsample_archive,
                             ax=ax,
                             cmap="viridis",
                             cbar=None)
        fig.savefig("heatmap_figs/surr_archive.png")
        fig.savefig("heatmap_figs/surr_archive.pdf")
        fig.savefig("heatmap_figs/surr_archive.svg")

    def execute(self):
        """Runs the entire algorithm."""
        while not self.finished():
            self.msg_all(
                f"----- Outer Itr {self.outer_itrs_completed + 1} "
                f"({self.total_evals} evals | {self.get_summary()}) -----")
            self.metrics.start_itr()
            self.archive.new_history_gen()
            if self.train_archive_type is not None and not self.is_em:
                self.train_archive.new_history_gen()

            if self.is_em:
                if self.outer_itrs_completed == 0:
                    self.evaluate_initial_emulation_solutions()
                else:
                    logger.info("Running inner loop")
                    self.env_manager.em_train()
                    emulation_archive = self.build_emulation_archive()
                    self.evaluate_emulation_archive(emulation_archive)
            else:
                logger.info("Running classic pyribs")
                sols = self.scheduler.ask()
                self.evaluate_solutions(sols)

            # Restarting may be done for a Dask client so that workers are
            # guaranteed to have a "clean slate," but this is unnecessary for
            # our use case.
            #  if XXXXX:
            #      self.client.restart()

            logger.info("Outer itr complete - now logging and saving data")
            self.outer_itrs_completed += 1
            self.add_performance_metrics(self.archive)
            if self.train_archive_type is not None and not self.is_em:
                self.add_performance_metrics(self.train_archive)
            self.metrics.end_itr()
            self.plot_metrics()
            self.save_data()  # Keep at end of loop (see method docstring).

        self.msg_all(f"----- Done! {self.outer_itrs_completed} itrs, "
                     f"{self.total_evals} evals | {self.get_summary()} -----")
