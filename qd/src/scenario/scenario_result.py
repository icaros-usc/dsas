"""Class representing the results of an evaluation."""
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ScenarioResult:  # pylint: disable = too-many-instance-attributes
    """Represents `n` results from an objective function evaluation.

    `n` is typically the number of evals (n_evals).
    """

    ## Raw data ##

    solution: np.ndarray = None  # (solution_dim,) array
    all_objs: np.ndarray = None  # (n,) array
    all_measures: np.ndarray = None  # (n, measure_dim) array
    all_metadata: List[dict] = None  # (n,) list

    ## Aggregate data ##

    agg_obj: float = None
    agg_measures: np.ndarray = None  # (measure_dim,) array

    ## Measures of spread ##

    std_obj: float = None
    std_measures: np.ndarray = None  # (measure_dim,) array

    ## Other data ##

    failed: bool = False
    error_message: str = None
    reg_cost: float = None  # Regularization cost (like MIP cost) (optional)
    unreg_obj: float = None  # Unregularized objective (optional)
    extra_measures: np.ndarray = None  # (extra_measure_dim,) array (optional)

    @staticmethod
    def from_raw(
        solution: "array-like",
        objs: "array-like",
        measures: "array-like",
        metadata: List[dict],
        opts: dict = None,
    ):
        """Constructs a ScenarioResult from raw data.

        `opts` is a dict with several configuration options. It may be better as
        a gin parameter, but since ScenarioResult is created on workers, gin
        parameters are unavailable (unless we start loading gin on workers too).
        Options in `opts` are:

            `aggregation` (default="mean"): How each piece of data should be
                aggregated into single values. Options are:
                - "mean": Take the mean, e.g. mean measure
                - "median": Take the median, e.g. median measure (element-wise)
        """
        # Handle config options.
        opts = opts or {}
        opts.setdefault("aggregation", "mean")

        assert opts["aggregation"] == "mean", \
            "Only mean aggregation is currently supported."

        objs = np.array(objs)
        measures = np.array(measures)

        return ScenarioResult(
            solution=solution,
            all_objs=objs,
            all_measures=measures,
            all_metadata=metadata,
            agg_obj=np.mean(objs),
            agg_measures=np.mean(measures, axis=0),
            std_obj=np.std(objs),
            std_measures=np.std(measures, axis=0),
            unreg_obj=np.mean(objs),
        )
