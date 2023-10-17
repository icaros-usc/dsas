"""Provides ScenarioConfig."""
from dataclasses import dataclass
from typing import List

import gin


@gin.configurable
@dataclass
class ServerScenarioConfig:
    """Server side config for scenarios"""

    human_mode: str = "mdp"
    num_goals: int = 3
    one_shot: bool = False
    softmax: bool = True
    discount_factor: float = 0.9999
    reward_goal: float = 1
    reward_move: float = -0.01
    reward_stay: float = -0.01
    vi_max_iters: float = 5000
    softmax_temperature: float = 1000


@gin.configurable
@dataclass
class ScenarioConfig:
    """Config for scenarios."""

    # Measures.
    measure_names: List[str] = gin.REQUIRED

    # Results.
    aggregation_type: str = "mean"

    # Number of times to evaluate each solution during real evaluation.
    n_evals: int = gin.REQUIRED

    # Name of the scenario function.
    scenario_function: str = gin.REQUIRED

    #  Function that takes a solution batch and applies any postprocessing like re-mapping
    #  the solutions or passing it through a GAN. Does no postprocessing by default.
    sol_postprocessing_func: callable = None

    # Coefficient for subtracting MIP cost from the objective during training. Default is
    # 0.0 to prevent any regularization.
    reg_obj_coeff: float = 0.0

    # Kwargs that will be passed to the server
    kwargs_to_pass: ServerScenarioConfig = None
