"""Provides ScenarioManager."""
import logging
import warnings
from pathlib import Path
from typing import Tuple, Union

import gin
import numpy as np
import torch
from torch.autograd.functional import jacobian

from src.client import Client
from src.device import DEVICE
from src.scenario.emulation_model.buffer import Experience
from src.scenario.emulation_model.emulation_model import SurrogateModel
from src.scenario.scenario_config import ScenarioConfig, ServerScenarioConfig
from src.scenario.scenario_result import ScenarioResult
from src.sol_postprocessors import tanh_two_ws, clip_one_ws

logger = logging.getLogger(__name__)


@gin.configurable(denylist=["client", "rng", "config"])
class ScenarioManager:
    """Manager for the HRI scenarios.

    Args:
        client: Client for distributed compute.
        rng: Random generator. Can be set later. Uses `np.random.default_rng()`
            by default.
        config: ScenarioConfig to config/pass parameters for the scenario.
        initial_sol_sigma: Sigma for generating initial solutions.
        raw_em_input: True if QD solutions should directly be passed to surrogate models
            instead of preprocessing. (default: False)
        inner_loop_clip: True if solution clipping should be done in the inner loop.
            (default False)
    """

    def __init__(
        self,
        client: Client,
        rng: np.random.Generator = None,
        config: ScenarioConfig = None,
        initial_sol_sigma: float = None,
        raw_em_input: bool = False,
        inner_loop_clip: bool = False,
    ):
        self.client = client
        self.rng = rng or np.random.default_rng()

        # If config is not provided, assume everything comes from gin.
        self.config = config or ScenarioConfig(kwargs_to_pass=ServerScenarioConfig())
        self.initial_sol_sigma = initial_sol_sigma

        self.emulation_model = None
        self.raw_em_input = raw_em_input
        self.inner_loop_clip = inner_loop_clip

    def em_init(self, seed: int, pickle_path: Path = None, pytorch_path: Path = None):
        """Initialize the emulation model and optionally load from saved state.

        Args:
            seed: Random seed to use.
            pickle_path: Path to the saved emulation model data (optional).
            pytorch_path: Path to the saved emulation model network (optional).
        """
        self.emulation_model = SurrogateModel(
            collab="collab" in self.config.scenario_function, seed=seed + 420
        )
        if pickle_path is not None:
            self.emulation_model.load(pickle_path, pytorch_path)
        logger.info("Emulation Model: %s", self.emulation_model)

    def get_initial_em_sols(self, size: Tuple):
        """Returns random solutions with the given size.

        Args:
            size: Tuple with (n_solutions, sol_size).

        Returns:
            Randomly generated solutions.
        """
        if "collab" in self.config.scenario_function:
            if self.config.kwargs_to_pass.human_mode in ["mdp", "stochastic"]:
                bounds = [
                    [
                        [0, 0],
                        [0.52, 0.68],
                        [-0.08, 0.18],
                        [0, 0],
                        [0.52, 0.68],
                        [-0.08, 0.18],
                        [0, 0],
                        [0.52, 0.68],
                        [-0.08, 0.18],
                    ],
                    [
                        [0, 0],
                        [0.138, 0.298],
                        [0.496, 0.756],
                        [0, 0],
                        [0.138, 0.298],
                        [0.496, 0.756],
                        [0, 0],
                        [0.138, 0.298],
                        [0.496, 0.756],
                    ],
                ]

                if self.config.kwargs_to_pass.human_mode == "stochastic":
                    bounds[0].extend([[2.5, 3.5], [0.5, 2]])
                    bounds[1].extend([[2.5, 3.5], [0.5, 2]])
            else:
                raise NotImplementedError(
                    f"Human mode {self.config.kwargs_to_pass.human_mode} not implemented."
                )

            bounds = np.array(bounds).transpose((2, 0, 1))
            aug_lower_bounds = bounds[0]
            aug_upper_bounds = bounds[1]
        else:
            bounds = np.array(
                [
                    [0.52, 0.68],
                    [-0.08, 0.18],
                    [0.52, 0.68],
                    [-0.08, 0.18],
                    [-0.05, 0.05],
                    [-0.05, 0.05],
                    [-0.05, 0.05],
                    [-0.05, 0.05],
                    [-0.05, 0.05],
                ]
            ).T
            aug_lower_bounds = bounds[0][None, ...]
            aug_upper_bounds = bounds[1][None, ...]

        all_rands_size = [aug_lower_bounds.shape[0]] + list(size)
        all_rands = self.rng.uniform(
            low=aug_lower_bounds[:, None, :],
            high=aug_upper_bounds[:, None, :],
            size=all_rands_size,
        )

        bound_choices = self.rng.integers(aug_lower_bounds.shape[0], size=size)
        return np.choose(bound_choices, all_rands)

        # if self.initial_sol_sigma is None:
        #     raise ValueError(
        #         "initial_sol_sigma needs to be specified for surrogate model experiments."
        #     )
        #
        # return self.initial_sol_sigma * self.rng.standard_normal(
        #     size=size, dtype=np.float32
        # )

    def em_train(self):
        self.emulation_model.train()

    def emulation_pipeline(self, sols, grad_estimate=False):
        """Pipeline that takes solutions and uses the emulation model to predict
        the objective and measures. If grad_estimate is True, also returns the
        Jacobians.

        Args:
            sols: Emitted solutions.
            grad_estimate: True if gradient should be calculated (for
                DQD algorithms).

        Returns:
            objs: Predicted objective values.
            measures: Predicted measure values.
            jacobians (optional): Jacobian of the objective, measures w.r.t.
                inputs.
            success_mask: Array of size `len(lvls)`. An element in the array is
                False if some part of the prediction pipeline failed for the
                corresponding solution.
        """
        if self.raw_em_input:
            if "collab" in self.config.scenario_function:
                reg, clipped_sols = self._obj_reg(sols)
                solution_batch = clipped_sols if self.inner_loop_clip else sols
                solution_batch[:, 0:9:3] = 0
                model_input = np.array(solution_batch)
            else:
                clipped_sols, info_list = clip_one_ws(sols)
                reg = np.array([info["mip_cost"] for info in info_list])
                solution_batch = clipped_sols if self.inner_loop_clip else sols
                model_input = np.array(solution_batch)
        else:
            if "collab" not in self.config.scenario_function:
                raise NotImplementedError(
                    "Shared autonomy currently only works with raw_em_input"
                )

            if self.config.kwargs_to_pass.human_mode != "mdp":
                raise NotImplementedError(
                    "Human modes other than mdp currently only work with raw_em_input"
                )

            # TODO: Not sure how to make this clean.
            solution_batch = tanh_two_ws(sols)
            goal_idx = np.array([1, 2, 4, 5, 7, 8])
            goal_xys = solution_batch[:, goal_idx]
            obstacle_data = np.repeat([0.409, 0.338, 0.05], len(solution_batch)).reshape(
                len(solution_batch), -1
            )
            model_input = np.concatenate([goal_xys, obstacle_data], axis=-1)
            reg = 0

        success_mask = np.ones(len(sols), dtype=bool)
        objs, measures = self.emulation_model.predict(model_input)

        # Regularize obj
        objs -= reg

        # Aux measures
        if "collab" in self.config.scenario_function:
            scaled_x = solution_batch[:, 1:9:3].copy()
            c1 = scaled_x >= 0.52
            c2 = scaled_x <= 0.298
            scaled_x[c1] -= 0.52
            scaled_x[c2] -= 0.298
            measures = np.concatenate([measures, scaled_x], axis=-1)

        if grad_estimate:
            jacobians = self._jacobian_batch(sols)
            if np.all(jacobians == 0):
                raise ValueError(
                    "All elements of the Jacobian is 0. This is most "
                    "likely due to a non-differentiable operator in the "
                    "pipeline."
                )
            return objs, measures, jacobians, success_mask

        return objs, measures, success_mask

    def eval_pipeline(
        self, solution_batch, eval_kwargs,  # pylint: disable = unused-argument
    ):
        """Pipeline that takes a list of solutions and evaluates it.

        Args:
            solution_batch: Emitted solutions.
            eval_kwargs: Arguments for the evaluation module.

        Returns:
            Results of the evaluation.
        """
        # Currently, the scenario evaluations do not take in a seed, so no need
        # for this part.
        #
        # Make each solution evaluation have a different seed. Note that we
        # assign seeds to solutions rather than workers, which means that we
        # are agnostic to worker configuration.
        #  evaluation_seeds = self.rng.integers(np.iinfo(np.int32).max / 2,
        #                                       size=len(sols),
        #                                       endpoint=True)

        info_list = None
        if self.config.sol_postprocessing_func is not None:
            logger.info("Postprocessing solutions")
            if "collab" in self.config.scenario_function:
                solution_batch[:, 0:9:3] = 0
            solution_batch, info_list = self.config.sol_postprocessing_func(
                solution_batch
            )

        logger.info("Evaluating solutions")
        results = self.client.evaluate(solution_batch, self.config)

        if info_list is not None:
            for i, (r, info) in enumerate(zip(results, info_list)):
                if not r.failed:
                    # This doesn't work for any postprocessor other than the current
                    # MIP or something similar. But I don't think I care.
                    r.reg_cost = info.get("mip_cost")
                    if r.reg_cost is not None:
                        r.agg_obj -= self.config.reg_obj_coeff * r.reg_cost

                    extra_measures = info.get("obj_regions")
                    if extra_measures is not None:
                        r.extra_measures = np.array(extra_measures)

        return results

    def add_experience(self, sol, result):
        """Add required experience to the emulation model based on the solution
        and the results.

        Args:
            sol: Emitted solution (not postprocessed).
            result: Evaluation result.
        """
        # TODO: Maybe make this configurable? This is somewhat related to MDP as well.
        if "collab" in self.config.scenario_function:
            size = (32, 32)
            bounding_box = [[0.1, 2.0], [-0.15, 2.0]]
        else:
            size = (32, 32)
            bounding_box = [(0.4, 0.7), (-0.1, 0.2)]

        d = result.all_metadata[0]  # all_metadata is a list for multiple repeats

        if self.raw_em_input:
            surr_input = sol
        else:
            surr_input = np.concatenate(
                [
                    np.array(d.get("goals")).flatten(),
                    np.array(d.get("obstacles")).flatten(),
                ]
            )

        if "collab" in self.config.scenario_function:
            human_grid = self._compute_occupancy_grid(
                size, bounding_box, d.get("human_trajectory")
            )
            robot_grid = self._compute_occupancy_grid(
                size, bounding_box, d.get("robot_trajectory")
            )
            occ_grids = np.stack([human_grid, robot_grid])
        else:
            occ_grids = self._compute_occupancy_grid(
                size, bounding_box, d.get("robot_trajectory")
            )

        norm_factor = 100 if "collab" in self.config.scenario_function else 10
        output = np.concatenate(
            [
                np.array([result.agg_obj / norm_factor]),
                np.array(result.agg_measures).flatten(),
            ]
        )

        self.emulation_model.add(Experience(surr_input, occ_grids, output))

    def _compute_occupancy_grid(self, size, bounding_box, trajectory):
        grid = np.zeros(size)

        n = len(trajectory)
        for timestamp, pt in trajectory:
            i0 = self._compute_index(size[0], bounding_box[0], pt[0])
            i1 = self._compute_index(size[1], bounding_box[1], pt[1])
            grid[i0, i1] += 1.0
        grid /= n

        return grid

    @staticmethod
    def _compute_index(size, bounds, pos, eps=1e-9):
        interval_size = bounds[1] - bounds[0]
        index = (size * (pos - bounds[0]) + eps) / interval_size
        return np.clip(index, 0, size - 1).astype(np.int32)

    @staticmethod
    def add_failed_info(solution, result: ScenarioResult) -> dict:
        """Returns a dict containing relevant information about failed
        solutions.

        Args:
            solution: Emitted solution.
            result: Evaluation result.

        Returns:
            Dict with failure information.
        """
        return {
            "solution": solution,
            "log_message": result.error_message,
        }

    MIN_SCORE = 0.0

    def actual_qd_score(self, objs: "array-like"):
        """Calculates QD score of the given objectives.

        Scores are normalized to be non-negative by subtracting a constant min
        score.

        Args:
            objs: List of objective values.
        """
        objs = np.array(objs)
        objs -= self.MIN_SCORE
        if np.any(objs < 0):
            warnings.warn("Some objective values are still negative.")
        return np.sum(objs)

    def _jacobian_batch(self, batch_solutions: np.ndarray) -> np.ndarray:
        """
        Calculates Jacobian for a batch of solutions. Each Jacobian matrix
        consists of the gradient for the objective followed by the gradients for
        the BCs.

        Args:
            batch_solutions: Batch of solutions given by the optimizer.

        Returns:
            The jacobian matrix of shape (batch size, measure_dim + 1,
                solution_dim).
        """
        # The below trick computes the sum of outputs along the batch and then asks for
        # the Jacobian. Assuming cross-batch terms are zero, the Jacobian function returns
        # a transposed version required batch Jacobian. See
        # https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571/7.
        def f_sum(x):
            return torch.sum(self._em_grad_pipeline(x), dim=0)

        jac = jacobian(
            f_sum,
            torch.as_tensor(batch_solutions, dtype=torch.float, device=DEVICE),
            vectorize=True,
        ).permute((1, 0, 2))

        jac_cpu = jac.cpu().detach().numpy()

        if np.all(jac_cpu == 0):
            raise ValueError(
                "All elements of the Jacobian are 0. This is most likely due to a "
                "non-differentiable operator in the pipeline."
            )

        return jac_cpu

    def _em_grad_pipeline(self, sols: torch.Tensor) -> torch.Tensor:
        """The pipeline from QD solution to objective and measures through the surrogate
        model. Similar to emulation_pipeline, but goes from tensor->tensor through mostly
        differentiable functions instead of array->array.

        Args:
            sols: Batch of QD solutions.

        Returns:
            objective, measures in the form of a (batch_size, 6) tensor.
        """
        if self.raw_em_input:
            if "collab" in self.config.scenario_function:
                reg, clipped_sols = self._obj_reg(sols)
                solution_batch = clipped_sols if self.inner_loop_clip else sols * 1
                solution_batch[:, 0:9:3] = 0
                model_input = solution_batch
            else:
                clipped_sols, reg = clip_one_ws(sols)
                solution_batch = clipped_sols if self.inner_loop_clip else sols * 1
                model_input = solution_batch
        else:
            if "collab" not in self.config.scenario_function:
                raise NotImplementedError(
                    "Shared autonomy currently only works with raw_em_input"
                )

            if self.config.kwargs_to_pass.human_mode != "mdp":
                raise NotImplementedError(
                    "Human modes other than mdp currently only work with raw_em_input"
                )

            solution_batch = tanh_two_ws(sols)
            goal_idx = torch.Tensor([1, 2, 4, 5, 7, 8]).to(DEVICE).long()
            goal_xys = solution_batch[:, goal_idx]
            obstacle_data = (
                torch.Tensor([0.409, 0.338, 0.05])
                .to(DEVICE)
                .repeat(len(solution_batch), 1)
            )
            model_input = torch.cat([goal_xys, obstacle_data], dim=-1)
            reg = 0

        outputs = self.emulation_model.eval_predict_with_grad(model_input)
        outputs[:, 0] -= reg

        # Include aux measures
        if "collab" in self.config.scenario_function:
            x_idx = torch.Tensor([1, 4, 7]).to(DEVICE).long()
            scaled_x = torch.index_select(solution_batch, 1, x_idx)
            c1 = scaled_x >= 0.52
            c2 = scaled_x <= 0.298
            scaled_x[c1] -= 0.52
            scaled_x[c2] -= 0.298
            outputs = torch.cat([outputs, scaled_x], dim=-1)

        return outputs

    def _obj_reg(self, sols: Union[np.ndarray, torch.Tensor]):
        """Calculates the quadratic regularization for a solution. Checks the closest
        workspace and adds distance to it as the regularization. Regularization is zero if
        the solution is already inside a workspace.

        Args:
            sols: Solutions in array or tensor form.

        Returns:
            Regularization penalty for each solution.
        """
        # FIXME: I don't know where to put these things
        goal_x_bounds = [[0.52, 0.68], [0.138, 0.298]]
        goal_y_bounds = [[-0.08, 0.18], [0.496, 0.756]]
        st_bounds = [2.5, 3.5]  # softmax_temperature between 2.5 and 3.5 (will be 10 ^ x)
        vel_bounds = [0.5, 2]  # human_vel_coeff between 0.5 and 2

        obj_x = sols[:, 1:9:3]
        obj_y = sols[:, 2:9:3]

        if isinstance(sols, np.ndarray):
            goal_x_bounds = np.array(goal_x_bounds)
            # dx is an array of shape (batch_size, num_objects, num_workspace). Each
            # element in the array denotes the x-distance from the object to the closest
            # point in the workspace. In np.maximum, first term is positive if the object
            # is to the left of the workspace, second term is positive if it is to the
            # right, both are negative if it is within the x bounds of the workspace, so
            # the stored value after clip gives the x-distance to the closest boundary
            # line of the workspace if it is out of bounds and 0 otherwise. dy is
            # calculated similarly and the distance is dx ** 2 + dy ** 2. The
            # regularization term chooses the distance to the closest workspace for each
            # goal object and sums them up for all objects in a scenario, giving a
            # (batch_size,) vector.
            dx = np.clip(
                np.maximum(
                    goal_x_bounds[:, 0] - obj_x[:, :, None],
                    obj_x[:, :, None] - goal_x_bounds[:, 1],
                ),
                a_min=0,
                a_max=None,
            )
            goal_y_bounds = np.array(goal_y_bounds)
            dy = np.clip(
                np.maximum(
                    goal_y_bounds[:, 0] - obj_y[:, :, None],
                    obj_y[:, :, None] - goal_y_bounds[:, 1],
                ),
                a_min=0,
                a_max=None,
            )
            min_dist = np.min(dx ** 2 + dy ** 2, axis=-1)
            min_idx = np.argmin(dx ** 2 + dy ** 2, axis=-1)
            reg = np.sum(min_dist, axis=-1)

            if self.config.kwargs_to_pass.human_mode == "stochastic":
                st_bounds = np.array(st_bounds)
                vel_bounds = np.array(vel_bounds)

                dst = np.clip(
                    np.maximum(st_bounds[0] - sols[:, 9], sols[:, 9] - st_bounds[1]),
                    a_min=0,
                    a_max=None,
                )

                dv = np.clip(
                    np.maximum(vel_bounds[0] - sols[:, 10], sols[:, 10] - vel_bounds[1]),
                    a_min=0,
                    a_max=None,
                )

                reg += dst ** 2 + dv ** 2

            clipped_sols = sols.copy()

            clip_x1 = np.clip(
                sols[:, 1:9:3], a_min=goal_x_bounds[0, 0], a_max=goal_x_bounds[0, 1]
            )
            clip_x2 = np.clip(
                sols[:, 1:9:3], a_min=goal_x_bounds[1, 0], a_max=goal_x_bounds[1, 1]
            )
            clipped_sols[:, 1:9:3] = np.where(min_idx.astype(bool), clip_x2, clip_x1)

            clip_y1 = np.clip(
                sols[:, 2:9:3], a_min=goal_y_bounds[0, 0], a_max=goal_y_bounds[0, 1]
            )
            clip_y2 = np.clip(
                sols[:, 2:9:3], a_min=goal_y_bounds[1, 0], a_max=goal_y_bounds[1, 1]
            )
            clipped_sols[:, 2:9:3] = np.where(min_idx.astype(bool), clip_y2, clip_y1)

            if self.config.kwargs_to_pass.human_mode == "stochastic":
                clipped_sols[:, 9] = np.clip(
                    sols[:, 9], a_min=st_bounds[0], a_max=st_bounds[1]
                )
                clipped_sols[:, 10] = np.clip(
                    sols[:, 10], a_min=vel_bounds[0], a_max=vel_bounds[1]
                )
        else:
            goal_x_bounds = torch.Tensor(goal_x_bounds).to(DEVICE)
            dx = torch.clip(
                torch.maximum(
                    goal_x_bounds[:, 0] - obj_x[:, :, None],
                    obj_x[:, :, None] - goal_x_bounds[:, 1],
                ),
                min=0,
                max=None,
            )
            goal_y_bounds = torch.Tensor(goal_y_bounds).to(DEVICE)
            dy = torch.clip(
                torch.maximum(
                    goal_y_bounds[:, 0] - obj_y[:, :, None],
                    obj_y[:, :, None] - goal_y_bounds[:, 1],
                ),
                min=0,
                max=None,
            )
            min_dist, min_idx = torch.min(dx ** 2 + dy ** 2, dim=-1)
            reg = torch.sum(min_dist, dim=-1)

            if self.config.kwargs_to_pass.human_mode == "stochastic":
                st_bounds = torch.Tensor(st_bounds).to(DEVICE)
                vel_bounds = torch.Tensor(vel_bounds).to(DEVICE)

                dst = torch.clip(
                    torch.maximum(st_bounds[0] - sols[:, 9], sols[:, 9] - st_bounds[1]),
                    min=0,
                    max=None,
                )

                dv = torch.clip(
                    torch.maximum(
                        vel_bounds[0] - sols[:, 10], sols[:, 10] - vel_bounds[1]
                    ),
                    min=0,
                    max=None,
                )

                reg += dst ** 2 + dv ** 2

            clipped_sols = sols * 1

            clip_x1 = torch.clip(
                sols[:, 1:9:3], min=goal_x_bounds[0, 0], max=goal_x_bounds[0, 1]
            )
            clip_x2 = torch.clip(
                sols[:, 1:9:3], min=goal_x_bounds[1, 0], max=goal_x_bounds[1, 1]
            )
            clipped_sols[:, 1:9:3] = torch.where(min_idx.bool(), clip_x2, clip_x1)

            clip_y1 = torch.clip(
                sols[:, 2:9:3], min=goal_y_bounds[0, 0], max=goal_y_bounds[0, 1]
            )
            clip_y2 = torch.clip(
                sols[:, 2:9:3], min=goal_y_bounds[1, 0], max=goal_y_bounds[1, 1]
            )
            clipped_sols[:, 2:9:3] = torch.where(min_idx.bool(), clip_y2, clip_y1)

            if self.config.kwargs_to_pass.human_mode == "stochastic":
                clipped_sols[:, 9] = torch.clip(
                    sols[:, 9], min=st_bounds[0], max=st_bounds[1]
                )
                clipped_sols[:, 10] = torch.clip(
                    sols[:, 10], min=vel_bounds[0], max=vel_bounds[1]
                )

        return reg, clipped_sols
