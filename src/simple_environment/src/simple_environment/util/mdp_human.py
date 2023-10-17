"""Provides MDPHuman.

Run this file directly to test the class.

Keep in mind this file is intended to run with Python 2.
"""
import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.spatial import distance

try:
    from scipy.special import logsumexp
except ImportError:  # Old version of scipy
    from scipy.misc import logsumexp


class Cell:
    """Represents a cell in the MDP grid.

    This class is similar to an enum, but it is not meant to be instantiated.
    """

    EMPTY = 0
    OBSTACLE = 1
    GOAL = 2
    HUMAN = 3

    CELL_TO_STR = ["_", "X", "G", "H"]

    @staticmethod
    def to_str(cell):
        return Cell.CELL_TO_STR[cell]


class MDPHuman:
    """Represents the human as an agent which maximizes a (soft) MDP with
    multiple goals.

    Notable features:
    - The environment is currently assumed to be deterministic.
    - It is okay to move into obstacles, but this will result in a large
      negative reward.
    - It is okay to try to move outside the grid, but doing so will result in
      staying in the same place. Thus, all actions are valid for all cells,
      except for the goal.

    Note: To ensure coordinates end up in the correct cells, specify locations
    with the _center_ of the cells rather than the edges of the cells.

    Args:
        grid_bounds (np.ndarray): `[[x_min, x_max], [y_min, y_max]]` for the
            whole grid (should contain the whole region where human can go).
        cell_size (np.ndarray): Size of a cell in x and y axis.
        start_location (np.ndarray): (x, y) of the human start location
        goal_locations (np.ndarray): (x, y) for each goal location. Goals are
            assumed to be one cell large.
        obstacles (np.ndarray): (x, y, r) for each obstacle. Obstacles are
            assumed to extend r cells on each side ("square" of size 2r + 1).
        inaccessible_func (Optional[callable]): Function that takes in (x, y)
            and returns True if that point is inaccessible.
        epsilon: Adjusts for floating point precision errors when computing grid
            coordinates as is done in the pyribs GridArchive.
        softmax: Whether human should act according to soft MDP policy. Values
            returned by get_values and get_q_values are always soft MDP values.
        stochastic: True if actions should be stochastic (only applicable when
            softmax=True, default False).
        discount_factor: Discount factor of the MDP.
        reward_goal: Reward for reaching a goal.
        reward_move: Base reward for moving one cell (movements may be to any of
            the eight neighboring cells).
        reward_scale_diagonal: If True, the cost for moving diagonally will be
            scaled by sqrt(2).
        reward_stay: Reward for taking an action that results in staying in the
            same cell. If the cell is an obstacle, reward_obstacle is applied
            instead.
        reward_obstacle: Reward for moving into an obstacle.
        vi_max_iters: Maximum number of iterations to run value iteration.
        vi_mean_delta: Value iteration terminates when the average change in the
            value function's output is less than this amount.
        softmax_temperature: Q-values are scaled by this when doing logmeanexp
            to make it closer to max (should be >= 0).
    """

    def __init__(
        self,
        grid_bounds,
        cell_size,
        start_location,
        goal_locations,
        obstacles,
        inaccessible_func=None,
        epsilon=1e-6,
        softmax=True,
        stochastic=False,
        discount_factor=0.99,
        reward_goal=1.0,
        reward_move=-0.01,
        reward_scale_diagonal=True,
        reward_stay=-0.01,
        reward_obstacle=-1.0,
        vi_max_iters=500,
        vi_mean_delta=1e-3,
        softmax_temperature=1,
    ):
        self.epsilon = epsilon
        self.softmax = softmax
        self.stochastic = stochastic

        if self.stochastic:
            assert self.softmax, "softmax should be True if stochastic is True"
        if self.softmax:
            assert softmax_temperature >= 0, "softmax_temperature should be >= 0"

        # Process grid info.

        self.grid_bounds = np.asarray(grid_bounds, dtype=np.float64)
        assert self.grid_bounds.shape == (2, 2), "grid_bounds must be shape (2,2)"
        self.range_size = self.grid_bounds[:, 1] - self.grid_bounds[:, 0]

        self.cell_size = np.asarray(cell_size, dtype=np.float64)
        assert self.cell_size.shape == (2,), "cell_size must be shape (2,)"

        self.grid_dims = np.floor(
            (self.grid_bounds[:, 1] - self.grid_bounds[:, 0]) / self.cell_size
        ).astype(np.int32)

        self.start_location = np.asarray(start_location, dtype=np.float64)
        assert self.start_location.shape == (2,), "start_location must be shape (2,)"
        self.start_coords = self.real_to_grid(self.start_location)

        obstacles = np.asarray(obstacles, dtype=np.float64)
        assert (
            obstacles.ndim == 2 and obstacles.shape[1] == 3
        ), "obstacles must be an array of shape (n_obstacles, 3)"
        self.obstacles = obstacles

        goal_locations = np.asarray(goal_locations, dtype=np.float64)
        assert (
            goal_locations.ndim == 2 and goal_locations.shape[1] == 2
        ), "goal_locations must be an array of shape (n_goal_locations, 2)"
        self.goal_locations = goal_locations
        self.goal_coords = self.real_to_grid(self.goal_locations)
        self.goal_coords_to_locations = dict(
            zip(map(tuple, self.goal_coords), self.goal_locations)
        )
        self.n_goals = len(self.goal_locations)

        # There is a separate grid world for each goal because when addressing
        # each goal, the other goals should be obstacles. Thus, there is a batch
        # dimension identifying the goal grid followed by the x-dimension and
        # y-dimension.
        self.grid = np.full(
            (self.n_goals, self.grid_dims[0], self.grid_dims[1],), Cell.EMPTY, dtype=int,
        )

        # Dynamics and rewards.
        self.discount_factor = discount_factor
        self.reward_goal = reward_goal
        self.reward_move = reward_move
        self.reward_scale_diagonal = reward_scale_diagonal
        self.reward_stay = reward_stay
        self.reward_obstacle = reward_obstacle
        self.softmax_temperature = softmax_temperature

        # Changes in x and y for each of the 8 actions (which move to
        # neighboring cells).
        self.action_deltas = [
            [-1, -1, True],  # dx, dy, is_diagonal
            [-1, 0, False],
            [-1, 1, True],
            [0, 1, False],
            [1, 1, True],
            [1, 0, False],
            [1, -1, True],
            [0, -1, False],
        ]
        self.n_actions = len(self.action_deltas)

        # For each cell, indicates the state where the action would lead - fill
        # with int min to indicate invalid state.
        self.next_state = np.full(
            (self.n_goals, self.grid_dims[0], self.grid_dims[1], self.n_actions, 2,),
            # Must have a valid default value because we need to index with the
            # default value during value iteration.
            0,
            dtype=np.int32,
        )

        # The reward for taking the given action in each cell - fill with nan to
        # indicate invalid reward.
        self.reward = np.full(
            (self.n_goals, self.grid_dims[0], self.grid_dims[1], self.n_actions,),
            np.nan,
            dtype=np.float64,
        )

        # Value iteration.
        self.vi_max_iters = vi_max_iters
        self.vi_mean_delta = vi_mean_delta

        # Value function and Q function.
        self.vf = np.full(
            (self.n_goals, self.grid_dims[0], self.grid_dims[1],),
            np.nan,
            dtype=np.float64,
        )
        self.vf_softmax = self.vf.copy()
        self.qf = np.full(
            (self.n_goals, self.grid_dims[0], self.grid_dims[1], self.n_actions,),
            np.nan,
            dtype=np.float64,
        )
        self.qf_softmax = self.qf.copy()

        # The best action for each state with each goal.
        self.best_action = np.full(
            (self.n_goals, self.grid_dims[0], self.grid_dims[1],),
            np.iinfo(np.int32).min,
            dtype=np.int32,
        )
        self.best_action_softmax = self.best_action.copy()

        self.construct_grid(inaccessible_func)
        self.compute_mdp_dynamics_and_rewards()
        self.value_iteration(self.vf, self.qf, self.best_action, softmax=False)
        self.value_iteration(
            self.vf_softmax, self.qf_softmax, self.best_action_softmax, softmax=True
        )

    def construct_grid(self, inaccessible_func):
        """Constructs the grid world for the MDP.

        To be used during construction.
        """
        base_grid = np.full(self.grid_dims, Cell.EMPTY, dtype=int)
        base_grid[tuple(self.start_coords)] = Cell.HUMAN
        for gr_x in range(base_grid.shape[0]):
            for gr_y in range(base_grid.shape[1]):
                cell_center = self.center_of_cell(np.array([gr_x, gr_y]))
                for ob_x, ob_y, ob_r in self.obstacles:
                    if distance.euclidean(cell_center, (ob_x, ob_y)) < ob_r:
                        base_grid[gr_x, gr_y] = Cell.OBSTACLE

                if inaccessible_func and inaccessible_func(cell_center):
                    base_grid[gr_x, gr_y] = Cell.OBSTACLE

        self.grid[:] = base_grid

        for cur_goal in range(self.n_goals):
            # In each grid, goals other than the current one should be
            # obstacles.
            for goal_i, (goal_x, goal_y) in enumerate(self.goal_coords):
                if goal_i == cur_goal:
                    self.grid[cur_goal, goal_x, goal_y] = Cell.GOAL
                else:
                    self.grid[cur_goal, goal_x, goal_y] = Cell.OBSTACLE

    def compute_mdp_dynamics_and_rewards(self):
        """Computes the MDP dynamics and rewards.

        To be used during construction.
        """
        for g, x, y in itertools.product(
            # Equivalent to nested for loop over g, x, y.
            range(self.n_goals),
            range(self.grid_dims[0]),
            range(self.grid_dims[1]),
        ):
            # Goal cell is terminal, so all actions stay in the same place.
            if self.grid[g, x, y] == Cell.GOAL:
                self.next_state[g, x, y, :] = [x, y]
                self.reward[g, x, y, :] = 0.0
                continue

            # Analyze actions for other cells.
            for i, (dx, dy, is_diagonal) in enumerate(self.action_deltas):
                x2, y2 = x + dx, y + dy
                # Check if the action is invalid.
                invalid_cond = (
                    x2 < 0 or x2 >= self.grid_dims[0] or y2 < 0 or y2 >= self.grid_dims[1]
                )

                # This would prevent moving diagonally when a neighboring cell
                # is an obstacle, e.g.
                #
                #   _H_
                #   HXX
                #   _XX
                #
                #  if is_diagonal:
                #      xd1, yd1 = x, y + dy
                #      xd2, yd2 = x + dx, y
                #      invalid_cond = (invalid_cond or
                #                  self.grid[g, xd1, yd1] == Cell.OBSTACLE or
                #                  self.grid[g, xd2, yd2] == Cell.OBSTACLE)

                if invalid_cond:
                    # Stay in the same place for an invalid action.
                    self.next_state[g, x, y, i] = [x, y]
                    if self.grid[g, x, y] == Cell.OBSTACLE:
                        self.reward[g, x, y, i] = self.reward_obstacle
                    else:
                        self.reward[g, x, y, i] = self.reward_stay
                    continue

                # Otherwise, mark the next state and reward.
                self.next_state[g, x, y, i] = [x2, y2]

                # Note that the goal reward is captured in the value iteration.
                if self.grid[g, x2, y2] == Cell.OBSTACLE:
                    # Moving into an obstacle incurs a negative reward.
                    self.reward[g, x, y, i] = self.reward_obstacle
                elif is_diagonal:
                    # Moving diagonally incurs a certain reward.
                    if self.reward_scale_diagonal:
                        self.reward[g, x, y, i] = self.reward_move * np.sqrt(2)
                    else:
                        self.reward[g, x, y, i] = self.reward_move
                else:
                    # Moving in cardinal direction incurs a certain reward.
                    self.reward[g, x, y, i] = self.reward_move

    def value_iteration(self, vf, qf, best_action, softmax=True):
        """Runs value iteration.

        Args:
            vf (np.ndarray): Value function array to update in-place.
            qf (np.ndarray): Q-value function array to update in-place.
            best_action (np.ndarray): Array of best actions to update in-place.
            softmax (bool): True if softmax VI should be used.
        """
        vf[:] = 0.0
        qf[:] = np.nan

        # Loop will be broken if the value function converges.
        for itr in range(1, self.vi_max_iters + 1):
            vf2 = np.zeros_like(vf)

            # Goal cells have no action out of them.
            is_goal_cell = self.grid == Cell.GOAL
            vf2[is_goal_cell] = self.reward_goal
            qf[is_goal_cell] = np.nan

            # Mask for computing following values.
            valid_cells = self.grid != Cell.GOAL

            # Collect the values of the next states.
            next_s = self.next_state[valid_cells]
            # Figure out the goals which correspond to the valid cells, and
            # stack them so that we can index self.vf to obtain the values.
            goals = np.where(valid_cells)[0][:, None]
            # next_s_vf as well as each index array has shape (n_valid_cells,
            # n_actions) (except for goals, which has shape (n_valid_cells, 1)
            # and gets broadcasted.
            #
            # The following used to be the case when we considered invalid
            # actions:
            #
            #   Note that we are indeed retrieving the values of cells for
            #   actions which are invalid, but we cannot avoid this because that
            #   would result in a ragged array. However, the default value of 0
            #   for next_state makes sure we do not cause index errors, and the
            #   nan rewards below ensure these states are ultimately ignored.
            next_s_vf = vf[goals, next_s[:, :, 0], next_s[:, :, 1]]
            qf[valid_cells] = self.reward[valid_cells] + self.discount_factor * next_s_vf

            # Okay to use argmax since all Q-values are real numbers because all
            # actions are valid.
            best_action[valid_cells] = np.argmax(qf[valid_cells], axis=-1)

            if softmax:
                # We are taking the log mean exp here, and that is equivalent to
                # subtracting log(n_actions).
                # log mean exp is a lower bound on max, which makes more sense
                # since the values would explode if V(s) > Q(s, .)
                vf2[valid_cells] = (
                    logsumexp(qf[valid_cells] * self.softmax_temperature, axis=1)
                    - np.log(self.n_actions)
                ) / self.softmax_temperature
            else:
                # qf should not have any more nan's since all actions are valid.
                vf2[valid_cells] = np.max(qf[valid_cells], axis=1)

            # Compute the mean change in the value function.
            mean_delta = np.mean(np.abs(vf2[valid_cells] - vf[valid_cells]))

            # Update value function.
            vf[:] = vf2[:]

            if mean_delta < self.vi_mean_delta:
                # print("VI converged after {} iterations (mean_delta={})".format(
                #     itr, mean_delta))
                break

        if mean_delta >= self.vi_mean_delta:
            raise RuntimeError(
                "VI did not converge after {} iterations (mean_delta={})".format(
                    itr, mean_delta
                )
            )

    ## Utilities ##

    def real_to_grid(self, location):
        """Converts the 2D coordinates into grid coordinates.

        This method will work with both a single coordinate array like [x,y] and
        a batch like [[x1,y1], [x2,y2]], but keep in mind that it returns an
        array either way, and you may need to turn this array into a tuple
        before indexing the grid with it.
        """
        return (
            (self.grid_dims * (location - self.grid_bounds[:, 0]) + self.epsilon)
            / self.range_size
        ).astype(np.int32)

    def center_of_cell(self, coords):
        """Returns the 2D coordinates of the center of the given grid cell.

        coords should be an array of integer indices in the grid.
        """
        return (coords + 0.5) * self.cell_size + self.grid_bounds[:, 0]

    def visualize_grid(self, human_path=False):
        """Returns a string representation of the grid.

        Note that the grid should have a grid for each goal.

        Pass `human_path` to show the path the human would take from start to
        goal in each grid.
        """
        lines = []

        # Copy grid since we modify it with the human path.
        for i, g in enumerate(np.copy(self.grid)):
            if i != 0:
                # Separate the goal grids by a line.
                lines.append("")

            if human_path:
                cur_pt = self.start_location
                cur_coords = self.start_coords
                while g[tuple(cur_coords)] != Cell.GOAL:
                    g[tuple(cur_coords)] = Cell.HUMAN
                    cur_pt = self.get_next_location(self.center_of_cell(cur_coords), i)
                    cur_coords = self.real_to_grid(cur_pt)

            # Transpose since x is usually the first dimension. Then flip so
            # that lower y values get shown lower on the screen.
            g = np.flip(g.T, axis=0)

            lines.append("Goal {}".format(i))
            lines.extend("".join(Cell.to_str(cell) for cell in row) for row in g)

        return "\n".join(lines)

    def visualize_vf(self, width=6, precision=3):
        """Returns a string representation of the value function.

        Args:
            width: Width of floating point values.
            precision: Precision of floating point values.
        """
        lines = []

        for i, (g, vf) in enumerate(zip(self.grid, self.vf)):
            if i != 0:
                # Separate the goal grids by a line.
                lines.append("")

            # Transpose since x is usually the first dimension. Then flip so
            # that lower y values get shown lower on the screen.
            g = np.flip(g.T, axis=0)
            vf = np.flip(vf.T, axis=0)

            lines.append("Goal {}".format(i))
            for g_row, vf_row in zip(g, vf):

                cell_strs = [
                    " {} {:{}.{}f} ".format(Cell.to_str(cell), v, width, precision)
                    for cell, v in zip(g_row, vf_row)
                ]
                divider = "+" + "+".join("-" * len(s) for s in cell_strs) + "+"

                lines.append(divider)
                lines.append("|" + "|".join(s for s in cell_strs) + "|")
            lines.append(divider)

        return "\n".join(lines)

    def visualize_grid_img(
        self,
        outputs=("grid.svg", "grid.pdf"),
        human_path=False,
        show_vf=False,
        show_qf=False,
        width=6,
        precision=3,
    ):
        """Saves a Matplotlib image of the grid to the given file.

        Note that the grid should have a grid for each goal.

        Args:
            output: List of files to save the image to.
            human_path: show the path the human would take from start to goal in
                each grid.
            show_vf: Show the value function.
            show_qf: Show the q function.
            width: Width of floating point values.
            precision: Precision of floating point values.
        """
        fig, axs = plt.subplots(
            nrows=1, ncols=self.n_goals, figsize=(40 * self.n_goals, 40),
        )

        def fmt(f):
            """Formats a floating point number."""
            if np.isnan(f):
                return "N/A"
            else:
                return "{:{}.{}f}".format(f, width, precision)

        for g, (ax, grid) in enumerate(zip(axs, np.copy(self.grid))):
            if human_path:
                cur_pt = self.start_location
                cur_coords = self.start_coords
                while grid[tuple(cur_coords)] != Cell.GOAL:
                    grid[tuple(cur_coords)] = Cell.HUMAN
                    cur_pt = self.get_next_location(self.center_of_cell(cur_coords), g)
                    cur_coords = self.real_to_grid(cur_pt)

            ax.set_title("Goal {}".format(g), fontsize=20, pad=20)
            ax.set_xlim(0, self.grid_dims[0])
            ax.set_ylim(0, self.grid_dims[1])

            for x, y in itertools.product(
                range(self.grid_dims[0]), range(self.grid_dims[1])
            ):
                if grid[x, y] == Cell.GOAL:
                    # Add 0.5 to get to center of cell.
                    ax.plot(
                        [x + 0.5],
                        [y + 0.5],
                        marker="*",
                        markersize=40,
                        color="green",
                        alpha=0.5,
                    )

                if grid[x, y] == Cell.HUMAN:
                    ax.plot(
                        [x + 0.5],
                        [y + 0.5],
                        marker="o",
                        markersize=40,
                        color="blue",
                        alpha=0.5,
                    )

                if grid[x, y] == Cell.OBSTACLE:
                    ax.add_patch(
                        Rectangle((x, y), width=1, height=1, color="red", alpha=0.5)
                    )

                if show_vf:
                    ax.text(
                        x + 0.5,
                        y + 0.5,
                        fmt(self.vf_softmax[g, x, y]),
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontweight="bold",
                        fontsize=5,
                    )

                if show_qf:
                    for i, (dx, dy, _) in enumerate(self.action_deltas):
                        ax.text(
                            x + 0.5 + 0.3 * dx,
                            y + 0.5 + 0.3 * dy,
                            fmt(self.qf_softmax[g, x, y, i]),
                            horizontalalignment="center",
                            verticalalignment="center",
                            fontsize=4,
                        )

            # Create grid by plotting lines -- put it last so it shows up on
            # top.
            ax.vlines(
                np.arange(1, self.grid_dims[0]), 0, self.grid_dims[1], linewidth=1,
            )
            ax.hlines(
                np.arange(1, self.grid_dims[1]), 0, self.grid_dims[0], linewidth=1,
            )

        for output in outputs:
            fig.savefig(output, dpi=300)

    ## Methods called by collaboration code. ##

    def get_next_location(self, cur_location, cur_goal):
        """Take action according to the human policy from given current location
        and return the (x, y) of the center of the next cell. If the next cell
        is the goal cell, return the actual (x, y) of the goal location. This
        function is only called when simulating human.

        Returns the coordinates of the goal if the human is already at cur_goal.

        Args:
            cur_location (np.ndarray): (x, y) of the current human location
            cur_goal (int): Index of the current human goal
        """
        cur_coords = self.real_to_grid(cur_location)

        # Check if already in goal cell.
        cur_goal_coords = self.goal_coords[cur_goal]
        if np.all(cur_coords == cur_goal_coords):
            return self.goal_locations[cur_goal]

        # Otherwise, figure out the action to take.
        if self.softmax:
            if self.stochastic:
                vf = self.vf_softmax[cur_goal, cur_coords[0], cur_coords[1]]
                qfs = self.qf_softmax[cur_goal, cur_coords[0], cur_coords[1]]
                action_probs = np.exp((qfs - vf) * self.softmax_temperature)
                action_probs /= action_probs.sum()
                # TODO: Pass the random seed from QD client and use that here
                action_idx = np.random.choice(len(action_probs), p=action_probs)
            else:
                action_idx = self.best_action_softmax[
                    cur_goal, cur_coords[0], cur_coords[1]
                ]
        else:
            action_idx = self.best_action[cur_goal, cur_coords[0], cur_coords[1]]
        next_coords = self.action_deltas[action_idx][:2] + cur_coords

        if self.grid[cur_goal, next_coords[0], next_coords[1]] == Cell.GOAL:
            # Return the goal coordinates if the next step enters a goal cell.
            return self.goal_coords_to_locations[tuple(next_coords)]
        else:
            return self.center_of_cell(next_coords)

    def get_q_values(self, cur_location, cur_goal):
        """Q-values corresponding to each goal for the current location and
        action taken (or would have been taken) by the human policy. If the
        human is simulated, the action should match the one taken in
        `get_next_location()`.

        If we are already in the goal cell, the Q-values will all be NaN.

        Args:
            cur_location (np.ndarray): (x, y) of the current human location
            cur_goal (int): Index of the current human goal
        """
        cur_coords = self.real_to_grid(cur_location)

        # Check if already in goal cell.
        cur_goal_coords = self.goal_coords[cur_goal]
        if np.all(cur_coords == cur_goal_coords):
            return np.full(self.n_goals, np.nan)

        action_idx = self.best_action[cur_goal, cur_coords[0], cur_coords[1]]
        return self.qf_softmax[:, cur_coords[0], cur_coords[1], action_idx]

    def get_values(self, cur_location):
        """Values corresponding to each goal for the current location.

        Args:
            cur_location (np.ndarray): (x, y) of the current human location
        """
        cur_coords = self.real_to_grid(cur_location)
        return self.vf_softmax[:, cur_coords[0], cur_coords[1]]


if __name__ == "__main__":
    start_time = time.time()
    # Small case.
    #  mdp_human = MDPHuman(
    #      grid_bounds=np.array([[-1, 1], [-1, 1]]),
    #      cell_size=np.array([0.20, 0.20]),
    #      start_location=np.array([-0.75, -0.75]),
    #      goal_locations=np.array([[0.97, 0.96], [0.76, -0.25]]),
    #      obstacles=np.array([[0, 0, 1], [-0.95, 0.95, 1]]),
    #      softmax=True,
    #  )

    # Large case.
    mdp_human = MDPHuman(
        grid_bounds=np.array([[-1, 1], [-1, 1]]),
        cell_size=np.array([0.05, 0.05]),
        start_location=np.array([-0.75, -0.75]),
        goal_locations=np.array([[0.97, 0.96], [0.76, -0.25]]),
        obstacles=np.array([[0, 0, 1], [-0.95, 0.95, 1]]),
        softmax=True,
        discount_factor=0.999,
        reward_goal=10000,
        reward_obstacle=-1000,
    )

    print("MDP construction time: {} s".format(time.time() - start_time))
    print("")
    print(mdp_human.visualize_grid())
    print("")  # Python 2 and 3 compatibility :)
    print(mdp_human.visualize_vf(width=7))

    print("Testing functions in obstacle")
    print(
        "next_location_coords",
        mdp_human.real_to_grid(
            mdp_human.get_next_location(
                cur_location=mdp_human.center_of_cell(np.array([19, 21])), cur_goal=0
            )
        ),
    )

    print("Testing functions at goal")
    print(
        "next_location",
        mdp_human.get_next_location(cur_location=np.array([0.97, 0.96]), cur_goal=0),
    )
    print(
        "q_values",
        mdp_human.get_q_values(cur_location=np.array([0.97, 0.96]), cur_goal=0),
    )
    print(
        "values", mdp_human.get_values(cur_location=np.array([0.97, 0.96])),
    )

    print("Testing functions on location next to goal")
    print(
        "next_location",
        mdp_human.get_next_location(cur_location=np.array([0.75, 0.75]), cur_goal=0),
    )
    print(
        "q_values",
        mdp_human.get_q_values(cur_location=np.array([0.75, 0.75]), cur_goal=0),
    )
    print(
        "values", mdp_human.get_values(cur_location=np.array([0.75, 0.75])),
    )

    print("Testing functions on another location")
    print(
        "next_location",
        mdp_human.get_next_location(cur_location=np.array([-0.9, -0.75]), cur_goal=0),
    )
    print(
        "q_values",
        mdp_human.get_q_values(cur_location=np.array([-0.9, -0.75]), cur_goal=0),
    )
    print(
        "values", mdp_human.get_values(cur_location=np.array([-0.9, -0.75])),
    )

    print(mdp_human.visualize_grid(human_path=True))

    print("Saving images")
    mdp_human.visualize_grid_img(
        human_path=True, show_vf=True,
    )

    print("Done")

    # For debugging the MDP dynamics.
    #  print("Human position")
    #  print(mdp_human.valid_action[0, 1, 1])
    #  print(mdp_human.next_state[0, 1, 1])
    #  print(mdp_human.reward[0, 1, 1])
    #  print("Next to goal 0 when targeting goal 0")
    #  print(mdp_human.valid_action[0, 8, 9])
    #  print(mdp_human.next_state[0, 8, 9])
    #  print(mdp_human.reward[0, 8, 9])
    #  print("Next to goal 0 when targeting goal 1")
    #  print(mdp_human.valid_action[1, 8, 9])
    #  print(mdp_human.next_state[1, 8, 9])
    #  print(mdp_human.reward[1, 8, 9])
    #  print("At goal 0 when targeting goal 0")
    #  print(mdp_human.valid_action[0, 9, 9])
    #  print(mdp_human.next_state[0, 9, 9])
    #  print(mdp_human.reward[0, 9, 9])
    #  print("At goal 0 when targeting goal 1")
    #  print(mdp_human.valid_action[1, 9, 9])
    #  print(mdp_human.next_state[1, 9, 9])
    #  print(mdp_human.reward[1, 9, 9])
