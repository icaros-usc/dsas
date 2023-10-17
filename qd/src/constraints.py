"""Constraint functions for emitters."""
import numpy as np


def shared_autonomy_two_obj(solution_batch):
    bounds = np.array(
        [
            [0.5, 0.7],  # x bounds
            [-0.05, 0.2],  # y bounds
            [0.5, 0.7],
            [-0.05, 0.2],
            [-0.05, 0.05],  # disturbance bounds
            [-0.05, 0.05],
            [-0.05, 0.05],
            [-0.05, 0.05],
            [-0.05, 0.05],
        ]
    )
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]
    out_of_bounds = np.logical_or(
        solution_batch < np.expand_dims(lower_bounds, axis=0),
        solution_batch > np.expand_dims(upper_bounds, axis=0),
    )
    return ~np.any(out_of_bounds, axis=1)


def collab_task_goal_collision(solution_batch):
    d = 0.05  # Diameter of the goal object

    p0 = solution_batch[:, 0:2]
    p1 = solution_batch[:, 2:4]
    p2 = solution_batch[:, 4:6]
    d0 = np.linalg.norm(p0 - p1, axis=1)
    d1 = np.linalg.norm(p1 - p2, axis=1)
    d2 = np.linalg.norm(p2 - p0, axis=1)

    # Valid if all distances are above d
    return np.all(np.array([d0, d1, d2]) > d, axis=0)


def collab_task_two_workspace(solution_batch):
    goal_collision_constraint = collab_task_goal_collision(solution_batch)

    ws1_bounds = np.array([[0.52, 0.68], [-0.08, 0.18]])
    ws2_bounds = np.array([[0.138, 0.298], [0.496, 0.756]])
    disturbance_bounds = np.array([[-0.05, 0.05]] * 15)

    def in_bounds(bounds, values):
        lb, ub = bounds[:, 0], bounds[:, 1]
        elem_in_bounds = np.logical_and(
            values >= np.expand_dims(lb, axis=0), values <= np.expand_dims(ub, axis=0)
        )
        return np.all(elem_in_bounds, axis=1)

    # Each goal should be in one of the workspaces
    g0_valid = in_bounds(ws1_bounds, solution_batch[:, 0:2]) | in_bounds(
        ws2_bounds, solution_batch[:, 0:2]
    )
    g1_valid = in_bounds(ws1_bounds, solution_batch[:, 2:4]) | in_bounds(
        ws2_bounds, solution_batch[:, 2:4]
    )
    g2_valid = in_bounds(ws1_bounds, solution_batch[:, 4:6]) | in_bounds(
        ws2_bounds, solution_batch[:, 4:6]
    )

    goal_placement_constraint = g0_valid & g1_valid & g2_valid
    disturbance_constraint = in_bounds(disturbance_bounds, solution_batch[:, 6:])

    return goal_collision_constraint & goal_placement_constraint & disturbance_constraint
