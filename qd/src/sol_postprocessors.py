"""Postprocessing functions for solutions."""
import numpy as np
import torch

from src.scenario.repair import repair_objects


def _scale_and_trans_tanh(x, scale, trans):
    # eps stricter scaling since x can be zero and cause floating point errors in
    # comparison during constraint check.
    eps = 1e-6
    if isinstance(x, np.ndarray):
        return (scale - 2 * eps) * np.tanh(x) + trans + eps
    else:
        return (scale - 2 * eps) * torch.tanh(x) + trans + eps


def _fix_coord_two_ws(x, bounds, cond):
    if isinstance(x, np.ndarray):
        # Make a copy if numpy
        x = x.copy()

    x[cond] = _fix_coord_one_ws(x[cond], bounds[0])
    x[~cond] = _fix_coord_one_ws(x[~cond], bounds[1])
    return x


def _fix_coord_one_ws(x, bounds):
    scale = (bounds[1] - bounds[0]) / 2
    trans = bounds[0] + scale
    return _scale_and_trans_tanh(x, scale, trans)


def tanh_two_ws(solution_batch):
    def fix_d(d):
        # No need for eps here since tanh will stay strictly between -1, 1
        if isinstance(d, np.ndarray):
            return 0.05 * np.tanh(d)
        else:
            return 0.05 * torch.tanh(d)

    if isinstance(solution_batch, np.ndarray):
        # Make a copy if numpy
        solution_batch = solution_batch.copy()

    x_bounds = [[0.52, 0.68], [0.138, 0.298]]
    y_bounds = [[-0.08, 0.18], [0.496, 0.756]]

    solution_batch = solution_batch * 1  # To fix pytorch's leaf node in-place issues

    solution_batch[:, 1:9:3] = _fix_coord_two_ws(
        solution_batch[:, 1:9:3], x_bounds, solution_batch[:, 0:9:3] >= 0
    )
    solution_batch[:, 2:9:3] = _fix_coord_two_ws(
        solution_batch[:, 2:9:3], y_bounds, solution_batch[:, 0:9:3] >= 0
    )
    solution_batch[:, 9:] = fix_d(solution_batch[:, 9:])
    return solution_batch


def mip_repair(solution_batch):
    postprocessed_batch = solution_batch.copy()

    # Bounds should probably be from the config since they affect constraint check as
    # well. All bounds are stricter by eps to prevent floating points errors during check.
    eps = 1e-6
    regions = np.array(
        [[0.6, 0.05, 0.08 - eps, 0.13 - eps], [0.218, 0.626, 0.08 - eps, 0.13 - eps]]
    )
    object_to_region = np.ones((3, 2), dtype=bool)
    bounding_rect = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    r = np.ones(3) * 0.025 + eps

    info = []
    for i, sol in enumerate(solution_batch):
        obj_x = sol[1:9:3]
        obj_y = sol[2:9:3]
        objects = np.array(list(zip(obj_x, obj_y, r)))

        new_objs, cost, obj_regions = repair_objects(
            objects, regions, object_to_region, bounding_rect
        )
        postprocessed_batch[i, 1:9:3] = new_objs[:, 0]
        postprocessed_batch[i, 2:9:3] = new_objs[:, 1]

        scaled_x = new_objs[:, 0].copy()
        c1 = scaled_x >= 0.52
        c2 = scaled_x <= 0.298
        scaled_x[c1] -= 0.52
        scaled_x[c2] -= 0.298

        info.append({"mip_cost": cost, "obj_regions": scaled_x})

    return postprocessed_batch, info


def tanh_two_ws_repair(solution_batch):
    return mip_repair(tanh_two_ws(solution_batch))


def clip_one_ws(solution_batch):
    goal_x_bounds = [0.52, 0.68]
    goal_y_bounds = [-0.08, 0.18]
    goal_d_bounds = [-0.05, 0.05]

    obj_x = solution_batch[:, 0:4:2]
    obj_y = solution_batch[:, 1:4:2]
    obj_d = solution_batch[:, 4:]

    if isinstance(solution_batch, np.ndarray):
        dx = np.clip(
            np.maximum(goal_x_bounds[0] - obj_x, obj_x - goal_x_bounds[1]),
            a_min=0,
            a_max=None,
        )
        dy = np.clip(
            np.maximum(goal_y_bounds[0] - obj_y, obj_y - goal_y_bounds[1]),
            a_min=0,
            a_max=None,
        )
        dd = np.clip(
            np.maximum(goal_d_bounds[0] - obj_d, obj_d - goal_d_bounds[1]),
            a_min=0,
            a_max=None,
        )
        reg = dx.sum(axis=-1) ** 2 + dy.sum(axis=-1) ** 2 + dd.sum(axis=-1) ** 2

        clipped_sols = solution_batch.copy()
        clipped_sols[:, 0:4:2] = np.clip(
            solution_batch[:, 0:4:2], a_min=goal_x_bounds[0], a_max=goal_x_bounds[1]
        )
        clipped_sols[:, 1:4:2] = np.clip(
            solution_batch[:, 1:4:2], a_min=goal_y_bounds[0], a_max=goal_y_bounds[1]
        )
        clipped_sols[:, 4:] = np.clip(
            solution_batch[:, 4:], a_min=goal_d_bounds[0], a_max=goal_d_bounds[1]
        )
        return clipped_sols, [{"mip_cost": cost} for cost in reg]
    else:
        dx = torch.clip(
            torch.maximum(goal_x_bounds[0] - obj_x, obj_x - goal_x_bounds[1]),
            min=0,
            max=None,
        )
        dy = torch.clip(
            torch.maximum(goal_y_bounds[0] - obj_y, obj_y - goal_y_bounds[1]),
            min=0,
            max=None,
        )
        dd = torch.clip(
            torch.maximum(goal_d_bounds[0] - obj_d, obj_d - goal_d_bounds[1]),
            min=0,
            max=None,
        )
        reg = dx.sum(dim=-1) ** 2 + dy.sum(dim=-1) ** 2 + dd.sum(dim=-1) ** 2

        clipped_sols = solution_batch * 1
        clipped_sols[:, 0:4:2] = torch.clip(
            solution_batch[:, 0:4:2], min=goal_x_bounds[0], max=goal_x_bounds[1]
        )
        clipped_sols[:, 1:4:2] = torch.clip(
            solution_batch[:, 1:4:2], min=goal_y_bounds[0], max=goal_y_bounds[1]
        )
        clipped_sols[:, 4:] = torch.clip(
            solution_batch[:, 4:], min=goal_d_bounds[0], max=goal_d_bounds[1]
        )
        return clipped_sols, reg  # Hack to use it for Jacobian calc


def noisy_human_repair(solution_batch):
    postprocessed_batch = solution_batch.copy()

    st_bounds = [2.5, 3.5]  # softmax_temperature between 2.5 and 3.5 (will be 10 ^ x)
    vel_bounds = [0.5, 2]  # human_vel_coeff between 0.5 and 2

    dst = np.clip(
        np.maximum(
            st_bounds[0] - solution_batch[:, 9], solution_batch[:, 9] - st_bounds[1]
        ),
        a_min=0,
        a_max=None,
    )
    postprocessed_batch[:, 9] = np.clip(
        solution_batch[:, 9], a_min=st_bounds[0], a_max=st_bounds[1]
    )

    dv = np.clip(
        np.maximum(
            vel_bounds[0] - solution_batch[:, 10], solution_batch[:, 10] - vel_bounds[1]
        ),
        a_min=0,
        a_max=None,
    )
    postprocessed_batch[:, 10] = np.clip(
        solution_batch[:, 10], a_min=vel_bounds[0], a_max=vel_bounds[1]
    )

    cost_human = dst ** 2 + dv ** 2

    postprocessed_batch, info = mip_repair(postprocessed_batch)

    for i, ch in zip(info, cost_human):
        i["mip_cost"] += ch ** 2

    return postprocessed_batch, info
