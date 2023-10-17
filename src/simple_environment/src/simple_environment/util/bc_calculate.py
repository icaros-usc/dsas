import logging
from itertools import combinations

import numpy as np
from scipy.spatial import distance


def calc_distance_between_objects(scenario, scenario_config):
    return np.min(
        [distance.euclidean(pi, pj) for pi, pj in combinations(scenario.points, 2)]
    )


def calc_variation(scenario, scenario_config):
    if scenario.disturbances is None:
        logging.warning("scenario.disturbances is None. Returning 0.")
        return 0
    return np.sqrt(np.sum(np.square(scenario.disturbances)))


def human_path_length(scenario, scenario_config):
    if hasattr(scenario, "human_trajectory") and scenario.human_trajectory:
        path_length = 0
        for (_, pos1), (_, pos2) in zip(
            scenario.human_trajectory[:-1], scenario.human_trajectory[1:]
        ):
            dis = distance.euclidean(pos1, pos2)
            path_length += dis

        return path_length
    else:
        logging.warning("scenario.human_trajectory is not set. Returning 0.")
        return 0


def robot_path_length(scenario, scenario_config):
    if hasattr(scenario, "robot_trajectory") and scenario.robot_trajectory:
        path_length = 0
        for (_, pos1), (_, pos2) in zip(
            scenario.robot_trajectory[:-1], scenario.robot_trajectory[1:]
        ):
            dis = distance.euclidean(pos1, pos2)
            path_length += dis

        return path_length
    else:
        logging.warning("scenario.robot_trajectory is not set. Returning 0.")
        return 0


def highest_wrong_goal_prob(scenario, scenario_config):
    if hasattr(scenario, "max_wrong_goal_prob"):
        return scenario.max_wrong_goal_prob
    else:
        logging.warning("scenario.max_wrong_goal_prob is not set. Returning 0.")
        return 0


def dist_std(scenario, scenario_config):
    return np.std(
        [distance.euclidean(pi, pj) for pi, pj in combinations(scenario.points, 2)]
    )


def human_wait_time(scenario, scenario_config):
    if hasattr(scenario, "human_wait_time"):
        return scenario.human_wait_time
    else:
        logging.warning("scenario.human_wait_time is not set. Returning 0.")
        return 0


def robot_wait_time(scenario, scenario_config):
    if hasattr(scenario, "robot_wait_time"):
        return scenario.robot_wait_time
    else:
        logging.warning("scenario.robot_wait_time is not set. Returning 0.")
        return 0


def total_wait_time(scenario, scenario_config):
    total_time = 0
    if hasattr(scenario, "robot_wait_time"):
        total_time += scenario.robot_wait_time
    else:
        logging.warning(
            "scenario.robot_wait_time is not set. Adding 0 to total_wait_time."
        )

    if hasattr(scenario, "human_wait_time"):
        total_time += scenario.human_wait_time
    else:
        logging.warning(
            "scenario.human_wait_time is not set. Adding 0 to total_wait_time."
        )

    return total_time
