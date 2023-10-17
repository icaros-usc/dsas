import logging

import numpy as np
import os
from catkin.find_in_workspaces import find_in_workspaces

from bypassable_action import BypassableAction
from simple_environment.scenarios import CollabScenario

logger = logging.getLogger("simple_environment")

# name which we will add indices to
waypoint_base_name = "waypoint"


def waypoint_index_to_name(ind):
    return waypoint_base_name + str(ind)


class PlanHumanPath(BypassableAction):
    def __init__(self, scenario, bypass=False):
        BypassableAction.__init__(self, "PlanHumanPath", bypass=bypass)
        self.scenario = scenario

        self.collab = isinstance(scenario, CollabScenario)

    def _run(self, robot, timeout=None):
        raise NotImplementedError("ERROR! WE SHOULD NOT BE HERE")

    def _bypass(self, robot):
        h_planner = HumanPathPlanner(robot, self.scenario)

        if hasattr(self.scenario, "waypoints") and self.scenario.waypoints is not None:
            if self.collab:
                waypoint_list = self.scenario.waypoints.reshape((-1, 3))
            else:
                waypoint_list = self.scenario.waypoints
            num_waypoints = len(waypoint_list)

            for i in range(num_waypoints):
                h_planner.add_waypoint(waypoint_index_to_name(i))

            env = robot.GetEnv()
            all_waypoints = get_all_waypoints_in_env(env)
            num_waypoints_before_filter = len(all_waypoints)

            # remove the kinbodies we used in previous timesteps not used here
            self.remove_waypoints_next_indices(
                robot.GetEnv(), num_waypoints, end_ind=num_waypoints_before_filter
            )

        if self.collab:
            h_planner.add_human_sphere()

    def remove_waypoints_next_indices(self, env, start_ind, end_ind=0):
        """ Removes the OpenRAVE kin bodies for all morsels with index at
        or greater than the start index
        If end index is specified, will remove morsels up to that index
        Otherwise, will check indices until no morsels with index i is in the environment

        @param env the OpenRAVE environment
        @param start_ind the index to start checking
        @param end_ind the (optional) index to check morsels up until
        """

        ind = start_ind
        waypoint_body = env.GetKinBody(waypoint_index_to_name(ind))
        while waypoint_body or ind < end_ind:
            if waypoint_body:
                env.Remove(waypoint_body)
            ind += 1
            waypoint_body = env.GetKinBody(waypoint_index_to_name(ind))


class HumanPathPlanner:
    def __init__(self, robot, scenario):
        self.env = robot.GetEnv()
        self.robot = robot
        self.scenario = scenario
        self.sub = None

        # True if collaboration task
        self.collab = isinstance(scenario, CollabScenario)

        self.object_base_path = find_in_workspaces(
            search_dirs=["share"],
            project="simple_environment",
            path="data",
            first_match_only=True,
        )[0]

    def add_waypoint(self, waypoint_name=None):
        if self.collab:
            waypoint_list = self.scenario.waypoints.reshape((-1, 3))
        else:
            waypoint_list = self.scenario.waypoints

        did_find = False
        for i, pos in enumerate(waypoint_list):
            test_name = "waypoint{}".format(i)
            if waypoint_name == test_name:
                waypoint_in_world = np.array(
                    [
                        [0, 0, 1, pos[0]],
                        [1, 0, 0, pos[1]],
                        [0, 1, 0, pos[2]],
                        [0, 0, 0, 1],
                    ]
                )
                did_find = True
                break

        if not did_find:
            raise ValueError("Incorrect waypoint name")

        import openravepy.misc

        # h1 = openravepy.misc.DrawAxes(self.env, camera_in_world)
        h2 = openravepy.misc.DrawAxes(self.env, waypoint_in_world)

        if waypoint_name is None:
            waypoint_name = "waypoint"

        ball_path = os.path.join(
            self.object_base_path, "objects", "tinysphere.kinbody.xml"
        )
        self._safe_add_obj(waypoint_name, ball_path, waypoint_in_world)

    def add_human_sphere(self, sphere_name="human_sphere"):
        if not self.collab:
            raise NotImplementedError(
                "Human sphere can only be added in collaborative tasks"
            )

        sphere_path = os.path.join(
            self.object_base_path, "objects", "largesphere.kinbody.xml"
        )
        human_pos = self.scenario.human_position

        waypoint_in_world = np.array(
            [
                [0, 0, 1, human_pos[0]],
                [1, 0, 0, human_pos[1]],
                [0, 1, 0, human_pos[2]],
                [0, 0, 0, 1],
            ]
        )
        self._safe_add_obj(sphere_name, sphere_path, waypoint_in_world)

    def _safe_add_obj(self, obj_name, obj_path, obj_transform):
        if self.env.GetKinBody(obj_name) is None:
            with self.env:
                obj = self.env.ReadKinBodyURI(obj_path)
                obj.SetName(obj_name)
                self.env.Add(obj)
                obj.Enable(False)
        else:
            obj = self.env.GetKinBody(obj_name)
        obj.SetTransform(obj_transform)


def get_all_waypoints_in_env(env, start_ind=0, end_ind=0):
    """ Tries to get all the morsels in the environment, based on naming
    Assumes the function morsel_index_to_name was used to name all morsels
    And all morsel numbers are consecutive

    @param env the OpenRAVE environment
    @param start_ind the index to start checking
    @param end_ind the (optional) index to check morsels up until. If unspecifed, 
            will continue to check until no bodies are found
    """

    all_waypoints = []
    ind = start_ind
    waypoint_body = env.GetKinBody(waypoint_index_to_name(ind))
    while waypoint_body or ind < end_ind:
        if waypoint_body:
            all_waypoints.append(waypoint_body)
        ind += 1
        waypoint_body = env.GetKinBody(waypoint_index_to_name(ind))
    return all_waypoints
