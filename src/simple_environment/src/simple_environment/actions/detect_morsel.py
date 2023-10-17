import logging
import os

import numpy
from catkin.find_in_workspaces import find_in_workspaces

from bypassable_action import BypassableAction

logger = logging.getLogger("simple_environment")

# name which we will add indices to
morsel_base_name = "morsel"


def morsel_index_to_name(ind):
    return morsel_base_name + str(ind)


class DetectMorsel(BypassableAction):
    def __init__(self, scenario, bypass=False):
        BypassableAction.__init__(self, "DetectBite", bypass=bypass)
        self.scenario = scenario

    def _run(self, robot, timeout=None):
        raise NotImplementedError("ERROR! WE SHOULD NOT BE HERE")

    def _bypass(self, robot):
        m_detector = MorselDetector(robot, self.scenario)
        num_morsels = len(self.scenario.points)
        for i in range(num_morsels):
            m_detector.add_morsel(morsel_index_to_name(i))

        if self.scenario.has_obstacle:
            m_detector.add_obstacle()

        env = robot.GetEnv()
        all_morsels = get_all_morsels_in_env(env)
        num_morsels_before_filter = len(all_morsels)

        # remove the kinbodies we used in previous timesteps not used here
        self.remove_morsels_next_indices(
            robot.GetEnv(), num_morsels, end_ind=num_morsels_before_filter
        )

    def remove_morsels_next_indices(self, env, start_ind, end_ind=0):
        """ Removes the OpenRAVE kin bodies for all morsels with index at
        or greater than the start index
        If end index is specified, will remove morsels up to that index
        Otherwise, will check indices until no morsels with index i is in the environment

        @param env the OpenRAVE environment
        @param start_ind the index to start checking
        @param end_ind the (optional) index to check morsels up until
        """

        ind = start_ind
        morsel_body = env.GetKinBody(morsel_index_to_name(ind))
        while morsel_body or ind < end_ind:
            if morsel_body:
                env.Remove(morsel_body)
            ind += 1
            morsel_body = env.GetKinBody(morsel_index_to_name(ind))


class MorselDetector(object):
    def __init__(self, robot, scenario):
        self.env = robot.GetEnv()
        self.robot = robot
        self.scenario = scenario
        self.sub = None

    def add_obstacle(self):
        obstacle_name = "obstacle"
        sphere_name = "sphere"
        obstacle_pos = self.scenario.obstacle_pos
        obstacle_in_world = numpy.array(
            [
                [0, 0, 1, obstacle_pos[0]],
                [1, 0, 0, obstacle_pos[1]],
                [0, 1, 0, obstacle_pos[2] - 0.05],
                [0, 0, 0, 1],
            ]
        )
        sphere_in_world = numpy.array(
            [
                [0, 0, 1, obstacle_pos[0]],
                [1, 0, 0, obstacle_pos[1]],
                [0, 1, 0, obstacle_pos[2]],
                [0, 0, 0, 1],
            ]
        )

        object_base_path = find_in_workspaces(
            search_dirs=["share"],
            project="simple_environment",
            path="data",
            first_match_only=True,
        )[0]
        box_path = os.path.join(
            object_base_path, "objects", "obstacle_bottle.kinbody.xml"
        )
        ball_path = os.path.join(object_base_path, "objects", "largesphere.kinbody.xml")

        if self.env.GetKinBody(obstacle_name) is None:
            with self.env:
                obstacle = self.env.ReadKinBodyURI(box_path, {"scalegeometry": "0.8"})
                sphere = self.env.ReadKinBodyURI(ball_path)

                obstacle.SetName(obstacle_name)
                self.env.Add(obstacle)
                sphere.SetName(sphere_name)
                sphere.GetLinks()[0].GetGeometries()[0].SetTransparency(0.8)
                sphere.Enable(False)

                self.env.Add(sphere)
                obstacle.Enable(True)
        else:
            obstacle = self.env.GetKinBody(obstacle_name)
            sphere = self.env.GetKinBody(sphere_name)

        obstacle.SetTransform(obstacle_in_world)
        sphere.SetTransform(sphere_in_world)

    def add_morsel(self, morsel_name=None):
        did_find = False
        for i, pos in enumerate(self.scenario.points):
            test_name = "morsel{}".format(i)
            if morsel_name == test_name:
                morsel_in_world = numpy.array(
                    [[0, 0, 1, pos[0]], [1, 0, 0, pos[1]], [0, 1, 0, 0.96], [0, 0, 0, 1]]
                )
                did_find = True
                break

        if not did_find:
            raise ValueError("Incorrect waypoint name")

        import openravepy.misc

        # h1 = openravepy.misc.DrawAxes(self.env, camera_in_world)
        h2 = openravepy.misc.DrawAxes(self.env, morsel_in_world)

        if morsel_name is None:
            morsel_name = "morsel"

        object_base_path = find_in_workspaces(
            search_dirs=["share"],
            project="simple_environment",
            path="data",
            first_match_only=True,
        )[0]
        ball_path = os.path.join(object_base_path, "objects", "goal_box.kinbody.xml")
        if self.env.GetKinBody(morsel_name) is None:
            with self.env:
                morsel = self.env.ReadKinBodyURI(ball_path, {"scalegeometry": "0.8"})
                morsel.SetName(morsel_name)
                self.env.Add(morsel)
                morsel.Enable(False)
        else:
            morsel = self.env.GetKinBody(morsel_name)
        morsel.SetTransform(morsel_in_world)


def get_all_morsels_in_env(env, start_ind=0, end_ind=0):
    """ Tries to get all the morsels in the environment, based on naming
    Assumes the function morsel_index_to_name was used to name all morsels
    And all morsel numbers are consecutive

    @param env the OpenRAVE environment
    @param start_ind the index to start checking
    @param end_ind the (optional) index to check morsels up until. If unspecifed, 
            will continue to check until no bodies are found
    """

    all_morsels = []
    ind = start_ind
    morsel_body = env.GetKinBody(morsel_index_to_name(ind))
    while morsel_body or ind < end_ind:
        if morsel_body:
            all_morsels.append(morsel_body)
        ind += 1
        morsel_body = env.GetKinBody(morsel_index_to_name(ind))
    return all_morsels
