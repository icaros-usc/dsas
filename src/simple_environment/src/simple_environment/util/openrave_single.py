import warnings

import numpy as np
from catkin.find_in_workspaces import find_in_workspaces
from std_msgs.msg import String

import adapy
import logging
import openravepy
import os
import prpy
import rospy
from prpy.planning.base import PlanningError
from simple_environment.actions.bite_serving import BiteServing
from simple_environment.actions.bypassable_action import ActionException

warnings.simplefilter(action="ignore", category=FutureWarning)

project_name = "simple_environment"
logger = logging.getLogger(project_name)

# start_config = np.array([-2.38, -0.1959051, -0.34364456, -0.688872349, 0, 3.55386329])
start_config = np.array([-2.38, -0.43, -0.28, -0.75, 0, 3.55])


class OpenraveSingle:
    def __init__(self, pid, elite_map_config, simulate_user=True, simulate_robot=True):
        self.id = pid
        self.elite_map_config = elite_map_config
        self.simulate_user = simulate_user
        self.simulate_robot = simulate_robot
        self.method = elite_map_config["method"]
        self.blend_level = elite_map_config["blend_level"]

    def start(self):
        self.init_environment("starting_process")

    def setup(self, sim=False, viewer=None, debug=True):
        # find the openrave environment file
        data_base_path = find_in_workspaces(
            search_dirs=["share"],
            project=project_name,
            path="data",
            first_match_only=True,
        )
        if len(data_base_path) == 0:
            raise Exception(
                "Unable to find environment path. Did you source devel/setup.bash?"
            )
        env_path = os.path.join(data_base_path[0], "environments", "table.env.xml")

        # Initialize logging
        if debug:
            openravepy.RaveInitialize(True, level=openravepy.DebugLevel.Debug)
        else:
            openravepy.RaveInitialize(True, level=openravepy.DebugLevel.Info)
        openravepy.misc.InitOpenRAVELogging()
        prpy.logger.initialize_logging()

        # Load the environment and robot
        self.env, self.robot = adapy.initialize(
            attach_viewer=viewer, sim=sim, env_path=env_path
        )

        # Set the active manipulator on the robot
        self.robot.arm.SetActive()

        # else:
        START_ROBOT_POS = self.elite_map_config["start_robot_pos"]
        robot_pose = np.array(
            [
                [1.0, 0.0, 0.0, START_ROBOT_POS[0]],
                [0.0, 1.0, 0.0, START_ROBOT_POS[1]],
                [0.0, 0.0, 1.0, 0.795],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        with self.env:
            self.robot.SetTransform(robot_pose)

        self.reset_trial()

    def reset_trial(self):
        # self.env.GetViewer().Reset()
        # set the robot to the start configuration for next trial
        logger.info("Resetting Robot")
        if self.robot.simulated:
            self.robot.arm.SetDOFValues(start_config)
            values = self.robot.GetDOFValues()
            values[6] = 0.15
            values[7] = 0.15
            self.robot.SetDOFValues(values)

        else:
            # first try to plan to serving
            try:
                self.robot.arm.hand.OpenHand()
                self.robot.arm.PlanToConfiguration(start_config, execute=True)
            except PlanningError, e:
                logger.error("Failed to plan to start config")
                # if it doesn't work, unload controllers

    @staticmethod
    def setup_trial_recording(record_next_trial, file_directory_user):
        # creates user directory if we will be recording
        if record_next_trial and not os.path.exists(file_directory_user):
            os.makedirs(file_directory_user)

    def evaluate(self, scenario):
        if self.method == "blend":
            scenario.max_time = 20

        times = []
        for i in range(0, 1):
            try:
                # robot set initial position
                self.robot.arm.SetDOFValues(start_config)

                # print(self.robot.arm.GetEndEffectorTransform())
                self.manip = self.robot.GetActiveManipulator()
                action = BiteServing()

                action.execute(
                    self.manip,
                    self.env,
                    method=self.method,
                    blend_level=self.blend_level,
                    ui_device="kinova",
                    state_pub=self.state_pub,
                    detection_sim=True,
                    record_trial=False,
                    scenario=scenario,
                    plot_trajectory=True,
                    simulate_user=self.simulate_user,
                )

            except ActionException, e:
                logger.info("Failed to complete bite serving: %s" % str(e))

            self.reset_trial()
            times.append(scenario.getTime())

        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Min time is: " + str(min(times))
        return min(times)

    def init_environment(self, rospyname):
        self.state_pub = rospy.Publisher("ada_tasks", String, queue_size=10)
        rospy.init_node(rospyname, anonymous=True)
        self.setup(sim=self.simulate_robot, viewer="InteractiveMarker", debug=True)
