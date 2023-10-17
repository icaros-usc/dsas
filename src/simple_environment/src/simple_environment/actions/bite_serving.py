from bypassable_action import BypassableAction, ActionException
from trajectory_actions import LookAtPlate, Serve
from detect_morsel import DetectMorsel
from plan_human_path import PlanHumanPath
from get_morsel import GetMorsel
from direct_teleop_action import DirectTeleopAction
from std_msgs.msg import String
import numpy as np

from ada_teleoperation.DataRecordingUtils import *

import logging

logger = logging.getLogger("simple_environment")


class BiteServing(BypassableAction):
    def __init__(self, bypass=False):
        BypassableAction.__init__(self, "BiteServing", bypass=bypass)

    def execute(
        self,
        manip,
        env,
        method,
        blend_level,
        ui_device,
        state_pub,
        scenario,
        detection_sim=False,
        record_trial=False,
        file_directory=None,
        plot_trajectory=False,
        simulate_user=False,
    ):

        if record_trial:
            if file_directory is None:
                file_directory = (
                    rospkg.RosPack().get_path("simple_environment") + "/trajectory_data"
                )

            rosbag_topic_names = [
                "/ada_tasks",
                "/hydra_calib",
                "/ada/joy",
                "/perception/morsel_detection",
                "/joint_states",
            ]
            filename_trajdata, filename_bag = get_next_filename_pair(
                file_directory=file_directory
            )

            rosbag_process = start_rosbag(rosbag_topic_names, filename=filename_bag)
            state_pub.publish("recording data to " + str(filename_bag))
        else:
            filename_trajdata = None

        try:
            # Place waypoints for human motion
            action = PlanHumanPath(scenario, bypass=True)
            state_pub.publish(action.name)
            action.execute(manip.GetRobot())

            # Place goal objects
            action = DetectMorsel(scenario, bypass=detection_sim)
            state_pub.publish(action.name)
            action.execute(manip.GetRobot())

            # Move to get object
            action = GetMorsel(bypass=self.bypass)
            state_pub.publish(action.name)

            action.execute(
                manip,
                method,
                blend_level,
                ui_device,
                state_pub=state_pub,
                scenario=scenario,
                filename_trajdata=filename_trajdata,
                plot_trajectory=plot_trajectory,
                simulate_user=simulate_user,
            )

            state_pub.publish("Finished bite serving")
            if record_trial:
                stop_rosbag(rosbag_process)

        except ActionException, e:
            state_pub.publish("Failed to run bite serving")
            if record_trial:
                stop_rosbag(rosbag_process)
            raise Exception(e.message)
