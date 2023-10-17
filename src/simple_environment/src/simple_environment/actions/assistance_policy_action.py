# creates a shared autonomy policy that runs essentially as an action
import numpy as np

from ada_assistance_policy.AdaHandler import AdaHandler
from ada_assistance_policy.Goal import Goal
from ada_teleoperation.DataRecordingUtils import TrajectoryData
from bypassable_action import BypassableAction


class AssistancePolicyAction(BypassableAction):
    def __init__(self, bypass=False):
        BypassableAction.__init__(self, "ASSISTANCE_POLICY", bypass=bypass)

    def _run(
        self,
        manip,
        objects,
        desired_ee_poses,
        ui_device,
        scenario,
        fix_magnitude_user_command=False,
        blend_level="aggressive",
        blend_only=False,
        filename_trajdata=None,
        plot_trajectory=False,
        simulate_user=False,
    ):
        robot = manip.GetRobot()
        env = robot.GetEnv()

        if filename_trajdata:
            traj_data_recording = TrajectoryData(filename_trajdata)
        else:
            traj_data_recording = None

        if scenario.has_obstacle:
            all_goals = []
            goal_list = [
                (obj.GetTransform(), desired_ee_pose)
                for obj, desired_ee_pose in zip(objects, desired_ee_poses)
            ]
            for gg in range(len(objects)):
                all_goals.append(Goal(goal_list[gg][0], goal_list[gg][1]))
        else:
            def rot_by_180(x):
                rot_z_180_mat = np.array(
                    [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                )
                trans_to_origin_mat = np.eye(4)
                trans_to_origin_mat[:3, 3] = -x[:3, 3]
                trans_back_mat = np.eye(4)
                trans_back_mat[:3, 3] = x[:3, 3]

                x = np.matmul(trans_to_origin_mat, x)
                x = np.matmul(rot_z_180_mat, x)
                x = np.matmul(trans_back_mat, x)

                return x

            all_goals = [
                Goal(obj.GetTransform(), [desired_ee_pose, rot_by_180(desired_ee_pose)])
                for obj, desired_ee_pose in zip(objects, desired_ee_poses)
            ]

        ada_handler = AdaHandler(
            env,
            robot,
            all_goals,
            objects,
            input_interface_name=ui_device,
            num_input_dofs=2,
            use_finger_mode=False,
            scenario=scenario,
        )

        ada_handler.execute_policy(
            simulate_user=simulate_user,
            blend_level=blend_level,
            blend_only=blend_only,
            scenario=scenario,
            fix_magnitude_user_command=fix_magnitude_user_command,
            traj_data_recording=traj_data_recording,
            plot_trajectory=plot_trajectory,
        )
