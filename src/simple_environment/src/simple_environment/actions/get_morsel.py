import logging

import numpy as np
import openravepy
import prpy.viz
import tf.transformations as transmethods

from assistance_policy_action import AssistancePolicyAction
from bypassable_action import ActionException, BypassableAction
from detect_morsel import morsel_index_to_name
from direct_teleop_action import DirectTeleopAction
from prpy.ik_ranking import MultipleNominalConfigurations
from prpy.planning.base import PlanningError
from simple_environment.scenarios import CollabScenario

logger = logging.getLogger("ada_meal_scenario")


def get_all_morsel_bodies(env):
    all_bodies = []
    for i in range(1000):
        morsel_name = morsel_index_to_name(i)
        morsel_body = env.GetKinBody(morsel_name)
        if morsel_body:
            all_bodies.append(morsel_body)
        else:
            break
    return all_bodies


class GetMorsel(BypassableAction):
    def __init__(self, bypass=False):
        BypassableAction.__init__(self, "GetMorsel", bypass=bypass)

    def _run(
        self,
        manip,
        method,
        blend_level,
        ui_device,
        scenario,
        state_pub=None,
        filename_trajdata=None,
        plot_trajectory=False,
        simulate_user=False,
    ):
        """
        Execute a sequence of plans that pick up the morsel
        @param manip The manipulator
        """
        collab = isinstance(scenario, CollabScenario)
        robot = manip.GetRobot()
        env = robot.GetEnv()
        all_morsels = get_all_morsel_bodies(env)
        # morsel = all_morsels[0]
        if not all_morsels:
            raise ActionException(self, "Failed to find morsel in environment.")

        if scenario.has_obstacle:
            all_desired_ee_pose = [
                get_prestab_pose_for_morsel(morsel, manip) for morsel in all_morsels
            ]
            all_desired_stab_ee_pose = [
                pose for pose in all_desired_ee_pose if pose is not None
            ]
        else:
            all_desired_ee_pose = [morsel.GetTransform() for morsel in all_morsels]
            all_desired_ee_pose = [
                pose for pose in all_desired_ee_pose if pose is not None
            ]
            all_desired_stab_ee_pose = [np.copy(pose) for pose in all_desired_ee_pose]
            zoffset = -0.06
            for pose in all_desired_stab_ee_pose:
                pose[2, 3] += zoffset

        if state_pub:
            state_pub.publish("getting morsel with method " + str(method))
            if filename_trajdata and "direct" not in method:
                state_pub.publish("recording data to " + str(filename_trajdata))

        if "shared_auton" in method:
            if method == "shared_auton_prop":
                fix_magnitude_user_command = True
            else:
                fix_magnitude_user_command = False
            assistance_policy_action = AssistancePolicyAction(bypass=self.bypass)
            # print all_desired_ee_pose
            assistance_policy_action.execute(
                manip,
                all_morsels,
                all_desired_ee_pose,
                ui_device,
                scenario=scenario,
                fix_magnitude_user_command=fix_magnitude_user_command,
                filename_trajdata=filename_trajdata,
                plot_trajectory=plot_trajectory,
                simulate_user=simulate_user,
            )
        elif method == "blend":
            if collab:
                raise NotImplementedError("Blend not implemented for collaborative tasks")
            assistance_policy_action = AssistancePolicyAction(bypass=self.bypass)
            assistance_policy_action.execute(
                manip,
                all_morsels,
                all_desired_ee_pose,
                ui_device,
                scenario=scenario,
                blend_level=blend_level,
                blend_only=True,
                filename_trajdata=filename_trajdata,
                plot_trajectory=plot_trajectory,
                simulate_user=simulate_user,
            )
        elif method == "direct":
            if collab:
                raise NotImplementedError(
                    "Direct teleop not implemented for collaborative tasks"
                )
            direct_teleop_action = DirectTeleopAction(bypass=self.bypass)
            direct_teleop_action.execute(
                manip, ui_device, filename_trajdata=filename_trajdata
            )
        elif method == "autonomous":
            desired_ee_pose = all_desired_ee_pose[0]
            try:
                with prpy.viz.RenderPoses([desired_ee_pose], env):

                    # since we know we will soon go to stabbed, rank iks based on both
                    # stabbed and current
                    ik_ranking_nominal_configs = [
                        robot.arm.GetDOFValues(),
                        np.array(
                            robot.configurations.get_configuration(
                                "ada_meal_scenario_morselStabbedConfiguration"
                            )[1]
                        ),
                    ]
                    ik_ranker = MultipleNominalConfigurations(ik_ranking_nominal_configs)
                    path = robot.PlanToEndEffectorPose(
                        desired_ee_pose, execute=True, ranker=ik_ranker
                    )

            except PlanningError, e:
                raise ActionException(
                    self, "Failed to plan to pose near morsel: %s" % str(e)
                )


def get_prestab_pose_for_morsel(morsel, manip):

    obj_transform = morsel.GetTransform()

    obj_aabb = morsel.ComputeAABB()

    # for bowl objects from here

    angle1 = np.pi / 6
    angle2 = -np.pi / 6
    angle3 = 0
    roty1 = transmethods.rotation_matrix(angle1, np.array([0.0, 1, 0.0]))
    roty2 = transmethods.rotation_matrix(angle2, np.array([0.0, 1, 0.0]))
    roty3 = transmethods.rotation_matrix(angle3, np.array([0.0, 1, 0.0]))

    # rotz = transmethods.rotation_matrix(np.pi/3, np.array([0.,0.,1.]))

    hand_trans_base = np.eye(4)
    # hand_trans_base = np.dot(rotx, np.dot(rotx, hand_trans_base))

    # we will sample a point on top object by sampling a number from 0 to 2pi
    # and sampling a point on circle
    # hand will be rotated to keep 2 fingers outside
    poses = []

    hand_trans = np.copy(hand_trans_base)
    hand_trans = np.dot(roty1, hand_trans)
    poses.append(np.dot(obj_transform, hand_trans))

    hand_trans = np.copy(hand_trans_base)
    hand_trans = np.dot(roty2, hand_trans)
    poses.append(np.dot(obj_transform, hand_trans))

    hand_trans = np.copy(hand_trans_base)
    hand_trans = np.dot(roty3, hand_trans)
    poses.append(np.dot(obj_transform, hand_trans))

    valid_poses = []
    robot = manip.GetRobot()
    with robot:
        # logger.info('looking for ik for morsel ' + morsel.GetName())
        ik_filter_options = openravepy.IkFilterOptions.CheckEnvCollisions
        # first call FindIKSolution which is faster if it succeeds
        for pp in range(len(poses)):
            ik_sol = manip.FindIKSolution(poses[pp], ik_filter_options)
            if ik_sol is not None:
                valid_poses.append(poses[pp])

        if valid_poses == []:
            for pp in range(len(poses)):
                ik_sols = manip.FindIKSolutions(poses[pp], ik_filter_options)
                if ik_sols is not None:
                    valid_poses.append(poses[pp])
                    break

    if valid_poses == []:
        raise ValueError("no valid pose found!")

    return valid_poses
