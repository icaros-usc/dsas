import copy
import time

import numpy as np
import rospy

from UserInputMapper import UserInputMapper
from ada_teleoperation.RobotState import RobotState, Action
from input_handlers import KinovaJoystickListener, MouseJoystickListener
from prpy.util import GeodesicTwist, ComputeJointVelocityFromTwist

CONTROL_HZ = 50.0

mouse_interface_name = "mouse"
kinova_joy_interface_name = "kinova"
hydra_interface_name = "hydra"

possible_teleop_interface_names = [
    mouse_interface_name,
    kinova_joy_interface_name,
    hydra_interface_name,
]


def is_done_func_default(*args):
    return False


def is_done_func_button_hold(env, robot, user_input):
    return user_input.buttons_held[0]


class AdaTeleopHandler:
    def __init__(
        self, env, robot, teleop_interface, num_input_dofs, use_finger_mode=True
    ):
        self.env = env
        self.robot = robot

        self.sim = robot.simulated
        self.manip = robot.arm
        self.hand = robot.arm.hand

        num_finger_dofs = len(self.hand.GetIndices())
        Action.set_no_finger_vel(num_finger_dofs)

        # number of different modes used for motion is related to how many dofs of input
        # we have
        if num_input_dofs == 2:
            self.num_motion_modes = 3
        elif num_input_dofs == 3:
            self.num_motion_modes = 2
        else:
            raise Exception("Number of input dofs being used must be 2 or 3")

        # select which input device we will be using
        self.teleop_interface = teleop_interface
        if teleop_interface == mouse_interface_name:
            self.joystick_listener = MouseJoystickListener()
        elif teleop_interface == kinova_joy_interface_name:
            self.joystick_listener = KinovaJoystickListener()
        elif teleop_interface == hydra_interface_name:
            pass
        else:
            raise Exception(
                "Teleop interface not specified. Please specify one of: "
                + str(possible_teleop_interface_names)
            )

        if use_finger_mode:
            num_finger_modes = 1
        else:
            num_finger_modes = 0
        self.user_input_mapper = UserInputMapper(
            interface_listener=self.joystick_listener,
            num_motion_modes=self.num_motion_modes,
            num_finger_modes=num_finger_modes,
        )

        self.init_robot()

    def init_robot(self):
        if not self.sim:
            self.robot.SwitchToTeleopController()

        # set the robot state we keep track of
        self.robot_state = RobotState(
            self.GetEndEffectorTransform(),
            self.hand.GetDOFValues(),
            num_modes=self.user_input_mapper.num_motion_modes
            + self.user_input_mapper.num_finger_modes,
        )

    def GetEndEffectorTransform(self):
        return self.manip.GetEndEffectorTransform()

    def execute_action(self, action):
        self.robot_state.mode = self.robot_state.mode_after_action(action)

        joint_limit_reached = self.execute_twist(action.twist)
        self.execute_finger_velocities(action.finger_vel)

        return joint_limit_reached

    def execute_twist_to_transform(self, target_trans, magnitude=1.0):
        twist = GeodesicTwist(self.manip.GetEndEffectorTransform(), target_trans)
        return self.execute_twist(magnitude * twist)

    # NOTE: twist is stacked [cartesian angular]
    def execute_twist(self, twist):
        joint_vels, twist_opt = ComputeJointVelocityFromTwist(self.robot, twist)
        self.execute_joint_velocities(joint_vels)
        twist_ratio = twist_opt / twist

        # Joint limit reached if translation part of twist_opt much less than twist
        if np.linalg.norm(twist_ratio[:3]) < 0.6:  # was 0.1 for sim
            return True
        else:
            return False

    def execute_joint_velocities(self, joint_velocities):
        # make sure in velocity is in limits
        joint_vel_limits = self.manip.GetVelocityLimits()
        ratio = np.absolute(joint_velocities / joint_vel_limits)

        if np.max(ratio) > 0.95:
            joint_velocities /= np.max(ratio) / 0.95

        self.manip.Servo(joint_velocities)

    def execute_finger_velocities(self, finger_velocities):
        self.hand.Servo(finger_velocities)

    def execute_direct_teleop(
        self, is_done_func=is_done_func_default, traj_data_recording=None
    ):
        robot_state = self.robot_state

        time_per_iter = 1.0 / CONTROL_HZ

        if traj_data_recording:
            traj_data_recording.set_init_info(
                start_state=copy.deepcopy(robot_state),
                input_interface_name=self.teleop_interface,
                assist_type="None",
            )

        user_input_raw = self.joystick_listener.get_most_recent_cmd()
        while not is_done_func(self.env, self.robot, user_input_raw):
            start_time = time.time()
            robot_state.ee_trans = self.GetEndEffectorTransform()
            robot_state.finger_dofs = self.manip.hand.GetDOFValues()
            # ee_trans = robot_state.ee_trans

            user_input_raw = self.joystick_listener.get_most_recent_cmd()
            direct_teleop_action = self.user_input_mapper.input_to_action(
                user_input_raw, robot_state
            )
            self.execute_action(direct_teleop_action)

            if traj_data_recording:
                robot_dof_values = self.robot.GetDOFValues()

                traj_data_recording.add_datapoint(
                    robot_state=copy.deepcopy(robot_state),
                    robot_dof_values=copy.copy(robot_dof_values),
                    user_input_all=copy.deepcopy(user_input_raw),
                    direct_teleop_action=copy.deepcopy(direct_teleop_action),
                    executed_action=copy.deepcopy(direct_teleop_action),
                )

            end_time = time.time()

            rospy.sleep(max(0.0, time_per_iter - (end_time - start_time)))

        # execute zero velocity to stop movement
        self.execute_joint_velocities(np.zeros(len(self.manip.GetDOFValues())))

        if traj_data_recording:
            traj_data_recording.tofile()


def weighted_quadratic_objective(dq, J, dx, *args):
    """
    Quadratic objective function for SciPy's optimization that penalizes translation error
    more then rotation error
    @param dq joint velocity
    @param J Jacobian
    @param dx desired twist
    @return objective the objective function
    @return gradient the analytical gradient of the objective
    """
    error = np.dot(J, dq) - dx
    error *= np.array([3.0, 3.0, 3.0, 1.0, 1.0, 1.0])
    objective = 0.5 * np.dot(np.transpose(error), error)
    gradient = np.dot(np.transpose(J), error)
    return objective, gradient
