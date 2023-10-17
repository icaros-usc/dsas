import numpy as np
import tf.transformations as transmethods

import AssistancePolicyOneTarget
from ada_assistance_policy.Utils import QuaternionDistance
from ada_assistance_policy.collision_detector import (
    is_inside_sphere,
    line_sphere_intersection,
)

ACTION_DIMENSION = 6


class HuberAssistancePolicyObstacle(AssistancePolicyOneTarget.AssistancePolicyOneTarget):
    def __init__(self, pose, obstacle_pos, obstacle_radius, obstacle_padding):
        super(HuberAssistancePolicyObstacle, self).__init__(pose)
        self.set_constants(
            self.TRANSLATION_LINEAR_MULTIPLIER,
            self.TRANSLATION_DELTA_SWITCH,
            self.TRANSLATION_CONSTANT_ADD,
            self.ROTATION_LINEAR_MULTIPLIER,
            self.ROTATION_DELTA_SWITCH,
            self.ROTATION_CONSTANT_ADD,
            self.ROTATION_MULTIPLIER,
        )
        self.obstacle_pos = obstacle_pos
        self.obstacle_radius = obstacle_radius
        self.obstacle_padding = obstacle_padding

        self.dist_translation = None
        self.dist_translation_aftertrans = None
        self.robot_state = None
        self.robot_state_after_action = None
        self.quat_curr = None
        self.quat_after_trans = None
        self.dist_rotation = None
        self.dist_rotation_aftertrans = None

    def reset(self):
        self.dist_translation = None
        self.dist_translation_aftertrans = None
        self.robot_state = None
        self.robot_state_after_action = None
        self.quat_curr = None
        self.quat_after_trans = None
        self.dist_rotation = None
        self.dist_rotation_aftertrans = None

    def repair_point(self, point):
        robot_pos = self.robot_state.ee_trans[0:3, 3]

        # if it's inside the padded sphere try without padding
        if is_inside_sphere(
            robot_pos, self.obstacle_pos, self.obstacle_radius, self.obstacle_padding
        ):
            padding = 0
        else:
            padding = self.obstacle_padding

        max_motions = 1000
        new_point1 = point.copy()
        new_point2 = point.copy()

        motions = 0
        collision_free = False
        while not collision_free and motions < max_motions:
            new_point1[1] = new_point1[1] + 0.01
            res = line_sphere_intersection(
                robot_pos, new_point1, self.obstacle_pos, self.obstacle_radius, padding
            )
            if not res:
                collision_free = True
            motions = motions + 1
        new_point1[1] = new_point1[1] + 0.03  # move the point a bit more

        motions = 0
        collision_free = False
        while not collision_free and motions < max_motions:
            new_point2[1] = new_point2[1] - 0.01
            res = line_sphere_intersection(
                robot_pos, new_point2, self.obstacle_pos, self.obstacle_radius, padding
            )
            if not res:
                collision_free = True
            motions = motions + 1
        new_point2[1] = new_point2[1] - 0.03  # move the point a bit more

        d1 = np.linalg.norm(robot_pos - new_point1) + np.linalg.norm(
            new_point1 - self.goal_pos
        )
        d2 = np.linalg.norm(robot_pos - new_point2) + np.linalg.norm(
            new_point2 - self.goal_pos
        )
        if d1 <= d2:
            return new_point1
        else:
            return new_point2

    def compute_dist_translation(self):
        robot_pos = self.robot_state.ee_trans[0:3, 3]
        robot_pos_after = self.robot_state_after_action.ee_trans[0:3, 3]

        res = line_sphere_intersection(
            robot_pos,
            self.goal_pos,
            self.obstacle_pos,
            self.obstacle_radius,
            self.obstacle_padding,
        )
        if not res:
            self.dist_translation = np.linalg.norm(robot_pos - self.goal_pos)
            self.dist_translation_aftertrans = np.linalg.norm(
                robot_pos_after - self.goal_pos
            )
        elif len(res) == 1:
            point = res[0]
            repaired_point = self.repair_point(point)
            self.dist_translation = np.linalg.norm(
                robot_pos - repaired_point
            ) + np.linalg.norm(repaired_point - self.goal_pos)
            self.dist_translation_aftertrans = np.linalg.norm(
                robot_pos_after - repaired_point
            ) + np.linalg.norm(repaired_point - self.goal_pos)
        elif len(res) == 2:
            point2 = res[1]  # get the furthest away point for speed
            repaired_point2 = self.repair_point(point2)
            self.dist_translation = np.linalg.norm(
                robot_pos - repaired_point2
            ) + np.linalg.norm(repaired_point2 - self.goal_pos)
            self.dist_translation_aftertrans = np.linalg.norm(
                robot_pos_after - repaired_point2
            ) + np.linalg.norm(repaired_point2 - self.goal_pos)

    def compute_translation_diff(self):
        robot_pos_after = self.robot_state_after_action.ee_trans[0:3, 3]

        # if it has collided
        if is_inside_sphere(robot_pos_after, self.obstacle_pos, self.obstacle_radius, 0):
            translation_diff = -(robot_pos_after - self.obstacle_pos)
            dist_to_go = np.linalg.norm(translation_diff)
            return translation_diff, dist_to_go

        res = line_sphere_intersection(
            robot_pos_after,
            self.goal_pos,
            self.obstacle_pos,
            self.obstacle_radius,
            self.obstacle_padding,
        )
        # res = []

        if not res:
            translation_diff = robot_pos_after - self.goal_pos
            dist_to_go = np.linalg.norm(translation_diff)
        elif len(res) == 1:
            point1 = res[0]
            rep_point1 = self.repair_point(point1)
            translation_diff = robot_pos_after - rep_point1
            dist_to_go = np.linalg.norm(translation_diff)
        else:
            point1 = res[1]
            rep_point1 = self.repair_point(point1)
            translation_diff = robot_pos_after - rep_point1
            dist_to_go = np.linalg.norm(translation_diff)

        return translation_diff, dist_to_go

    def update(self, robot_state, user_action):
        self.robot_state = robot_state
        self.robot_state_after_action = robot_state
        self.compute_dist_translation()

        self.quat_curr = transmethods.quaternion_from_matrix(robot_state.ee_trans)
        self.quat_after_trans = transmethods.quaternion_from_matrix(
            self.robot_state_after_action.ee_trans
        )

        self.dist_rotation = QuaternionDistance(self.quat_curr, self.goal_quat)
        self.dist_rotation_aftertrans = QuaternionDistance(
            self.quat_after_trans, self.goal_quat
        )

    def get_action(self):
        return -self.get_q_derivative()

    def get_cost(self):
        return self.get_cost_translation() + self.get_cost_rotation()

    def get_value(self):
        return self.get_value_translation() + self.get_value_rotation()

    def get_qvalue(self):
        return self.get_qvalue_translation() + self.get_qvalue_rotation()

    def get_q_derivative(self):
        q_rot = self.get_qderivative_rotation()
        q_trans = self.get_qderivative_translation()
        return np.append(q_trans, q_rot)

    # parts split into translation and rotation
    def get_value_translation(self, dist_translation=None):
        if dist_translation is None:
            dist_translation = self.dist_translation

        if dist_translation <= self.TRANSLATION_DELTA_SWITCH:
            return (
                self.TRANSLATION_QUADRATIC_COST_MULTPLIER_HALF
                * dist_translation
                * dist_translation
                + self.TRANSLATION_CONSTANT_ADD * dist_translation
            )
        else:
            return (
                self.TRANSLATION_LINEAR_COST_MULT_TOTAL * dist_translation
                - self.TRANSLATION_LINEAR_COST_SUBTRACT
            )

    def get_cost_translation(self, dist_translation=None):
        if dist_translation is None:
            dist_translation = self.dist_translation

        if dist_translation > self.TRANSLATION_DELTA_SWITCH:
            return self.ACTION_APPLY_TIME * self.TRANSLATION_LINEAR_COST_MULT_TOTAL
        else:
            return self.ACTION_APPLY_TIME * (
                self.TRANSLATION_QUADRATIC_COST_MULTPLIER * dist_translation
                + self.TRANSLATION_CONSTANT_ADD
            )

    def get_qvalue_translation(self):
        return self.get_cost_translation() + self.get_value_translation(
            self.dist_translation_aftertrans
        )

    def get_qderivative_translation(self):
        translation_diff, dist_to_go = self.compute_translation_diff()
        if dist_to_go > self.TRANSLATION_DELTA_SWITCH:
            translation_derivative = self.TRANSLATION_LINEAR_COST_MULT_TOTAL * (
                translation_diff / dist_to_go
            )
        else:
            translation_derivative = self.TRANSLATION_CONSTANT_ADD * (
                translation_diff / dist_to_go
            )
            translation_derivative += (
                self.TRANSLATION_QUADRATIC_COST_MULTPLIER * translation_diff
            )

        # hacky part to make it not jumpy
        dist_translation_limit = 2e-2
        if dist_to_go < dist_translation_limit:
            translation_derivative *= dist_to_go / dist_translation_limit

        return translation_derivative / self.ROBOT_TRANSLATION_COST_MULTIPLIER

    def get_value_rotation(self, dist_rotation=None):
        if dist_rotation is None:
            dist_rotation = self.dist_rotation

        if dist_rotation <= self.ROTATION_DELTA_SWITCH:
            return self.ROTATION_MULTIPLIER * (
                self.ROTATION_QUADRATIC_COST_MULTPLIER_HALF
                * dist_rotation
                * dist_rotation
                + self.ROTATION_CONSTANT_ADD * dist_rotation
            )
        else:
            return self.ROTATION_MULTIPLIER * (
                self.ROTATION_LINEAR_COST_MULT_TOTAL * dist_rotation
                - self.ROTATION_LINEAR_COST_SUBTRACT
            )

    def get_cost_rotation(self, dist_rotation=None):
        if dist_rotation is None:
            dist_rotation = self.dist_rotation

        if dist_rotation > self.ROTATION_DELTA_SWITCH:
            return (
                self.ACTION_APPLY_TIME
                * self.ROTATION_MULTIPLIER
                * self.ROTATION_LINEAR_COST_MULT_TOTAL
            )
        else:
            return (
                self.ACTION_APPLY_TIME
                * self.ROTATION_MULTIPLIER
                * (
                    self.ROTATION_QUADRATIC_COST_MULTPLIER * dist_rotation
                    + self.ROTATION_CONSTANT_ADD
                )
            )

    def get_qvalue_rotation(self):
        return self.get_cost_rotation() + self.get_value_rotation(
            self.dist_rotation_aftertrans
        )

    def get_qderivative_rotation(self):
        quat_between = transmethods.quaternion_multiply(
            self.goal_quat, transmethods.quaternion_inverse(self.quat_after_trans)
        )

        rotation_derivative = quat_between[0:-1]
        rotation_derivative /= np.linalg.norm(rotation_derivative)

        if self.dist_rotation_aftertrans > self.ROTATION_DELTA_SWITCH:
            rotation_derivative_magnitude = self.ROTATION_LINEAR_COST_MULT_TOTAL
        else:
            rotation_derivative_magnitude = self.ROTATION_CONSTANT_ADD
            rotation_derivative_magnitude += (
                self.ROTATION_QUADRATIC_COST_MULTPLIER * self.dist_rotation_aftertrans
            )

        rotation_derivative *= (
            self.ROTATION_MULTIPLIER
            * rotation_derivative_magnitude
            / self.ROBOT_ROTATION_COST_MULTIPLIER
        )

        if (np.sum(self.goal_quat * self.quat_after_trans)) > 0:
            rotation_derivative *= -1

        # hacky part to make it not jumpy
        dist_rotation_limit = np.pi / 12.0
        if self.dist_rotation_aftertrans < dist_rotation_limit:
            rotation_derivative *= self.dist_rotation_aftertrans / dist_rotation_limit

        return rotation_derivative

    # HUBER CONSTANTS
    # Values used when assistance always on
    TRANSLATION_LINEAR_MULTIPLIER = 2.25
    # TRANSLATION_DELTA_SWITCH = 0.07
    TRANSLATION_DELTA_SWITCH = 0.02
    TRANSLATION_CONSTANT_ADD = 0.2

    ROTATION_LINEAR_MULTIPLIER = 0.20
    # ROTATION_DELTA_SWITCH = np.pi/7.
    ROTATION_DELTA_SWITCH = np.pi / 32.0
    ROTATION_CONSTANT_ADD = 0.01
    ROTATION_MULTIPLIER = 0.07

    # ROBOT_TRANSLATION_COST_MULTIPLIER = 14.5
    # ROBOT_ROTATION_COST_MULTIPLIER = 0.10

    ROBOT_TRANSLATION_COST_MULTIPLIER = 40.0
    ROBOT_ROTATION_COST_MULTIPLIER = 0.05

    # HUBER CACHED CONSTANTS that will be calculated soon
    TRANSLATION_LINEAR_COST_MULT_TOTAL = 0.0
    TRANSLATION_QUADRATIC_COST_MULTPLIER = 0.0
    TRANSLATION_QUADRATIC_COST_MULTPLIER_HALF = 0.0
    TRANSLATION_LINEAR_COST_SUBTRACT = 0.0

    ROTATION_LINEAR_COST_MULT_TOTAL = 0.0
    ROTATION_QUADRATIC_COST_MULTPLIER = 0.0
    ROTATION_QUADRATIC_COST_MULTPLIER_HALF = 0.0
    ROTATION_LINEAR_COST_SUBTRACT = 0.0

    def set_constants(
        self,
        huber_translation_linear_multiplier,
        huber_translation_delta_switch,
        huber_translation_constant_add,
        huber_rotation_linear_multiplier,
        huber_rotation_delta_switch,
        huber_rotation_constant_add,
        huber_rotation_multiplier,
        robot_translation_cost_multiplier=None,
        robot_rotation_cost_multiplier=None,
    ):
        self.TRANSLATION_LINEAR_MULTIPLIER = huber_translation_linear_multiplier
        self.TRANSLATION_DELTA_SWITCH = huber_translation_delta_switch
        self.TRANSLATION_CONSTANT_ADD = huber_translation_constant_add

        self.ROTATION_LINEAR_MULTIPLIER = huber_rotation_linear_multiplier
        self.ROTATION_DELTA_SWITCH = huber_rotation_delta_switch
        self.ROTATION_CONSTANT_ADD = huber_rotation_constant_add
        self.ROTATION_MULTIPLIER = huber_rotation_multiplier

        if robot_translation_cost_multiplier:
            self.ROBOT_TRANSLATION_COST_MULTIPLIER = robot_translation_cost_multiplier
        if robot_rotation_cost_multiplier:
            self.ROBOT_ROTATION_COST_MULTIPLIER = robot_rotation_cost_multiplier

        self.calculate_cached_constants()

    # other constants that are cached for faster computation
    def calculate_cached_constants(self):
        self.TRANSLATION_LINEAR_COST_MULT_TOTAL = (
            self.TRANSLATION_LINEAR_MULTIPLIER + self.TRANSLATION_CONSTANT_ADD
        )
        self.TRANSLATION_QUADRATIC_COST_MULTPLIER = (
            self.TRANSLATION_LINEAR_MULTIPLIER / self.TRANSLATION_DELTA_SWITCH
        )
        self.TRANSLATION_QUADRATIC_COST_MULTPLIER_HALF = (
            0.5 * self.TRANSLATION_QUADRATIC_COST_MULTPLIER
        )
        self.TRANSLATION_LINEAR_COST_SUBTRACT = (
            self.TRANSLATION_LINEAR_MULTIPLIER * self.TRANSLATION_DELTA_SWITCH * 0.5
        )

        self.ROTATION_LINEAR_COST_MULT_TOTAL = (
            self.ROTATION_LINEAR_MULTIPLIER + self.ROTATION_CONSTANT_ADD
        )
        self.ROTATION_QUADRATIC_COST_MULTPLIER = (
            self.ROTATION_LINEAR_MULTIPLIER / self.ROTATION_DELTA_SWITCH
        )
        self.ROTATION_QUADRATIC_COST_MULTPLIER_HALF = (
            0.5 * self.ROTATION_QUADRATIC_COST_MULTPLIER
        )
        self.ROTATION_LINEAR_COST_SUBTRACT = (
            self.ROTATION_LINEAR_MULTIPLIER * self.ROTATION_DELTA_SWITCH * 0.5
        )


def UserInputToRobotAction(user_input):
    return np.append(user_input, np.zeros(3))


def transition_quaternion(quat, angular_vel, action_apply_time):
    norm_vel = np.linalg.norm(angular_vel)
    return transmethods.quaternion_multiply(
        transmethods.quaternion_about_axis(
            action_apply_time * norm_vel, angular_vel / norm_vel
        ),
        quat,
    )
