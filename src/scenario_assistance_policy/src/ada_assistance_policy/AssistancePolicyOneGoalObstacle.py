# WE DON'T EXPECT TO USE THIS CLASS, BUT RATHER ONE THAT INHERITS FROM IT
# Generic assistance policy for one target
import numpy as np

import HuberAssistancePolicyObstacle as TargetPolicy

TargetPolicyClass = TargetPolicy.HuberAssistancePolicyObstacle


class AssistancePolicyOneGoalObstacle:
    def __init__(self, goal, obstacle_pos, obstacle_radius, obstacle_padding):
        self.goal = goal

        self.target_assist_policies = []
        for pose in self.goal.target_poses:
            self.target_assist_policies.append(
                TargetPolicyClass(pose, obstacle_pos, obstacle_radius, obstacle_padding)
            )
        self.min_val_ind = 0

        self.last_robot_state = None
        self.last_user_action = None

    def reset(self):
        self.last_robot_state = None
        self.last_user_action = None
        self.min_val_ind = 0

        for target_policy in self.target_assist_policies:
            target_policy.reset()

    def update(self, robot_state, user_action):
        self.last_robot_state = robot_state
        self.last_user_action = user_action

        for target_policy in self.target_assist_policies:
            target_policy.update(robot_state, user_action)

        values = [targ_policy.get_value() for targ_policy in self.target_assist_policies]
        self.min_val_ind = np.argmin(values)

    def get_value(self):
        return self.target_assist_policies[self.min_val_ind].get_value()

    def get_qvalue(self):
        return self.target_assist_policies[self.min_val_ind].get_qvalue()

    def get_action(self):
        values = [targ_policy.get_value() for targ_policy in self.target_assist_policies]
        min_val_ind = np.argmin(values)
        return self.target_assist_policies[min_val_ind].get_action()

    def get_min_value_pose(self):
        return self.goal.target_poses[self.min_val_ind]
