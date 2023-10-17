# Generic assistance policy for one goal
import numpy as np

from ada_assistance_policy import HuberAssistancePolicyObstacle, HuberAssistancePolicy
from ada_assistance_policy.AssistancePolicyOneGoal import AssistancePolicyOneGoal
from ada_assistance_policy.AssistancePolicyOneGoalObstacle import (
    AssistancePolicyOneGoalObstacle,
)


class AssistancePolicy:
    def __init__(
        self,
        goals,
        has_obstacle=False,
        knows_obstacle=False,
        obstacle_pos=-1,
        obstacle_radius=-1,
        obstacle_padding=-1,
        collab=False,
    ):
        self.goals = goals
        self.visited_goals_idx = []
        self.goal_mapping = np.zeros(3, dtype=int) - 1
        self.cur_goal_dist = None

        self.goal_assist_policies = []
        for goal in goals:
            if knows_obstacle:
                if collab:
                    raise NotImplementedError(
                        "Obstacle support not implemented for collab task"
                    )
                self.goal_assist_policies.append(
                    AssistancePolicyOneGoalObstacle(
                        goal, obstacle_pos, obstacle_radius, obstacle_padding
                    )
                )
            else:
                self.goal_assist_policies.append(AssistancePolicyOneGoal(goal, collab))

        self.has_obstacle = has_obstacle
        self.knows_obstacle = knows_obstacle
        self.obstacle_pos = obstacle_pos
        self.obstacle_radius = obstacle_radius
        self.obstacle_padding = obstacle_padding
        self.collab = collab

        self.robot_state = None
        self.user_action = None

    # NEW
    def clone(self):
        return AssistancePolicy(
            list(self.goals),
            self.has_obstacle,
            self.knows_obstacle,
            self.obstacle_pos,
            self.obstacle_radius,
            self.obstacle_padding,
            self.collab,
        )

    def reset(self):
        self.robot_state = None
        self.user_action = None
        self.goal_mapping = np.zeros(3, dtype=int) - 1
        self.cur_goal_dist = None

        for goal_policy in self.goal_assist_policies:
            goal_policy.reset()

    def update(self, robot_state, user_action, human_ee_trans=None):
        self.robot_state = robot_state
        # user action corresponds to the effect of direct teleoperation on the robot
        self.user_action = user_action

        for goal_policy in self.goal_assist_policies:
            goal_policy.update(robot_state, self.user_action, human_ee_trans)

        if self.collab:
            for i in range(len(self.goals)):
                # Valid robot goals for each goal are those that are different and not
                # visited
                valid_goal_idx = []
                for j in range(len(self.goals)):
                    if i != j and j not in self.visited_goals_idx:
                        valid_goal_idx.append(j)

                if not valid_goal_idx:
                    if i not in self.visited_goals_idx:
                        # Move to the same goal as the human if it is the only option
                        self.goal_mapping[i] = i
                    else:
                        # Don't move if there are no valid goals and the current human
                        # goal has been visited already (should not happen in practice
                        # since it means the robot has reached all the goals and the
                        # scenario should end)
                        self.goal_mapping[i] = -1
                else:
                    # Select the index of the closest valid goal
                    sorted_valid_goal_idx = sorted(
                        valid_goal_idx,
                        key=lambda gi: self.goal_assist_policies[gi].get_value(),
                    )
                    self.goal_mapping[i] = sorted_valid_goal_idx[0]

    def check_at_goal(self):
        at_goal = None
        quat_threshold = np.pi / 4 if self.collab else np.pi / 48
        for i, g in enumerate(self.goals):
            if g.at_goal(self.robot_state.ee_trans, quat_threshold=quat_threshold):
                at_goal = i
                break

        if not self.collab or self.goal_mapping[np.argmax(self.cur_goal_dist)] == at_goal:
            self.visited_goals_idx.append(at_goal)
            return at_goal
        else:
            return None

    def get_values(self):
        values = np.ndarray(len(self.goal_assist_policies))
        qvalues = np.ndarray(len(self.goal_assist_policies))
        for ind, goal_policy in enumerate(self.goal_assist_policies):
            values[ind] = goal_policy.get_value()
            qvalues[ind] = goal_policy.get_qvalue()

        return values, qvalues

    def get_human_values(self):
        if not self.collab:
            raise ValueError("Human error only applicable to collab tasks.")
        values = np.ndarray(len(self.goal_assist_policies))
        qvalues = np.ndarray(len(self.goal_assist_policies))
        for ind, goal_policy in enumerate(self.goal_assist_policies):
            values[ind] = goal_policy.get_human_value()
            qvalues[ind] = goal_policy.get_human_qvalue()

        return values, qvalues

    def get_probs_last_user_action(self):
        values, qvalues = self.get_values()
        return np.exp(-(qvalues - values))

    def get_assisted_action(self, goal_distribution, fix_magnitude_user_command=False):
        assert goal_distribution.size == len(self.goal_assist_policies)
        self.cur_goal_dist = goal_distribution

        if self.knows_obstacle:
            if self.collab:
                raise ValueError("Obstacle support not implemented for collab task")
            action_dimension = HuberAssistancePolicyObstacle.ACTION_DIMENSION

            # TODO how do we handle mode switch vs. not?
            total_action_twist = np.zeros(action_dimension)
            for goal_policy, goal_prob in zip(
                self.goal_assist_policies, goal_distribution
            ):
                total_action_twist += goal_prob * goal_policy.get_action()

            total_action_twist /= np.sum(goal_distribution)

            robot_pos = self.robot_state.ee_trans[0:3, 3]
            dist = np.linalg.norm(robot_pos - self.obstacle_pos)
            if dist <= self.obstacle_radius + self.obstacle_padding + 0.02:
                # collision with sphere:
                to_ret_twist = total_action_twist
            else:
                to_ret_twist = total_action_twist + self.user_action.twist
        else:
            action_dimension = HuberAssistancePolicy.ACTION_DIMENSION
            total_action_twist = np.zeros(action_dimension)

            if self.collab:
                goal_policy_list = []
                for gi in self.goal_mapping:
                    if gi == -1:
                        goal_policy_list.append(None)
                    else:
                        goal_policy_list.append(self.goal_assist_policies[gi])
            else:
                goal_policy_list = self.goal_assist_policies

            for goal_policy, goal_prob in zip(goal_policy_list, goal_distribution):
                if goal_policy is not None:
                    total_action_twist += goal_prob * goal_policy.get_action()

            total_action_twist /= np.sum(goal_distribution)

            if self.collab:
                to_ret_twist = total_action_twist
            else:
                to_ret_twist = total_action_twist + self.user_action.twist

        if fix_magnitude_user_command:
            if self.collab:
                raise ValueError(
                    "Fix magnitude user command doesn't make sense for collab tasks"
                )
            to_ret_twist *= np.linalg.norm(self.user_action.twist) / np.linalg.norm(
                to_ret_twist
            )

        return to_ret_twist
