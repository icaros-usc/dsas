# Handles converting openrave items to generic assistance policy
import math

import numpy as np
import openravepy

import GoalPredictor as GoalPredictor
from ada_assistance_policy.AssistancePolicy import AssistancePolicy
from ada_assistance_policy.Goal import Goal
from ada_assistance_policy.OpenraveUtils import (
    GetTSRListForObject,
    SampleTSRList,
    get_bowl_poses,
)
from ada_teleoperation.RobotState import Action

ADD_MORE_IK_SOLS = False


class AdaAssistancePolicy:
    def __init__(
        self,
        goals,
        has_obstacle,
        knows_obstacle,
        obstacle_pos,
        obstacle_radius,
        obstacle_padding,
        collab,
    ):
        self.assist_policy = AssistancePolicy(
            goals,
            has_obstacle,
            knows_obstacle,
            obstacle_pos,
            obstacle_radius,
            obstacle_padding,
            collab,
        )
        self.goal_predictor = GoalPredictor.GoalPredictor(goals)

        self.has_obstacle = has_obstacle
        self.knows_obstacle = knows_obstacle
        self.obstacle_pos = obstacle_pos
        self.obstacle_radius = obstacle_radius
        self.obstacle_padding = obstacle_padding
        self.collab = collab

        self.goals = goals
        self.robot_state = None

    def clone(self):
        ret = AdaAssistancePolicy(
            list(self.goals),
            self.has_obstacle,
            self.knows_obstacle,
            self.obstacle_pos,
            self.obstacle_radius,
            self.obstacle_padding,
            self.collab,
        )
        ret.assist_policy = self.assist_policy.clone()
        return ret

    def reset(self):
        self.assist_policy.reset()
        self.goal_predictor.reset()
        self.robot_state = None

    def update(
        self,
        robot_state,
        user_action,
        human_ee_trans=None,
        mdp_human=None,
        cur_user_goal=None,
    ):
        """Update values/goal distribution and do any other required calculations.
        Args:
            robot_state: State of the robot.
            user_action: User's action (teleop action equivalent).
            human_ee_trans: End effector transform of the human (only relevant for collab
                tasks).
            mdp_human: MDP for human policy (only relevant when human is
                controlled/modelled by an MDP)
            cur_user_goal: Current goal of the human (only required since mdp_human is
                stateless and needs it for Q-value calculation)
        """
        self.assist_policy.update(robot_state, user_action, human_ee_trans)
        if self.collab:
            if mdp_human is not None:  # MDP model for human (never called for now)
                # Negative since goal belief expects costs instead of rewards.
                values = -mdp_human.get_values(human_ee_trans[:2, 3])
                q_values = -mdp_human.get_q_values(human_ee_trans[:2, 3], cur_user_goal)
            else:  # Straight line model for human
                values, q_values = self.assist_policy.get_human_values()
        else:
            values, q_values = self.assist_policy.get_values()

        # For MDP, values/q_values can be none when the human is waiting at the goal for
        # the robot to finish. So just don't update the goal distribution.
        if not np.any(np.isnan(values) | np.isnan(q_values)):
            self.goal_predictor.update_distribution(values, q_values)
        self.robot_state = robot_state

    def check_at_goal(self):
        return self.assist_policy.check_at_goal()

    def get_action(self, goal_distribution=np.array([]), **kwargs):
        if goal_distribution.size == 0:
            goal_distribution = self.goal_predictor.get_distribution()

        assisted_action = Action(
            twist=self.assist_policy.get_assisted_action(goal_distribution, **kwargs),
            finger_vel=self.assist_policy.user_action.finger_vel,
            switch_mode_to=self.assist_policy.user_action.switch_mode_to,
        )

        return assisted_action

    def get_blend_action(
        self, blend_level="aggressive", goal_distribution=np.array([]), **kwargs
    ):
        if self.collab:
            raise NotImplementedError("Blend action not implemented for collab tasks")

        if goal_distribution.size == 0:
            goal_distribution = self.goal_predictor.get_distribution()

        max_prob_goal_ind = np.argmax(goal_distribution)

        # check if we meet the confidence criteria which dictates whether or not
        # assistance is provided
        # use the one from ancas paper - euclidean distance and some threshold
        if blend_level == "aggressive":
            if blend_confidence_function_prob_diff(goal_distribution):
                # print("Blending!")
                goal_distribution_all_max = np.zeros(len(goal_distribution))
                goal_distribution_all_max[max_prob_goal_ind] = 1.0
                assisted_action = Action(
                    twist=self.assist_policy.get_assisted_action(
                        goal_distribution_all_max, **kwargs
                    ),
                    finger_vel=self.assist_policy.user_action.finger_vel,
                    switch_mode_to=self.assist_policy.user_action.switch_mode_to,
                )
                return assisted_action
            else:
                # if we don't meet confidence function, use direct teleop
                return self.assist_policy.user_action
        else:
            if blend_confidence_function_prob_diff(goal_distribution):
                if len(goal_distribution) <= 1:
                    conf = 1.0
                else:
                    goal_distribution_sorted = np.sort(goal_distribution)
                    conf = goal_distribution_sorted[-1] - goal_distribution_sorted[-2]

                if conf <= 0.8:
                    arbitration = 1.5 * (conf - 0.4)
                else:
                    arbitration = 0.6

                goal_distribution_all_max = np.zeros(len(goal_distribution))
                goal_distribution_all_max[max_prob_goal_ind] = 1.0
                user_action = self.assist_policy.user_action
                assisted_action = Action(
                    twist=self.assist_policy.get_assisted_action(
                        goal_distribution_all_max, **kwargs
                    ),
                    finger_vel=self.assist_policy.user_action.finger_vel,
                    switch_mode_to=self.assist_policy.user_action.switch_mode_to,
                )

                blended_action_twist = assisted_action.twist
                blended_action_twist[:2] = (
                    arbitration * assisted_action.twist[:2]
                    + (1 - arbitration) * user_action.twist[:2]
                )
                # fix going down
                blended_action = Action(twist=blended_action_twist)
                return blended_action
            else:
                # if we don't meet confidence function, use direct teleop
                return self.assist_policy.user_action


def blend_confidence_function_prob_diff(goal_distribution, prob_diff_required=0.4):
    if len(goal_distribution) <= 1:
        return True

    goal_distribution_sorted = np.sort(goal_distribution)
    return (
        goal_distribution_sorted[-1] - goal_distribution_sorted[-2] > prob_diff_required
    )


def blend_confidence_function_euclidean_distance(robot_state, goal, distance_thresh=0.10):
    manip_pos = robot_state.get_pos()
    goal_poses = goal.target_poses
    goal_pose_distances = [
        np.linalg.norm(manip_pos - pose[0:3, 3]) for pose in goal_poses
    ]
    dist_to_nearest_goal = np.min(goal_pose_distances)
    return dist_to_nearest_goal < distance_thresh


# generic functions
def goal_from_object(obj, manip):
    pose = obj.GetTransform()
    robot = manip.GetRobot()
    env = robot.GetEnv()

    # generate TSRs for object
    if "bowl" not in obj.GetName():
        target_tsrs = GetTSRListForObject(obj, manip)

    # turn TSR into poses
    num_poses_desired = 30
    max_num_poses_sample = 500

    target_poses = []
    target_iks = []
    num_sampled = 0
    while len(target_poses) < num_poses_desired and num_sampled < max_num_poses_sample:
        print "name: " + obj.GetName() + " currently has " + str(
            len(target_poses)
        ) + " goal poses"
        if "bowl" not in obj.GetName():
            num_sample_this = int(math.ceil(num_poses_desired / len(target_tsrs)))
            num_sampled += num_sample_this
            target_poses_idenframe = SampleTSRList(target_tsrs, num_sample_this)
            target_poses_tocheck = [
                np.dot(obj.GetTransform(), pose) for pose in target_poses_idenframe
            ]
        else:
            num_sample_this = num_poses_desired
            num_sampled += num_sample_this
            target_poses_tocheck = get_bowl_poses(
                obj, num_samples_pose=num_sample_this, ee_offset=0.15
            )
        for pose in target_poses_tocheck:

            ik_sol = manip.FindIKSolution(
                pose, openravepy.IkFilterOptions.CheckEnvCollisions
            )
            if ik_sol is not None:
                if ADD_MORE_IK_SOLS:
                    # get bigger list of ik solutions
                    ik_sols = manip.FindIKSolutions(
                        pose, openravepy.IkFilterOptions.CheckEnvCollisions
                    )
                    if ik_sols is None:
                        ik_sols = list()
                    else:
                        ik_sols = list(ik_sols)
                    # add the solution we found before
                    ik_sols.append(ik_sol)
                else:
                    # if we don't want to add more, just use the one we found
                    ik_sols = [ik_sol]
                # check env col
                target_poses.append(pose)
                target_iks.append(ik_sols)
                if len(target_poses) >= num_poses_desired:
                    break

    return Goal(pose, target_poses, target_iks)
