import time

import numpy as np

from ada_assistance_policy.UserBot import UserBot


class CollabHumanStateMachine:
    """Human state machine for collab tasks. Has states "move to goal", "waiting for
    space", "work on goal", "reset", "done". Execution starts in "move to goal" in which
    the human moves towards the current goal through waypoints or mdp policy. Once the
    goal is reached, the state switches to "waiting for space". If the goal is empty, ada
    handler will call `permit_goal_work`, switching the state to "work on goal". After
    `work_ticks` ticks, the state switches to "done" if that was the last goal, otherwise
    to "reset" after incrementing the current goal. In "reset", human waits for the robot
    to be done with the previous goal and switches back to "move to goal". "done" is the
    terminal state and human doesn't do anything else.

    Args:
        scenario (CollabScenario): Current scenario.
        goals (List): List of goal objects.
        simulate_user (bool): True if human should be simulated.
        work_ticks(int): Number of ticks to work on a goal.
    """

    def __init__(self, scenario, goals, simulate_user, work_ticks):
        self.scenario = scenario
        self.goals = goals
        self.simulate_user = simulate_user
        self.work_ticks = work_ticks

        self.cur_state = None
        self.cur_user_goal = 0
        self.cur_waypoint = 0
        self.cur_work_ticks = 0
        self.reset_cond = True

        if simulate_user:
            self.user_bot = UserBot()
            if hasattr(scenario, "mdp_human") and scenario.mdp_human is not None:
                self.human_mode = "mdp"
            elif hasattr(scenario, "waypoints") and scenario.waypoints is not None:
                self.human_mode = "waypoints"
            else:
                raise ValueError(
                    "Unknown human mode since scenario neither has mdp_human nor "
                    "waypoints."
                )
        else:
            self.user_bot = None
            self.human_mode = None

            time.sleep(10)  # it takes some time to receive tracking info
            print "Ready to start tracking!"
            self.prev_human_ee_trans = None

    def tick(self, human_ee_trans, tracked_human_ee_trans=None):
        """Called once per loop."""
        if self.cur_state is None:
            self.cur_state = "move to goal"
            self.cur_waypoint = 0

        if tracked_human_ee_trans and self.prev_human_ee_trans is None:
            self.prev_human_ee_trans = tracked_human_ee_trans

        if self.cur_state == "move to goal":
            if self.simulate_user:
                user_input_velocity, goal_reached = self._get_sim_velocity(human_ee_trans)
            else:
                user_input_velocity, goal_reached = self._get_real_velocity(
                    tracked_human_ee_trans
                )

            if goal_reached:
                self.cur_state = "waiting for space"
                return user_input_velocity, "waiting: {}".format(self.cur_user_goal)
            else:
                return user_input_velocity, ""
        elif self.cur_state == "waiting for space":
            return np.zeros(3), "waiting: {}".format(self.cur_user_goal)
        elif self.cur_state == "work on goal":
            if self.cur_work_ticks < self.work_ticks:
                self.cur_work_ticks += 1
                return np.zeros(3), ""
            else:
                self.cur_work_ticks = 0
                if self.cur_user_goal == len(self.goals) - 1:
                    self.cur_state = "done"
                    return np.zeros(3), "done: {}".format(self.cur_user_goal)
                else:
                    self.cur_waypoint = 0
                    self.cur_state = "reset"
                    return np.zeros(3), "resetting: {}".format(self.cur_user_goal)
        elif self.cur_state == "reset":
            if self.reset_cond:  # Will become False when ada handler calls stop_reset()
                return np.zeros(3), ""
            else:
                self.cur_user_goal += 1
                self.cur_state = "move to goal"
                self.reset_cond = True
                return np.zeros(3), "moving to goal"
        elif self.cur_state == "done":
            return np.zeros(3), ""
        else:
            raise ValueError("Human in unknown state {}".format(self.cur_state))

    def permit_goal_work(self):
        """Required goal is empty, can start work."""
        self.cur_state = "work on goal"

    def stop_reset(self):
        """Stops reset and moves to the next goal if current state is reset."""
        self.reset_cond = False

    def _get_sim_velocity(self, human_ee_trans):
        """Simulated velocity."""
        if self.human_mode == "waypoints":
            cur_waypoint_list = self.scenario.waypoints[self.cur_user_goal]

            wp_pos = cur_waypoint_list[self.cur_waypoint]
            waypoint_in_world = np.array(
                [
                    [0, 0, 1, wp_pos[0]],
                    [1, 0, 0, wp_pos[1]],
                    [0, 1, 0, wp_pos[2]],
                    [0, 0, 0, 1],
                ]
            )

            user_input_velocity = self.user_bot.get_usr_cmd(
                human_ee_trans, goal_pose=waypoint_in_world
            )

            # Update waypoint to go if it is not the last one
            if self.cur_waypoint != len(cur_waypoint_list) - 1:
                # Go to next WP if too close to the current one
                if np.linalg.norm(user_input_velocity) < 0.005:
                    self.cur_waypoint += 1
        elif self.human_mode == "mdp":
            new_human_pos_xy = self.scenario.mdp_human.get_next_location(
                human_ee_trans[:2, 3], self.cur_user_goal
            )
            # Currently assuming z value doesn't change.
            new_human_trans = np.array(
                [
                    [0, 0, 1, new_human_pos_xy[0]],
                    [1, 0, 0, new_human_pos_xy[1]],
                    [0, 1, 0, self.scenario.human_position[2]],
                    [0, 0, 0, 1],
                ]
            )
            user_input_velocity = self.user_bot.get_usr_cmd(
                human_ee_trans, goal_pose=new_human_trans
            )
        else:  # Should never reach this
            user_input_velocity = np.zeros(3)

        quat_threshold = np.pi / 4
        if self.goals[self.cur_user_goal].at_goal(
            human_ee_trans, quat_threshold=quat_threshold
        ):
            human_goal_reached = True
        else:
            human_goal_reached = False

        return user_input_velocity, human_goal_reached

    def _get_real_velocity(self, tracked_human_ee_trans):
        """Real human velocity."""
        user_input_velocity = tracked_human_ee_trans - self.prev_human_ee_trans

        quat_threshold = np.pi
        if self.goals[self.cur_user_goal].at_goal(
            tracked_human_ee_trans, trans_threshold=0.09, quat_threshold=quat_threshold
        ):
            human_goal_reached = True
        else:
            human_goal_reached = False

        self.prev_human_ee_trans = tracked_human_ee_trans
        return user_input_velocity, human_goal_reached
