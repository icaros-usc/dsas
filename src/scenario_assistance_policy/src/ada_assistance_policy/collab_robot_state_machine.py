import time

import numpy as np

from ada_assistance_policy.UserBot import UserBot


class CollabRobotStateMachine:
    """Robot state machine for collab tasks. Has states "move to goal", "replanning to new
    init", "waiting for space", "work on goal", "reset", "done". Execution starts in "move
    to goal" in which the robot just moves according to the shared autonomy policy action.
    If the robot has self collision or it hits joint limits, ada handler will call
    `replan` to make the robot replan to a new initial state and then continue shared
    autonomy policy. Once the goal is reached, the state switches to "waiting for space".
    If the goal is empty, ada handler will call `permit_goal_work`, switching the state to
    "work on goal". After `work_ticks` ticks, the state switches to "done" if that was the
    last goal, otherwise to "reset" after incrementing the number of goals done. In
    "reset", robot replans to the initial position and switches back to "move to goal".
    "done" is the terminal state and robot doesn't do anything else.

    Robot stops after reaching the goal once before human starts moving again. This is
    just a hack to maintain synchrony. It is implemented through a variable `can_move`
    that is initially True but becomes False after working on a goal. It becomes True
    again once the human starts to move towards the next goal. The case with robot being
    too fast is handled by this mechanism since the human has at least 4 seconds (2
    seconds of robot working on the goal and 2 seconds of reset) to reach the intended
    goal from the start and set `robot_inference_on` to False, which stops robot's motion
    once it goes back to "move to goal" state. The case with human being too fast is
    handled by the work and reset times for the human (5, 15 seconds respectively) during
    which the robot can't get confused and just needs to move to the correct goal.

    Args:
        robot_policy (AdaAssistancePolicy): Robot policy. The same object is used by both
            ada handler and robot state machine. Not good coding but whatever.
        total_goals (int): Number of goal objects.
        work_ticks(int): Number of ticks to work on a goal.
        reset_ticks(int): Number of ticks to wait during reset.
    """

    def __init__(self, robot_policy, total_goals, work_ticks, reset_ticks):
        self.robot_policy = robot_policy
        self.total_goals = total_goals
        self.work_ticks = work_ticks
        self.reset_ticks = reset_ticks

        self.cur_state = None
        self.goals_done = 0
        self.cur_work_ticks = 0
        self.cur_reset_ticks = 0
        self.robot_at_goal = None
        self.can_move = True

    def tick(self, robot_inference_on):
        """Called once per loop."""
        if self.cur_state is None:
            self.cur_state = "move to goal"

        if self.cur_state == "move to goal":
            if self.can_move:
                action = self.robot_policy.get_action(fix_magnitude_user_command=False)
                self.robot_at_goal = self.robot_policy.check_at_goal()

                if self.robot_at_goal is not None:
                    self.cur_state = "waiting for space"
                    return None, "waiting: {}".format(self.robot_at_goal)
                else:
                    return action, ""
            else:
                # can_move will be true once the human resets and starts moving again.
                self.can_move = robot_inference_on
                self.robot_policy.reset()
                return None, ""
        elif self.cur_state == "replanning to new init":
            if self.cur_reset_ticks < self.reset_ticks:
                self.cur_reset_ticks += 1
                return None, ""
            else:
                self.cur_reset_ticks = 0
                self.cur_state = "move to goal"
                return None, "moving to goal"
        elif self.cur_state == "waiting for space":
            return None, "waiting: {}".format(self.robot_at_goal)
        elif self.cur_state == "work on goal":
            if self.cur_work_ticks < self.work_ticks:
                self.cur_work_ticks += 1
                return None, ""
            else:
                self.goals_done += 1
                self.robot_policy.reset()
                self.cur_work_ticks = 0
                if self.goals_done == self.total_goals:
                    self.cur_state = "done"
                    return None, "done: {}".format(self.robot_at_goal)
                else:
                    self.cur_state = "reset"
                    return None, "resetting: {}".format(self.robot_at_goal)
        elif self.cur_state == "reset":
            self.robot_at_goal = None
            self.robot_policy.reset()
            if self.cur_reset_ticks < self.reset_ticks:
                self.cur_reset_ticks += 1
                return None, ""
            else:
                self.can_move = False
                self.cur_reset_ticks = 0
                self.cur_state = "move to goal"
                return None, "moving to goal"
        elif self.cur_state == "done":
            return None, ""
        else:
            raise ValueError("Robot in unknown state {}".format(self.cur_state))

    def permit_goal_work(self):
        """Required goal is empty, can start work."""
        self.cur_state = "work on goal"

    def real_reset_done(self):
        """Real robot planning to config is done, so the state can be changed."""
        self.cur_reset_ticks = self.reset_ticks

    def replan(self):
        """Called by ada handler when robot gets stuck and needs to replan to start."""
        self.cur_state = "replanning to new init"
