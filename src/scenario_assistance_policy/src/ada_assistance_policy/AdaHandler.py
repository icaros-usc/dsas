# The basic ADA interface, such as executing end effector motions, getting the state of
# the robot, etc.
import copy
import logging
import os
import time
from tf2_msgs.msg import TFMessage
import pdb

import numpy as np
import rospy
from catkin.find_in_workspaces import find_in_workspaces

import AssistancePolicyVisualizationTools as vistools
from ada_assistance_policy.AdaAssistancePolicy import AdaAssistancePolicy
from ada_assistance_policy.UserBot import UserBot
from ada_assistance_policy.collab_human_state_machine import CollabHumanStateMachine
from ada_assistance_policy.collab_robot_state_machine import CollabRobotStateMachine
from ada_teleoperation.AdaTeleopHandler import AdaTeleopHandler, is_done_func_button_hold
from ada_teleoperation.input_handlers.UserInputListener import UserInputData
from prpy.planning import PlanningError
from simple_environment.scenarios import CollabScenario

SIMULATE_DEFAULT = False
# SIMULATE_VELOCITY_MOVEJOINT = False  #NOTE if true, SIMULATE should also be true
# FLOATING_HAND_ONLY = False
RESAVE_GRASP_POSES = True
cached_data_dir = "cached_data"
CONTROL_HZ = 40.0
num_control_modes = 2

project_name = "simple_environment"
logger = logging.getLogger(project_name)


class AdaHandler:
    def __init__(
        self,
        env,
        robot,
        goals,
        goal_objects,
        input_interface_name,
        num_input_dofs,
        use_finger_mode=True,
        goal_object_poses=None,
        scenario=None,
    ):
        self.env = env
        self.robot = robot
        self.scenario = scenario
        self.collab = isinstance(scenario, CollabScenario)
        self.goals = goals
        self.goal_objects = goal_objects
        if not goal_object_poses and goal_objects:
            self.goal_object_poses = [
                goal_obj.GetTransform() for goal_obj in goal_objects
            ]
        else:
            self.goal_object_poses = goal_object_poses

        self.sim = robot.simulated
        self.manip = self.robot.arm

        self.ada_teleop = AdaTeleopHandler(
            env, robot, input_interface_name, num_input_dofs, use_finger_mode
        )  # , is_done_func=Teleop_Done)
        self.robot_state = self.ada_teleop.robot_state

        self.robot_policy = AdaAssistancePolicy(
            self.goals,
            has_obstacle=scenario.has_obstacle,
            knows_obstacle=scenario.knows_obstacle,
            obstacle_pos=scenario.obstacle_pos,
            obstacle_radius=scenario.obstacle_radius,
            obstacle_padding=scenario.obstacle_padding,
            collab=self.collab,
        )

        self.init_robot_arm_dof = self.robot.arm.GetDOFValues()
        self.init_robot_dof = self.robot.GetDOFValues()

        self.user_input_mapper = self.ada_teleop.user_input_mapper
        self.user_bot = None
        self.human = self.env.GetKinBody("human_sphere")
        self.cur_user_goal = None

        if self.collab:
            self.sub_hand = rospy.Subscriber(
                "tf", TFMessage, self.tfcallback, queue_size=10
            )
            self.tracked_human_ee_trans = None

            self.collab_human_state_machine = None
            self.collab_robot_state_machine = None

            start_human_pos = self.scenario.human_position
            self.start_human_trans = np.array(
                [
                    [0, 0, 1, start_human_pos[0]],
                    [1, 0, 0, start_human_pos[1]],
                    [0, 1, 0, start_human_pos[2]],
                    [0, 0, 0, 1],
                ]
            )
            self.robot_inference_on = True
            self.human_done = False
            self.robot_done = False
            self.goal_block = [False] * len(self.goals)

            if hasattr(scenario, "human_vel_coeff"):
                self.human_vel_coeff = scenario.human_vel_coeff
            else:
                self.human_vel_coeff = 0.5

    def tfcallback(self, data):

        # del_x/y/z is the transformation along each axis
        # del_x/y/z = robot_transform + camera_transform + adjustments
        del_x = 0.409 + 0.48
        del_y = 0.338 + 1.48
        del_z = 0.795 + 0.62 + 0.12

        for dtf in data.transforms:
            if "left_hand" in dtf.child_frame_id:
                human_ee_pos = np.array(
                    [
                        dtf.transform.translation.y + del_x,
                        -dtf.transform.translation.x + del_y,
                        dtf.transform.translation.z + del_z,
                    ]
                )

                self.tracked_human_ee_trans = np.array(
                    [
                        [0, 0, 1, human_ee_pos[0]],
                        [1, 0, 0, human_ee_pos[1]],
                        [0, 1, 0, human_ee_pos[2]],
                        [0, 0, 0, 1],
                    ]
                )

    def GetEndEffectorTransform(self):
        return self.manip.GetEndEffectorTransform()

    def remove_all_trajspos(self, traj_pos_pre="traj_pos"):
        ind = 0
        end_ind = 10000
        traj_pos_name = traj_pos_pre + str(ind)
        morsel_body = self.env.GetKinBody(traj_pos_name)
        while morsel_body or ind < end_ind:
            if morsel_body:
                self.env.Remove(morsel_body)
            ind += 1
            traj_pos_name = traj_pos_pre + str(ind)
            morsel_body = self.env.GetKinBody(traj_pos_name)

    def plot_trajectory(self, end_effector_trans, counter, traj_pos_pre="traj_pos"):
        traj_pos_name = traj_pos_pre + str(counter)
        if self.env.GetKinBody(traj_pos_name) is None:
            object_base_path = find_in_workspaces(
                search_dirs=["share"],
                project="simple_environment",
                path="data",
                first_match_only=True,
            )[0]
            obj_name = (
                "trajpossphere2.kinbody.xml"
                if "robot" in traj_pos_pre
                else "trajpossphere.kinbody.xml"
            )
            ball_path = os.path.join(object_base_path, "objects", obj_name)
            with self.env:
                traj_pos = self.env.ReadKinBodyURI(ball_path)
                traj_pos.SetName(traj_pos_name)
                self.env.Add(traj_pos)
                traj_pos.Enable(False)
        else:
            traj_pos = self.env.GetKinBody(traj_pos_name)
        traj_pos.SetTransform(end_effector_trans)

    def check_collision(self, scenario, ee_trans):
        robot_pos = ee_trans[0:3, 3]
        dist = np.linalg.norm(robot_pos - scenario.obstacle_pos)
        if dist < scenario.obstacle_radius:
            print "Robot collision!!!!!!!!!"
            return True
        else:
            return False

    def old_execute_policy(
        self,
        scenario,
        simulate_user=False,
        direct_teleop_only=False,
        blend_level="aggressive",
        blend_only=False,
        fix_magnitude_user_command=False,
        is_done_func=is_done_func_button_hold,
        finish_trial_func=None,
        traj_data_recording=None,
        plot_trajectory=False,
    ):
        if hasattr(scenario, "mdp_human") and scenario.mdp_human is not None:
            human_mode = "mdp"
        elif hasattr(scenario, "waypoints") and scenario.waypoints is not None:
            human_mode = "waypoints"
        else:
            print "Using default human policy of going straight to goal without waypoints"
            human_mode = None

        if simulate_user:
            self.user_bot = UserBot()
        self.cur_user_goal = 0

        if not simulate_user:
            time.sleep(10)  # it takes some time to receive tracking info
            print "Ready to start tracking!"
            prev_human_ee = copy.deepcopy(self.tracked_human_ee_trans)

        vis = vistools.VisualizationHandler()

        robot_state = self.robot_state
        robot_state.ee_trans = self.GetEndEffectorTransform()

        time_per_iter = 1.0 / CONTROL_HZ

        if direct_teleop_only:
            use_assistance = False
        else:
            if simulate_user or self.collab:
                use_assistance = True
            else:
                use_assistance = False

        # set the huber constants differently if the robot movement magnitude is fixed to
        # user input magnitude
        if not direct_teleop_only and fix_magnitude_user_command:
            for goal_policy in self.robot_policy.assist_policy.goal_assist_policies:
                for target_policy in goal_policy.target_assist_policies:
                    target_policy.set_constants(
                        huber_translation_linear_multiplier=1.55,
                        huber_translation_delta_switch=0.11,
                        huber_translation_constant_add=0.2,
                        huber_rotation_linear_multiplier=0.20,
                        huber_rotation_delta_switch=np.pi / 72.0,
                        huber_rotation_constant_add=0.3,
                        huber_rotation_multiplier=0.20,
                        robot_translation_cost_multiplier=14.0,
                        robot_rotation_cost_multiplier=0.05,
                    )

        # if specified traj data for recording, initialize
        if traj_data_recording:
            assist_type = "shared_auton"
            if direct_teleop_only:
                assist_type = "None"
            elif blend_only:
                assist_type = "blend"
            elif fix_magnitude_user_command:
                assist_type = "shared_auton_prop"

            traj_data_recording.set_init_info(
                start_state=copy.deepcopy(robot_state),
                goals=copy.deepcopy(self.goals),
                input_interface_name=self.ada_teleop.teleop_interface,
                assist_type=assist_type,
            )

        start_time = time.time()
        itrs_taken = 0
        delay_time = 0

        robot_positions = []
        self.remove_all_trajspos()
        counter = 0
        human_goal_reached = False
        robot_goal_reached = False
        cur_waypoint = 0
        while True:
            itrs_taken += 1
            itr_start_time = time.time()
            # if time.time() - start_time > scenario.max_time:
            if itrs_taken * time_per_iter > scenario.max_time:
                scenario.setTime(scenario.max_time)
                scenario.setRobotPositions(robot_positions)
                break

            robot_state.ee_trans = self.GetEndEffectorTransform()
            ee_trans = robot_state.ee_trans
            robot_dof_values = self.robot.GetDOFValues()

            if self.collab:
                # Grow the trajectory
                scenario.add_robot_pos(
                    itrs_taken * time_per_iter, list(ee_trans[:3, 3]),
                )

                if self.human is None:
                    raise ValueError("Human is not set for collaborative task")
                human_ee_trans = self.human.GetTransform()

                scenario.add_human_pos(
                    itrs_taken * time_per_iter, list(human_ee_trans[:3, 3]),
                )
            else:
                # Grow the trajectory
                scenario.add_robot_pos(
                    itrs_taken * time_per_iter, list(ee_trans[:3, 3]),
                )

                human_ee_trans = ee_trans

            if scenario.has_obstacle:
                if self.check_collision(scenario, ee_trans):
                    # scenario.setTime(time.time() - start_time)
                    scenario.setTime(itrs_taken * time_per_iter)

                    print "collision!!!!"
                    scenario.setCollided()

            if simulate_user:
                if not human_goal_reached:
                    if human_mode == "waypoints":
                        if self.collab:
                            cur_waypoint_list = scenario.waypoints[self.cur_user_goal]
                        else:
                            cur_waypoint_list = scenario.waypoints

                        wp_pos = cur_waypoint_list[cur_waypoint]
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
                        if cur_waypoint != len(cur_waypoint_list) - 1:
                            if self.collab:
                                # Go to next WP if too close to the current one
                                if np.linalg.norm(user_input_velocity) < 0.005:
                                    cur_waypoint += 1
                            else:
                                # In shared autonomy, the robot could go around the
                                # waypoint, so increment the waypoint number if robot
                                # crosses the current waypoint in the x-axis. This is a
                                # hack that was done before and ideally should be replaced
                                # by a human policy since human trajectory computed
                                # offline doesn't make sense for shared autonomy.
                                if (
                                    wp_pos[0] < human_ee_trans[0, 3]
                                    or np.linalg.norm(user_input_velocity) < 0.005
                                ):
                                    cur_waypoint += 1
                    elif human_mode == "mdp":
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
                    else:
                        min_val_target_pose = self.robot_policy.assist_policy.goal_assist_policies[
                            self.cur_user_goal
                        ].get_min_value_pose()
                        user_input_velocity = self.user_bot.get_usr_cmd(
                            human_ee_trans, goal_pose=min_val_target_pose
                        )

                    user_input_all = UserInputData(
                        user_input_velocity, button_changes=[0, 0], buttons_held=[0, 0]
                    )

                    quat_threshold = np.pi / 4 if self.collab else np.pi / 48
                    if self.goals[self.cur_user_goal].at_goal(
                        human_ee_trans, quat_threshold=quat_threshold
                    ):
                        human_goal_reached = True  # Wait for robot to reach the goal
                    # else:
                    #     user_input_all.close_hand_velocity = 0.0
                else:
                    # Zero velocity if human_goal_reached
                    user_input_velocity = np.zeros(3)
                    user_input_all = UserInputData(
                        user_input_velocity, button_changes=[0, 0], buttons_held=[0, 0]
                    )
            else:
                if self.collab:
                    curr_human_ee = copy.deepcopy(self.tracked_human_ee_trans)
                    user_input_velocity = curr_human_ee[:3, 3] - prev_human_ee[:3, 3]
                    user_input_all = UserInputData(
                        user_input_velocity, button_changes=[0, 0], buttons_held=[0, 0]
                    )
                    prev_human_ee = curr_human_ee
                else:
                    user_input_velocity = np.zeros(3)
                    user_input_all = (
                        self.ada_teleop.joystick_listener.get_most_recent_cmd()
                    )

                quat_threshold = np.pi if self.collab else np.pi / 12
                if self.goals[self.cur_user_goal].at_goal(
                    human_ee_trans, trans_threshold=0.09, quat_threshold=quat_threshold
                ):
                    human_goal_reached = True  # Wait for robot to reach the goal

            direct_teleop_action = self.user_input_mapper.input_to_action(
                user_input_all, robot_state
            )

            # if left trigger not being hit, then execute with assistance
            if not direct_teleop_only and user_input_all.button_changes[1] == 1:
                use_assistance = not use_assistance

            # if human_mode == "mdp":
            #     self.robot_policy.update(
            #         robot_state,
            #         direct_teleop_action,
            #         human_ee_trans,
            #         mdp_human=self.scenario.mdp_human,
            #         cur_user_goal=self.cur_user_goal,
            #     )
            # else:
            #     self.robot_policy.update(
            #         robot_state, direct_teleop_action, human_ee_trans
            #     )

            self.robot_policy.update(robot_state, direct_teleop_action, human_ee_trans)
            if self.collab:
                self.scenario.add_wrong_goal_prob(
                    self.robot_policy.goal_predictor.get_distribution(),
                    self.cur_user_goal,
                )

            if use_assistance and not direct_teleop_only:
                if blend_only:
                    if self.collab:
                        raise NotImplementedError(
                            "Blend not implemented for collaborative tasks"
                        )
                    action = self.robot_policy.get_blend_action(blend_level)
                else:
                    action = self.robot_policy.get_action(
                        fix_magnitude_user_command=fix_magnitude_user_command
                    )
            else:
                if self.collab:
                    raise NotImplementedError(
                        "Direct teleop not implemented for collaborative tasks"
                    )
                # if left trigger is being hit, direct teleop
                action = direct_teleop_action

            if not robot_goal_reached:  # Only move if not waiting for the human
                # Check if joint limits are reached while executing action so that we can
                # replan to a different start and continue with shared autonomy policy.
                joint_limit_reached = self.ada_teleop.execute_action(action)

                if self.collab and (
                    joint_limit_reached or self.robot.CheckSelfCollision()
                ):
                    # Rotate the starting position and plan there
                    # self.init_robot_arm_dof[0] += np.pi
                    # self.init_robot_dof[0] += np.pi
                    rotate_by = np.pi if self.init_robot_arm_dof[0] > -np.pi else -np.pi
                    self.init_robot_arm_dof[0] -= rotate_by
                    self.init_robot_dof[0] -= rotate_by
                    if self.robot.simulated:
                        self.robot.arm.SetDOFValues(self.init_robot_arm_dof)
                        self.robot.SetDOFValues(self.init_robot_dof)
                        delay_time += 3
                    else:
                        # self.robot.arm.hand.OpenHand()
                        self.robot.arm.PlanToConfiguration(
                            self.init_robot_arm_dof, execute=True
                        )
                        self.robot.SwitchToTeleopController()

            if self.collab:  # Robot goal checks
                robot_at_goal = self.robot_policy.check_at_goal()
                if robot_at_goal is not None:
                    # Robot thinks it has reached the correct goal. Check if it is
                    # actually the correct goal.
                    if robot_at_goal == self.cur_user_goal:
                        # Happens when the same goal is left for both human and robot or
                        # if the robot incorrectly infers the human goal. In this case, an
                        # additional delay is added to simulate either robot/human waiting
                        # for the other to finish their job at the goal and then letting
                        # the other do their job.
                        delay_time += 5
                    robot_goal_reached = True
                    # user_input_all.close_hand_velocity = 0.0

                    # Code below is just for debugging with one-shot scenarios.
                    if self.scenario.one_shot:
                        # sim_time = time.time() - start_time
                        sim_time = itrs_taken * time_per_iter
                        scenario_time = np.clip(
                            sim_time + delay_time, 0, scenario.max_time
                        )
                        scenario.setTime(scenario_time)
                        scenario.setRobotPositions(robot_positions)
                        break

            if not self.collab or self.cur_user_goal == len(self.goals) - 1:
                # In collab task, both human and robot should have finished the goals
                # In shared autonomy, robot goal and human goal should match.
                robot_at_goal = self.robot_policy.check_at_goal()

                if self.collab:
                    goal_cond = robot_goal_reached and human_goal_reached
                else:
                    goal_cond = robot_at_goal == self.cur_user_goal

                if goal_cond:
                    # user_input_all.close_hand_velocity = 1.0
                    # sim_time = time.time() - start_time
                    sim_time = itrs_taken * time_per_iter
                    scenario_time = np.clip(sim_time + delay_time, 0, scenario.max_time)
                    scenario.setTime(scenario_time)
                    scenario.setRobotPositions(robot_positions)
                    break

            # Move human if collab task
            if robot_goal_reached and human_goal_reached:  # Reached goal, reset positions
                start_human_pos = self.scenario.human_position
                start_human_trans = np.array(
                    [
                        [0, 0, 1, start_human_pos[0]],
                        [1, 0, 0, start_human_pos[1]],
                        [0, 1, 0, start_human_pos[2]],
                        [0, 0, 0, 1],
                    ]
                )
                self.human.SetTransform(start_human_trans)
                human_goal_reached = False
                cur_waypoint = 0
                self.cur_user_goal += 1

                if self.robot.simulated:
                    self.robot.arm.SetDOFValues(self.init_robot_arm_dof)
                    self.robot.SetDOFValues(self.init_robot_dof)
                else:
                    try:
                        # self.robot.arm.hand.OpenHand()
                        self.robot.arm.PlanToConfiguration(
                            self.init_robot_arm_dof, execute=True
                        )
                        self.robot.SwitchToTeleopController()
                    except PlanningError, e:
                        logger.error("Failed to plan to start config")

                self.robot_policy.reset()
                robot_goal_reached = False
            else:  # Regular movement
                if self.collab:
                    if simulate_user:
                        movement_vector = (
                            user_input_velocity * self.human_vel_coeff * time_per_iter
                        )
                        new_human_pos = human_ee_trans[0:3, 3] + movement_vector
                        new_human_trans = np.array(
                            [
                                [0, 0, 1, new_human_pos[0]],
                                [1, 0, 0, new_human_pos[1]],
                                [0, 1, 0, new_human_pos[2]],
                                [0, 0, 0, 1],
                            ]
                        )
                        self.human.SetTransform(new_human_trans)
                    else:
                        self.human.SetTransform(self.tracked_human_ee_trans)

            ### visualization ###
            vis.draw_probability_text(
                self.goal_object_poses,
                self.robot_policy.goal_predictor.get_distribution(),
            )

            if self.collab:
                vis.draw_action_arrows(
                    human_ee_trans,
                    direct_teleop_action.twist[0:3],
                    np.zeros_like(direct_teleop_action.twist[0:3]),
                )
            else:
                vis.draw_action_arrows(
                    ee_trans,
                    direct_teleop_action.twist[0:3],
                    action.twist[0:3] - direct_teleop_action.twist[0:3],
                )

            # plot trajectory
            if plot_trajectory:
                if counter % 1 == 0:
                    self.plot_trajectory(human_ee_trans, counter)
                counter = counter + 1

            ### end visualization ###

            if traj_data_recording:
                traj_data_recording.add_datapoint(
                    robot_state=copy.deepcopy(robot_state),
                    robot_dof_values=copy.copy(robot_dof_values),
                    user_input_all=copy.deepcopy(user_input_all),
                    direct_teleop_action=copy.deepcopy(direct_teleop_action),
                    executed_action=copy.deepcopy(action),
                    goal_distribution=self.robot_policy.goal_predictor.get_distribution(),
                )

            rospy.sleep(max(0.0, time_per_iter - (time.time() - itr_start_time)))

            if is_done_func(self.env, self.robot, user_input_all):
                break

        # set the intended goal and write data to file
        if traj_data_recording:
            values, qvalues = self.robot_policy.assist_policy.get_values()
            traj_data_recording.set_end_info(intended_goal_ind=np.argmin(values))
            traj_data_recording.tofile()

        # execute zero velocity to stop movement
        self.ada_teleop.execute_joint_velocities(np.zeros(len(self.manip.GetDOFValues())))

        if finish_trial_func:
            finish_trial_func()

    def execute_policy(
        self,
        scenario,
        simulate_user=False,
        direct_teleop_only=False,
        blend_level="aggressive",
        blend_only=False,
        fix_magnitude_user_command=False,
        is_done_func=is_done_func_button_hold,
        finish_trial_func=None,
        traj_data_recording=None,
        plot_trajectory=False,
    ):
        if not self.collab:
            return self.old_execute_policy(
                scenario,
                simulate_user,
                direct_teleop_only,
                blend_level,
                blend_only,
                fix_magnitude_user_command,
                is_done_func,
                finish_trial_func,
                traj_data_recording,
                plot_trajectory,
            )

        # raw_input("Press ENTER to continue")

        self.collab_human_state_machine = CollabHumanStateMachine(
            scenario=scenario,
            goals=self.goals,
            simulate_user=simulate_user,
            work_ticks=int(5 * CONTROL_HZ),
        )
        self.collab_robot_state_machine = CollabRobotStateMachine(
            robot_policy=self.robot_policy,
            total_goals=len(self.goals),
            work_ticks=int(3 * CONTROL_HZ),
            reset_ticks=int(3 * CONTROL_HZ),
        )
        time_per_iter = 1.0 / CONTROL_HZ

        # Setup visualization stuff
        vis = vistools.VisualizationHandler()
        self.remove_all_trajspos("traj_pos")
        self.remove_all_trajspos("robot_traj_pos")
        counter = 0

        human_wait_time = 0
        robot_wait_time = 0
        goal_human_ee_trans = None
        itrs_taken = 0
        while True:
            itrs_taken += 1
            itr_start_time = time.time()
            if itrs_taken * time_per_iter > scenario.max_time:
                scenario.setTime(scenario.max_time)
                scenario.set_human_wait_time(human_wait_time)
                scenario.set_robot_wait_time(robot_wait_time)
                break

            self.robot_state.ee_trans = self.GetEndEffectorTransform()
            human_ee_trans = self.human.GetTransform()

            user_input_velocity, human_status = self.collab_human_state_machine.tick(
                human_ee_trans, self.tracked_human_ee_trans
            )
            # Nothing special if the status is other than these
            if "waiting" in human_status:
                reached_goal = int(human_status[-1])
                if not self.goal_block[reached_goal]:
                    self.goal_block[reached_goal] = True
                    self.collab_human_state_machine.permit_goal_work()
                    goal_human_ee_trans = human_ee_trans.copy()
                    self.robot_inference_on = False
                else:
                    human_wait_time += 1 * time_per_iter
            elif "done" in human_status:
                self.human_done = True
                reached_goal = int(human_status[-1])
                self.goal_block[reached_goal] = False
                self.human.SetTransform(self.start_human_trans)
            elif "resetting" in human_status:
                reached_goal = int(human_status[-1])
                self.goal_block[reached_goal] = False
                self.human.SetTransform(self.start_human_trans)
            elif human_status == "moving to goal":
                self.robot_inference_on = True

            user_input_all = UserInputData(
                user_input_velocity, button_changes=[0, 0], buttons_held=[0, 0]
            )
            direct_teleop_action = self.user_input_mapper.input_to_action(
                user_input_all, self.robot_state
            )

            # Robot inference
            if self.robot_inference_on:
                self.robot_policy.update(
                    self.robot_state, direct_teleop_action, human_ee_trans
                )
            else:
                # Since robot state should continue being updated but not the belief,
                # human_ee_trans is passed as the last human_ee_trans before the human
                # started working on the goal.
                human_ee_trans_to_send = (
                    human_ee_trans if goal_human_ee_trans is None else goal_human_ee_trans
                )
                self.robot_policy.update(
                    self.robot_state, direct_teleop_action, human_ee_trans_to_send
                )

            # Robot action
            robot_action, robot_status = self.collab_robot_state_machine.tick(
                self.robot_inference_on
            )
            # Nothing special if the status is other than these
            if "waiting" in robot_status:
                reached_goal = int(robot_status[-1])
                if not self.goal_block[reached_goal]:
                    self.goal_block[reached_goal] = True
                    self.collab_robot_state_machine.permit_goal_work()
                else:
                    robot_wait_time += 1 * time_per_iter
            elif "done" in robot_status:
                self.robot_done = True
                reached_goal = int(robot_status[-1])
                self.goal_block[reached_goal] = False
                self._reset_robot()
                if not self.robot.simulated:
                    self.collab_robot_state_machine.real_reset_done()
            elif "resetting" in robot_status:
                reached_goal = int(robot_status[-1])
                self.goal_block[reached_goal] = False
                self._reset_robot()
                if not self.robot.simulated:
                    self.collab_robot_state_machine.real_reset_done()

            if not self.collab_robot_state_machine.can_move:
                self.collab_human_state_machine.stop_reset()

            # Execute robot action
            if robot_action is not None:
                joint_limit_reached = self.ada_teleop.execute_action(robot_action)

                if joint_limit_reached or self.robot.CheckSelfCollision():
                    rotate_by = np.pi if self.init_robot_arm_dof[0] > -np.pi else -np.pi
                    self.init_robot_arm_dof[0] -= rotate_by
                    self.init_robot_dof[0] -= rotate_by
                    self._reset_robot()
                    if self.robot.simulated:
                        self.collab_robot_state_machine.replan()

            # Execute human action
            if simulate_user:
                if not np.all(user_input_velocity == 0):
                    movement_vector = (
                        user_input_velocity * self.human_vel_coeff * time_per_iter
                    )
                    new_human_pos = human_ee_trans[0:3, 3] + movement_vector
                    new_human_trans = np.array(
                        [
                            [0, 0, 1, new_human_pos[0]],
                            [1, 0, 0, new_human_pos[1]],
                            [0, 1, 0, new_human_pos[2]],
                            [0, 0, 0, 1],
                        ]
                    )
                    self.human.SetTransform(new_human_trans)
            else:
                self.human.SetTransform(self.tracked_human_ee_trans)

            # Measures:
            # Grow the trajectory
            scenario.add_robot_pos(
                itrs_taken * time_per_iter, list(self.robot_state.ee_trans[:3, 3]),
            )
            scenario.add_human_pos(
                itrs_taken * time_per_iter, list(human_ee_trans[:3, 3]),
            )

            # Wrong goal probability
            self.scenario.add_wrong_goal_prob(
                self.robot_policy.goal_predictor.get_distribution(),
                self.collab_human_state_machine.cur_user_goal,
            )

            # visualization
            vis.draw_probability_text(
                self.goal_object_poses,
                self.robot_policy.goal_predictor.get_distribution(),
            )

            vis.draw_action_arrows(
                human_ee_trans,
                direct_teleop_action.twist[0:3],
                np.zeros_like(direct_teleop_action.twist[0:3]),
            )

            # plot trajectory
            if plot_trajectory:
                if counter % 1 == 0:
                    self.plot_trajectory(human_ee_trans, counter, "traj_pos")
                    self.plot_trajectory(
                        self.robot_state.ee_trans, counter, "robot_traj_pos"
                    )
                counter = counter + 1

            # Completion check
            if self.human_done and self.robot_done:
                scenario.setTime(itrs_taken * time_per_iter)
                scenario.set_human_wait_time(human_wait_time)
                scenario.set_robot_wait_time(robot_wait_time)
                break

            rospy.sleep(max(0.0, time_per_iter - (time.time() - itr_start_time)))

        # execute zero velocity to stop movement
        self.ada_teleop.execute_joint_velocities(np.zeros(len(self.manip.GetDOFValues())))

        # raw_input("Press ENTER to continue")

        if finish_trial_func:
            finish_trial_func()

    def _reset_robot(self):
        if self.robot.simulated:
            self.robot.arm.SetDOFValues(self.init_robot_arm_dof)
            self.robot.SetDOFValues(self.init_robot_dof)
        else:
            try:
                # self.robot.arm.hand.OpenHand()
                self.robot.arm.PlanToConfiguration(self.init_robot_arm_dof, execute=True)
                self.robot.SwitchToTeleopController()
            except PlanningError, e:
                logger.error("Failed to plan to start config")
