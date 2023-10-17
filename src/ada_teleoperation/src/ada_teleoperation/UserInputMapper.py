import rospy
import numpy as np

from input_handlers.UserInputListener import UserInputData
#from HydraListener import *
from RobotState import *
from input_handlers import KinovaJoystickListener, MouseJoystickListener, UserInputListener

finger_weighting = 0.2

#mapping from user input to action
#if we were to execute input via direct teleop
class UserInputMapper(object):
  def __init__(self, interface_listener, num_motion_modes=2, num_finger_modes=0):
    self.num_motion_modes = num_motion_modes
    self.num_finger_modes = num_finger_modes
    self.interface_listener = interface_listener

  def input_to_action(self, user_input_data, robot_state):
    action_to_ret = Action()
    #if the first button is was released, and we didn't activate hold, change modes
    if user_input_data.button_changes[0] == -1 and user_input_data.buttons_held[0] != 1:
      action_to_ret.switch_mode_to = robot_state.next_mode()
      return action_to_ret
    
    curr_robot_mode = robot_state.mode
    #if curr_robot_mode == 0:
    #  from IPython import embed
    #  embed()
    #if we are in the first few modes, it is a end effector velocity command
    if curr_robot_mode < self.num_motion_modes:
      if self.num_motion_modes == 2:
        if curr_robot_mode == 0:
          action_to_ret.twist[:3] = self.interface_listener.translation_input_conversion(user_input_data.axes[0:3], robot_state)
        else:
          action_to_ret.twist[3:] = self.interface_listener.rotation_input_conversion(user_input_data.axes[0:3], robot_state)

      elif self.num_motion_modes == 3:
        if curr_robot_mode == 0:
          action_to_ret.twist[:3] = self.interface_listener.translation_input_conversion(np.append(user_input_data.axes[0:2], 0.), robot_state)
          #action_to_ret.move[0] *= -1.
        elif curr_robot_mode == 1:
          action_to_ret.twist[:3] = self.interface_listener.translation_input_conversion(np.append(np.zeros(2), user_input_data.axes[1]), robot_state)
          rot_velocity = np.array([0, 0, -user_input_data.axes[0]])
          action_to_ret.twist[3:] = self.interface_listener.rotation_input_conversion(rot_velocity, robot_state)
        else:
          rot_velocity = np.array([user_input_data.axes[1], -user_input_data.axes[0], 0.])
          action_to_ret.twist[3:] = self.interface_listener.rotation_input_conversion(rot_velocity, robot_state)
    else:
      #both left-right and up-down control fingers. With kinova control, whichever input has higher
      #magnitude overrides the other
      axis_input_higher_mag = np.argmax(np.abs(user_input_data.axes[0:2]))
      if robot_state.num_finger_dofs == 3 and axis_input_higher_mag == 1:
        #if this is the jaco, and the user went up-down, only control two fingers
        action_to_ret.finger_vel[0:2] = finger_weighting * user_input_data.axes[axis_input_higher_mag]
      else:
        action_to_ret.finger_vel[:] = finger_weighting * user_input_data.axes[axis_input_higher_mag]


    return action_to_ret


