import rospy
from sensor_msgs.msg import Joy
import numpy as np

from UserInputListener import *


class KinovaJoystickListener(UserInputListener):
  def __init__(self):
    self.input_topic_name = "/ada/joy"
    self.input_message_type = Joy
    super(KinovaJoystickListener, self).__init__()

  def message_to_data(self, message):
    axes = np.array([message.axes[0], -message.axes[1], -message.axes[2]])
    #move_velocity = np.array([-message.axes[0], message.axes[1], 0])
    #mode_switch_button = message.buttons[0]
    #close_hand_velocity = message.buttons[1]
    #assist_switch_button = False

    return UserInputData(axes, np.array(message.buttons))


  #rotates the translation inputs to the correct world frame, applies weighting
  def translation_input_conversion(self, inputs, robot_state):
    return inputs * translation_weightings


  #puts the rotation input into the world frame, applies weighting
  def rotation_input_conversion(self, inputs, robot_state):
    inputs_rotated = [-inputs[1], -inputs[0], inputs[2]]
    ee_rot = robot_state.ee_trans[0:3,0:3]
    return np.dot(ee_rot, inputs_rotated * angular_weightings)
