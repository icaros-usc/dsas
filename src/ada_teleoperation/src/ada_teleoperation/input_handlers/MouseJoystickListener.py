import rospy
from sensor_msgs.msg import Joy
import numpy as np

from UserInputListener import *

rosmsg_topic = "/ada/joy"


class MouseJoystickListener(UserInputListener):
  def __init__(self):
    self.input_topic_name = "/ada/joy"
    self.input_message_type = Joy
    super(MouseJoystickListener, self).__init__()


  def message_to_data(self, message):
    return UserInputData(np.array(message.axes)*-1., np.array(message.buttons))

  #rotates the translation inputs to the correct world frame, applies weighting
  def translation_input_conversion(self, inputs, robot_state):
    return inputs * translation_weightings


  #puts the rotation input into the world frame, applies weighting
  def rotation_input_conversion(self, inputs, robot_state):
    inputs_rotated = [-inputs[1], -inputs[0], inputs[2]]
    ee_rot = robot_state.ee_trans[0:3,0:3]
    return np.dot(ee_rot, inputs_rotated * angular_weightings)
