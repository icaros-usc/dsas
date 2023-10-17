import numpy as np
from threading import Lock
import time
import rospy

from abc import ABCMeta, abstractmethod


translation_weightings = np.array([0.2, 0.2, 0.2])
angular_weightings = np.array([0.4, 0.4, 0.4])

#amount of time that a button must be held to call it held
time_required_button_hold = 0.5


class UserInputListener(object):
  __metaclass__ = ABCMeta

  def __init__(self):
    self.inputlock = Lock()
    self.init_listener()
    self.most_recent_message = None

    #keep track of the last state of buttons we returned
    #in order to keep track of when button states change
    self.last_buttons_returned = None
    self.last_buttons_held_returned = None

    self.time_since_press = None

  def callback(self, data):
    with self.inputlock:
      self.most_recent_message = data
      
  def init_listener(self):
    rospy.Subscriber(self.input_topic_name, self.input_message_type, self.callback)

  #TODO handle case of no most_recent_message without infinite loop
  def get_most_recent_cmd(self):
    while self.most_recent_message is None:
      print 'Error: No user input to return'
      time.sleep(0.02)

    with self.inputlock:
      data = self.message_to_data(self.most_recent_message)

    #set the change between the last buttons returned as data
    #and the currently pressed buttons
    if self.last_buttons_returned is None:
      data.button_changes = np.zeros(len(data.buttons))
    else:
      data.button_changes = data.buttons - self.last_buttons_returned


    #mark time the button has been held
    curr_time = time.time()
    if self.time_since_press is None:
      self.time_since_press = np.zeros(len(data.button_changes))

    
    next_times = [curr_time if change else last_time if press else 0 for last_time,press,change in zip(self.time_since_press, data.buttons, data.button_changes)]
    self.time_since_press = next_times

    data.buttons_held = np.array( [1 if press and (curr_time - press_start_time > time_required_button_hold) else 0 for press,press_start_time in zip(data.buttons, self.time_since_press)] )



    #self.time_since_press = [ curr_time if change > 0.5 else 0 for last_time,press,change in zip(self.time_since_press, data.buttons, data.button_changes)]

    #set the change between the last buttons held returned as data
    #and the currently held buttons
    if self.last_buttons_held_returned is None:
      data.button_held_changes = np.zeros(len(data.buttons))
    else:
      data.button_held_changes = data.buttons_held - self.last_buttons_held_returned

    self.last_buttons_returned = data.buttons
    self.last_buttons_held_returned = data.buttons_held

    return data

  @abstractmethod
  def message_to_data(self, message):
    raise NotImplementedError("Must override message_to_data")


#rotates the translation inputs to the correct world frame, applies weighting
def translation_input_conversion(inputs, robot_state):
  return inputs *translation_weightings


#puts the rotation input into the world frame, applies weighting
def rotation_input_conversion(inputs, robot_state):
  ee_rot = robot_state.ee_trans[0:3,0:3]
  return np.dot(ee_rot, inputs * angular_weightings)



#axes and buttons should have a consistent order for different inputs
#first button should correspond to switching modes
#second button (if applicable) should correspond to toggling assistance
class UserInputData(object):
  def __init__(self, axes=list(), buttons=list(), button_changes=list(), buttons_held=list(), button_held_changes=list()):
    self.axes = axes
    self.buttons = buttons
    self.button_changes = button_changes
    self.buttons_held = buttons_held
    self.button_held_changes = button_held_changes

  def __str__(self):
    return 'axes: ' + str(self.axes) + '  buttons: ' + str(self.buttons)  + '  button changes: ' + str(self.button_changes) + '  buttons_held: ' + str(self.buttons_held) + '  button_held_changes: ' + str(self.button_held_changes)
