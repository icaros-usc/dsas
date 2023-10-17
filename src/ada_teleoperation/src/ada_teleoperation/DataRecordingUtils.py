import numpy as np
import copy
import cPickle as pickle
import os.path
import glob
import time
import subprocess
import signal
import psutil

import logging
logging.basicConfig()
logger = logging.getLogger('ada_teleoperation')

import rospy, rospkg

#from RobotState import *
#from Goal import Goal

file_directory_default = rospkg.RosPack().get_path('ada_teleoperation') + '/trajectory_data'
filename_base_default = 'trajdata_'
user_folder_base_default = 'user_'

next_fileind_to_check = 0
next_userind_to_check = 0

class TrajectoryData(object):
  def __init__(self, filename, init_info=None):
    self.init_info = init_info
    self.end_info = None

    self.filename = filename


    self.data_per_time_step = []

  def set_init_info(self, *args, **kwargs):
    self.init_info = (args, add_time_to_dict(kwargs))
  
  def set_end_info(self, *args, **kwargs):
    self.end_info = (args, add_time_to_dict(kwargs))

  def add_datapoint(self, *args, **kwargs):
    if len(args) > 0:
      logger.warning('Attempting to save trajectory item with unspecified key. Please specify a key for all arguments')
    #self.data_per_time_step.append(TrajectoryData_PerTimeStep(*args, **kwargs))
    self.data_per_time_step.append((args, add_time_to_dict(kwargs)))

  def tofile(self):
    #first make sure directory exists
    file_directory = os.path.dirname(self.filename)
    if not os.path.exists(file_directory):
      os.makedirs(file_directory)
      
    #filename = get_next_filename(file_directory, filename_base)
    with open(self.filename, 'w') as f:
      pickle.dump(self, f)


#gets names for traj data and rosbag files
def get_next_filename_pair(file_directory=None, filename_base=None):
  if file_directory is None:
    file_directory = file_directory_default
  if filename_base is None:
    filename_base = filename_base_default

  file_ind = get_next_available_file_ind(file_directory, filename_base)
  filename_traj = get_filename(file_directory, filename_base, file_ind, file_type='.pckl')
  filename_bag = get_filename(file_directory, filename_base, file_ind, file_type='.bag')

  return filename_traj, filename_bag


def get_next_available_file_ind(file_directory=None, filename_base=None, file_type='.*'):
  if file_directory is None:
    file_directory = file_directory_default
  if filename_base is None:
    filename_base = filename_base_default

  #dir_and_filename_base = os.path.join(file_directory, filename_base)
  global next_fileind_to_check
  while True:
    filename = get_filename(file_directory, filename_base, next_fileind_to_check, file_type)
    print filename
    #if not os.path.isfile(filename):
    if len(glob.glob(filename)) == 0:
      break
    next_fileind_to_check += 1

  return next_fileind_to_check

def get_next_available_user_ind(file_directory=None, user_folder_base=None, make_dir=True):
  if file_directory is None:
    file_directory = file_directory_default
  if user_folder_base is None:
    user_folder_base = user_folder_base_default

  global next_userind_to_check
  while True:
    foldername = get_filename(file_directory, user_folder_base, next_userind_to_check, file_type="")
    if not os.path.isdir(foldername):
      break
    next_userind_to_check += 1

  if make_dir and not os.path.exists(foldername):
      os.makedirs(foldername)

  return next_userind_to_check, foldername


def get_filename(file_directory, filename_base, file_ind, file_type):
  return os.path.join(file_directory, filename_base) + str(file_ind).zfill(3) + file_type


def load_all_trajectorydata(file_directory=None, filename_base=None):
  if file_directory is None:
    file_directory = file_directory_default
  if filename_base is None:
    filename_base = filename_base_default
  dir_and_filename_base = os.path.join(file_directory, filename_base)

  all_filenames = glob.glob(dir_and_filename_base + '*.pckl')
  all_filenames.sort()
  return [load_trajectorydata_from_file(filename) for filename in all_filenames]

def load_trajectorydata_from_file(filename):
  return pickle.load(open(filename, 'r'))

#
#def get_next_filename(file_directory, filename_base, file_type='.pckl'):
#  dir_and_filename_base = os.path.join(file_directory, filename_base)
#  global next_fileind_to_check
#  while True:
#    filename = dir_and_filename_base + str(next_fileind_to_check).zfill(3) + file_type
#    if not os.path.isfile(filename):
#      break
#    next_fileind_to_check += 1
#
#  return filename



def add_time_to_dict(dict):
  dict['timestamp'] = time.time()
  return dict


def start_rosbag(topics, filename):
#  if file_directory is None:
#    file_directory = file_directory_default
#  if filename_base is None:
#    filename_base = filename_base_default
#  filename = get_next_filename(file_directory, filename_base, '.bag')
  
  topics_str = str(topics)[1:-2]
  topics_str = topics_str.replace("'", "")
  topics_str = topics_str.replace(",", "")

  command = "rosbag record -O " + filename + ' ' + topics_str
  
  file_directory = os.path.dirname(filename)
  rosbag_process = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True, cwd=file_directory)
  return rosbag_process

def stop_rosbag(rosbag_process):
  terminate_process_and_children(rosbag_process)


def terminate_process_and_children(p):
  children = psutil.Process(p.pid).get_children(recursive=True)
  for child_p in children:
    child_p.send_signal(signal.SIGINT)
  p.send_signal(subprocess.signal.SIGINT)

#maintain the data for each time step
#class TrajectoryData_PerTimeStep(object):
#  def __init__(self, *args, **kwargs):
#    if len(args) > 0:
#      logger.warning('Attempting to save trajectory item with unspecified key. Please specify a key for all arguments')
#
#  def __init__(self, state, dof_values, user_input, user_action, assistance_action, prior_useraction_each_goal, prob_each_goal, time):
#    self.state = copy.copy(state)
#    self.dof_values = copy.copy(dof_values)
#    self.user_input = copy.copy(user_input)
#    self.user_action = copy.copy(user_action)
#    self.assistance_action = copy.copy(assistance_action)
#    self.prior_useraction_each_goal = copy.copy(prior_useraction_each_goal)
#    self.prob_each_goal = copy.copy(prob_each_goal)
#    self.time = time
