#!/usr/bin/env python

import adapy, argparse, logging, numpy, os, openravepy, prpy, rospy, rospkg, time
import numpy as np
from catkin.find_in_workspaces import find_in_workspaces
from std_msgs.msg import String

from prpy.planning.base import PlanningError
from prpy.tsr.rodrigues import rodrigues

import random

from simple_environment.scenario import Scenario
from simple_environment.actions.bite_serving import BiteServing
from simple_environment.actions.bypassable_action import ActionException
from simple_environment.gui_handler import *

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

project_name = 'simple_environment'
logger = logging.getLogger(project_name)

import time

def setup(sim=False, viewer=None, debug=True):
    # load the robot and environment for meal serving

    # find the openrave environment file
    data_base_path = find_in_workspaces(
        search_dirs=['share'],
        project=project_name,
        path='data',
        first_match_only=True)
    if len(data_base_path) == 0:
        raise Exception('Unable to find environment path. Did you source devel/setup.bash?')
    env_path = os.path.join(data_base_path[0], 'environments', 'table.env.xml')
    
    # Initialize logging
    if debug:
        openravepy.RaveInitialize(True, level=openravepy.DebugLevel.Debug)
    else:
        openravepy.RaveInitialize(True, level=openravepy.DebugLevel.Info)
    openravepy.misc.InitOpenRAVELogging()
    prpy.logger.initialize_logging()

    # Load the environment and robot
    env, robot = adapy.initialize(attach_viewer=viewer, sim=sim, env_path=env_path)

    # Set the active manipulator on the robot
    robot.arm.SetActive()

    # Now set everything to the right location in the environment
    #if using jaco, assume we have the portable mount, and robot should have a different distance to table
    using_jaco = robot.GetName() == 'JACO'
    if using_jaco:
      robot_pose = numpy.array([[1., 0., 0., 0.409],
                              [0., 1., 0., 0.338],
                              [0., 0., 1., 0.754],
                              [0., 0., 0., 1.]])
    else:
      robot_pose = numpy.array([[1., 0., 0., 0.409],
                              [0., 1., 0., 0.338],
                              [0., 0., 1., 0.795],
                              [0., 0., 0., 1.]])

    with env:
        robot.SetTransform(robot_pose)

    # Set the robot joint configurations
    ResetTrial(robot)


    return env, robot


    



def ResetTrial(robot):
  # set the robot to the start configuration for next trial
  logger.info('Resetting Robot')
  if robot.simulated:
      indices, values = robot.configurations.get_configuration('ada_meal_scenario_servingConfiguration')
      robot.SetDOFValues(dofindices=indices, values=values)
  else:
    #first try to plan to serving
    try:
      robot.PlanToNamedConfiguration('ada_meal_scenario_servingConfiguration', execute=True)
    except PlanningError, e:
      logger.info('Failed to plan to start config')
      #if it doesn't work, unload controllers


def setup_trial_recording(record_next_trial, file_directory_user):
    # creates user directory if we will be recording
    if record_next_trial and not os.path.exists(file_directory_user):
        os.makedirs(file_directory_user)

def RandomSearch(robot):
    x_ub = 0.50
    x_lb = 0.70
    y_ub = 0.03
    y_lb = 0.031


    # initialize scenarios randomly
    num_scenarios = 3
    scenarios = []
    for ss in range(0,num_scenarios): 
      p1_x = random.uniform(x_lb, x_ub)
      p1_y = random.uniform(y_lb, y_ub)
      p2_x = random.uniform(x_lb, x_ub)
      p2_y = random.uniform(y_lb, y_ub)
      #scenario = Scenario([(p1_x, p1_y), (p1_x, p1_y)])
      scenario = Scenario([(0.50, 0.23), (0.60, 0.03)], 10)
      scenarios.append(scenario)

    # evaluate scenarios
    for ss in range(0, num_scenarios): 
        try:
          print "scenario: " + str(ss)+"\n"
          # robot set initial position
          robot.arm.SetDOFValues(np.array([-2.38, -0.1959051, -0.34364456 , -0.688872349, 0,
        3.55386329]))

          scenario = scenarios[ss]
          manip = robot.GetActiveManipulator()
          action = BiteServing(scenario)
  
          start_time = time.time()
          action.execute(manip, env, method='shared_auton_always', ui_device = 'kinova', state_pub=state_pub,detection_sim = args.detection_sim, record_trial=False,file_directory=file_directory_user, scenario = scenario)
        except ActionException, e:
          logger.info('Failed to complete bite serving: %s' % str(e))


        ResetTrial(robot)

    file = open("logs/trial.txt","w") 
    for ss in range(0, num_scenarios):
      file.write(str(ss) + "\t" + str(scenarios[ss].points[0][0])+"\t" +str(scenarios[ss].points[0][1])+"\t" + \
                str(scenarios[ss].points[1][0])+"\t" +str(scenarios[ss].points[1][1])+"\t" + \
                str(scenarios[ss].time)+"\n")
    file.close()
    #from IPython import embed
    #embed()



if __name__ == "__main__":
    state_pub = rospy.Publisher('ada_tasks',String, queue_size=10)
        
    rospy.init_node('bite_serving_scenario', anonymous=True)

    parser = argparse.ArgumentParser('Ada meal scenario')
    parser.add_argument("--debug", action="store_true", help="Run with debug")
    parser.add_argument("--real", action="store_true", help="Run on real robot (not simulation)")
    parser.add_argument("--viewer", type=str, default='interactivemarker', help="The viewer to load")
    parser.add_argument("--detection-sim", action="store_true", help="Simulate detection of morsel")
    parser.add_argument("--userid", type=int, help="User ID number")
    args = parser.parse_args(rospy.myargv()[1:]) # exclude roslaunch args

    sim = not args.real
    env, robot = setup(sim=sim, viewer=args.viewer, debug=args.debug)

    robot.arm.SetDOFValues(np.array([-2.38, -0.1959051, -0.34364456 , -0.688872349, 0,
        3.55386329]))

    #gui_get_event, gui_trial_starting_event, gui_queue, gui_process = start_gui_process()
 
    # Where to store rosbags and other user data - set this manually if userid was provided,
    # otherwise dynamically generate it as one more than highest in directory
    file_directory_user = None
    file_directory = rospkg.RosPack().get_path('simple_environment') + '/trajectory_data'
    if args.userid:
        from ada_teleoperation.DataRecordingUtils import get_filename
        file_directory_user = get_filename(file_directory=file_directory, filename_base='user_', file_ind=args.userid, file_type="")
        # Check whether the file_directory_user exists
        if os.path.exists(file_directory_user):
            inp = raw_input("Filename " + file_directory_user + " exists. Press q to quit or enter to continue")
            if inp == 'q':
                raise OSError("Filename already exists. Quitting.")
    else:
        from ada_teleoperation.DataRecordingUtils import get_next_available_user_ind
        user_number, file_directory_user = get_next_available_user_ind(file_directory=file_directory, make_dir=False)

    RandomSearch(robot)
    #self.button_full_auton = self.init_button_with_callback(self.select_assistance_method, 'autonomous', 'Fully Autonomous', frame)
    #gui_process.terminate()

