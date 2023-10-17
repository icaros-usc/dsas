#!/usr/bin/env python

import rospy
import numpy as np
import argparse

from UserInputMapper import UserInputMapper
from RobotState import RobotState
import input_handlers
from input_handlers import KinovaJoystickListener, HydraListener, MouseJoystickListener

from AdaTeleopHandler import *


VIEWER_DEFAULT = 'InteractiveMarker'
SIMULATE_DEFAULT = False

def Initialize_Adapy(args):
    """ Initializes robot and environment through adapy, using the specified environment path

    @param env_path path to OpenRAVE environment
    @return environment, robot
    """

    #env_path = '/environments/tablewithobjects_assisttest.env.xml'
    adapy_args = {'sim':args.sim,
                  'attach_viewer':args.viewer,
                  #'env_path':env_path
                  }
    openravepy.RaveInitialize(True, level=openravepy.DebugLevel.Debug)
    #openravepy.misc.InitOpenRAVELogging();
    env, robot = adapy.initialize(**adapy_args)
    Init_Robot(robot)
    return env,robot

def Init_Robot(robot):
    robot.SetActiveDOFs(range(6))
    #self.robot.arm.hand.OpenHand()
    #if (self.sim):
    robot_pose = np.array([[1, 0, 0, 0.409],[0, 1, 0, 0.338],[0, 0, 1, 0.795],[0, 0, 0, 1]])
    robot.SetTransform(robot_pose)
    if (robot.simulated):
      #servo simulator params
      robot.arm.servo_simulator.period=1./200.

def Reset_Robot(robot):
  if robot.simulated:
    num_hand_dofs = len(robot.arm.hand.GetDOFValues())
    inds, pos = robot.configurations.get_configuration('home')
    with robot.GetEnv():
      robot.SetDOFValues(pos, inds)
      robot.arm.hand.SetDOFValues(np.ones(num_hand_dofs)*0.1)
  #else:
    #robot.arm.hand.OpenHand()
  #  robot.arm.PlanToNamedConfiguration('home', execute=True)


if __name__ == "__main__":
    #parser.add_argument('-num', '--mouse-num', help='mouse number given by X in /dev/input/mouseX', type=str)
    parser = argparse.ArgumentParser(description="Direct Teleoperation for Ada")

    parser.add_argument('-s', '--sim', action='store_true', default=SIMULATE_DEFAULT,
                        help='simulation mode')
    parser.add_argument('-v', '--viewer', nargs='?', const=True, default=VIEWER_DEFAULT,
                        help='attach a viewer of the specified type')
    #parser.add_argument('--env-xml', type=str,
                        #help='environment XML file; defaults to an empty environment')
    parser.add_argument('--debug', action='store_true',
                        help='enable debug logging')
    parser.add_argument('-input', '--input-interface-name', help='name of the input interface. Possible choices: ' + str(possible_teleop_interface_names), type=str)
    parser.add_argument('-joy_dofs', '--num-input-dofs', help='number of dofs of input, either 2 or 3', type=int, default=2)
    args = parser.parse_args()

    rospy.init_node('ada_teleoperation', anonymous = True)

    env,robot = Initialize_Adapy(args)
    Reset_Robot(robot)
    ada_teleop = AdaTeleopHandler(env, robot, args.input_interface_name, args.num_input_dofs)
    ada_teleop.execute_direct_teleop()

