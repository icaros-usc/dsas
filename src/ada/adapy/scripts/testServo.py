#!/usr/bin/env python
import adapy, openravepy, numpy, prpy, rospy
from IPython import embed
from std_msgs.msg import Float64

rospy.init_node('test_scenario', anonymous = True)
openravepy.RaveInitialize(True, level=openravepy.DebugLevel.Debug)
openravepy.misc.InitOpenRAVELogging();
rospy.init_node('test_scenario', anonymous = True)

env, robot = adapy.initialize(attach_viewer='qtcoin', sim=False)
manip = robot.arm
twist = numpy.array([0,1,0,0,0,0])
joint_velocities =  prpy.util.ComputeJointVelocityFromTwist(robot,twist)
joint_velocities = joint_velocities[0]
#while True:

while True:
  print joint_velocities
  robot.arm.Servo(joint_velocities)
  #rospy.sleep(0.2)



#manip.SetIkSolver(openravepy.RaveCreateIkSolver(env, 'NloptIK'))
#robot.SetActiveDOFs([2,3,4,5,6,7])
#controller_j1_name = 'vel_j1_controller'
#controller_j2_name = 'vel_j2_controller'
#controller_j3_name = 'vel_j3_controller'
#controller_j4_name = 'vel_j4_controller'
#controller_j5_name = 'vel_j5_controller'
#controller_j6_name = 'vel_j6_controller'

#command_j1_topic_name = '{:s}/command'.format(controller_j1_name)
#command_j2_topic_name = '{:s}/command'.format(controller_j2_name)
#command_j3_topic_name = '{:s}/command'.format(controller_j3_name)
#command_j4_topic_name = '{:s}/command'.format(controller_j4_name)
#command_j5_topic_name = '{:s}/command'.format(controller_j5_name)
#command_j6_topic_name = '{:s}/command'.format(controller_j6_name)
#jointstate_topic_name = 'joint_states'

#velocity_j1_publisher = rospy.Publisher(command_j1_topic_name, Float64)
#velocity_j2_publisher = rospy.Publisher(command_j2_topic_name, Float64)
#velocity_j3_publisher = rospy.Publisher(command_j3_topic_name, Float64)
#velocity_j4_publisher = rospy.Publisher(command_j4_topic_name, Float64)
#velocity_j5_publisher = rospy.Publisher(command_j5_topic_name, Float64)
#velocity_j6_publisher = rospy.Publisher(command_j6_topic_name, Float64)





#from IPython import embed
#embed()
