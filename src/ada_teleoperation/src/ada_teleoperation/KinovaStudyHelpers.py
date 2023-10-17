#Helper files for our kinova study

import rospy
import numpy as np
from prpy.tsr.rodrigues import *
import openravepy



robot_handedness = rospy.get_param('/ada/handedness', 'left')


#def convertJointConfigToLeftHanded(config):




#takes pose (in world coordinates) for a right handed configuration
#robot and converts to left handed
def ConvertEEPoseHandedness(robot_pose, ee_pose):
  pose_robot_frame = np.dot(np.linalg.inv(robot_pose), ee_pose)

  mirrored_pose = np.eye(4)
  n = np.array([1., 0., 0.])
  for i in range(4):
    d = pose_robot_frame[0:3, i]
    mirrored_pose[0:3, i] = d - 2*(np.dot(d, np.transpose(n)))* n

  #this transform will be left-handed instead of right. flip y
  mirrored_pose[0:3, 1] *= -1.

  return np.dot(robot_pose, mirrored_pose)


def ConvertArmConfigHandedness(robot, config):
  with robot:
    robot.arm.SetDOFValues(config)

    pose = robot.arm.GetEndEffectorTransform()
    pose_converted = ConvertEEPoseHandedness(robot.GetTransform(), pose)
    #config_converted = robot.arm.FindIKSolution(pose_converted, 0)

    #find many solutions, return the one closest to the specified configuration
    sols = robot.arm.FindIKSolutions(pose_converted, 0)
    errs = np.sum(np.abs(sols - config)**2, axis=-1)
  return sols[np.argmin(errs), :]

