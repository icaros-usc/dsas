#!/usr/bin/env python
import adapy, openravepy, numpy, prpy, rospy
from IPython import embed

def PlanToTransform(env, robot, transform):
    handle = openravepy.misc.DrawAxes(env, transform);
    iksolver = robot.arm.GetIkSolver()
    param = openravepy.IkParameterization(transform, openravepy.IkParameterizationType.Transform6D)
    solution = iksolver.Solve(param, robot.GetActiveDOFValues(),  0)
    traj =  robot.PlanToConfiguration(solution.GetSolution())
    return traj;

def PlanToOffset(env, robot, offset):
    transform = robot.arm.GetEndEffectorTransform()
    transform[0:3, 3] += offset;
    traj = PlanToTransform(env, robot, transform);
    return traj

rospy.init_node('test_scenario', anonymous = True)
openravepy.RaveInitialize(True, level=openravepy.DebugLevel.Debug)
openravepy.misc.InitOpenRAVELogging();
rospy.init_node('test_scenario', anonymous = True)

from IPython import embed
embed()

env, robot = adapy.initialize(attach_viewer = 'rviz', sim=True)
manip = robot.arm




#manip.SetIkSolver(openravepy.RaveCreateIkSolver(env, 'NloptIK'))
#robot.SetActiveDOFs([2,3,4,5,6,7])
robot.arm.SetActive()
values = robot.GetActiveDOFValues()
values[1] = values[1] - 0.3
robot.PlanToConfiguration(values, execute=True)


