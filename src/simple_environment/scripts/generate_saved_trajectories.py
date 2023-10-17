#!/usr/bin/env python

import adapy, argparse, logging, numpy, os, openravepy, prpy, rospy
from catkin.find_in_workspaces import find_in_workspaces
from runBiteServing import setup
from prpy.planning import PlanningError
from IPython import embed

import ada_teleoperation.KinovaStudyHelpers as KinovaStudyHelpers

project_name = 'ada_meal_scenario'
logger = logging.getLogger(project_name)

def save_path(path, filename):
    with open(filename, 'w') as f:
        path_str = path.serialize()
        f.write(path_str)

    logger.info('Saved path to file: %s' % filename)

if __name__ == "__main__":
        
    rospy.init_node('bite_serving_scenario', anonymous=True)

    parser = argparse.ArgumentParser('Ada meal scenario')
    parser.add_argument("--debug", action="store_true", help="Run with debug")
    parser.add_argument("--real", action="store_true", help="Run on real robot (not simulation)")
    parser.add_argument("--viewer", type=str, default='qtcoin', help="The viewer to load")
    args = parser.parse_args()

    sim = not args.real
    env, robot = setup(sim=sim, viewer=args.viewer, debug=args.debug)

    using_jaco = robot.GetName() == 'JACO'

    from rospkg import RosPack
    rospack = RosPack()
    package_path = rospack.get_path(project_name)

    #save old limits
    old_acceleration_limits = robot.GetDOFAccelerationLimits()
    old_velocity_limits = robot.GetDOFVelocityLimits()
    #slow down robot
    #robot.SetDOFVelocityLimits(0.6*robot.GetDOFVelocityLimits())
    #robot.SetDOFAccelerationLimits(0.8*robot.GetDOFAccelerationLimits())

    try:
        indices, values = robot.configurations.get_configuration('ada_meal_scenario_morselStabbedConfiguration')
        robot.SetDOFValues(dofindices=indices, values=values)

        #path_to_morselstabbed_configuration = robot.PlanToNamedConfiguration('ada_meal_scenario_morselStabbedConfiguration', execute=True)

        #res = openravepy.planningutils.SmoothTrajectory(path_to_morselstabbed_configuration,1, 1, 'ParabolicSmoother', '')
        #robot.ExecuteTrajectory(path_to_morselstabbed_configuration)
        #path_to_looking_at_face = robot.PlanToNamedConfiguration('ada_meal_scenario_lookingAtFaceConfiguration', execute=False)    
        #trajfile = os.path.join(package_path, 'data', 'trajectories', 'traj_lookingAtFace.xml')
        #save_path(path_to_looking_at_face, trajfile)
        #robot.ExecutePath(path_to_morselstabbed_configuration)

        path_to_serving_configuration = robot.PlanToNamedConfiguration('ada_meal_scenario_servingConfiguration', execute=True)
        trajfile = os.path.join(package_path, 'data', 'trajectories', robot.GetName() + '_traj_serving.xml')
        #res = openravepy.planningutils.SmoothTrajectory(path_to_serving_configuration,1, 1, 'ParabolicSmoother', '')
        save_path(path_to_serving_configuration, trajfile)
        #robot.ExecutePath(path_to_serving_configuration)
        #robot.ExecuteTrajectory(path_to_serving_configuration)

        path_to_looking_at_plate = robot.PlanToNamedConfiguration('ada_meal_scenario_lookingAtPlateConfiguration', execute=True)
        trajfile = os.path.join(package_path, 'data', 'trajectories', robot.GetName() + '_traj_lookingAtPlate.xml')
        #res = openravepy.planningutils.SmoothTrajectory(path_to_looking_at_plate,1, 1, 'ParabolicSmoother', '')
        save_path(path_to_looking_at_plate, trajfile)
        #robot.ExecutePath(path_to_looking_at_plate)

        #robot.ExecuteTrajectory(path_to_looking_at_plate)




    except PlanningError, e:
        logger.error('Failed to plan: %s' % str(e))
    
    robot.SetDOFVelocityLimits(old_velocity_limits)
    robot.SetDOFAccelerationLimits(old_acceleration_limits)

    logger.info('Done')
