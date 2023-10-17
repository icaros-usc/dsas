#!/usr/bin/env python
from simple_environment.scenario_generators.scenario_generator import *

from simple_environment.util.openrave_single import OpenraveSingle
#from multiprocessing.managers import BaseManager
import time
import argparse
from simple_environment.util.bc_calculate import *
from simple_environment.util.collision_detector import *


if __name__ == "__main__":

    START_ROBOT_POS =  [0.409,0.338]
    END_EFFECTOR_POS = [0.41,0.05,0.95]
    NUM_WAYPOINTS = 5
    DISTURBANCE_MAX = 0.05


    random.seed(3)
    np.random.seed(3)

    #initialize scenarios
    x_ub = 0.6
    x_lb = 0.6
    y_lb = 0.1#-0.05
    y_ub = 0.1

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', help='path of BC config file',required=True)
    opt = parser.parse_args()
    #parser = argparse.ArgumentParser()
    elite_map_config=toml.load(opt.config)
    feature_ranges = []
    column_names = []
    bc_names = []
    for bc in elite_map_config["Map"]["Features"]:
      feature_ranges.append((bc["low"],bc["high"]))
      column_names.append(bc["name"])
      bc_names.append(bc["name"])

  
    scenarios = []
    num_scenarios = 100
    random.seed(3)
    np.random.seed(3)

    METHOD = elite_map_config["method"]
    BLEND_LEVEL = elite_map_config["blend_level"]


    for ss in range(num_scenarios):
 
      #print("Running scenario: " + str(ss))
      # initialize scenarios randomly

      #for ss in range(0,num_scenarios): 
      p1_x = elite_map_config["p1_x"]
      p1_y = random.uniform(y_lb, y_ub)
      #p1_y = random.uniform(y_lb, y_ub)
      # p1_y = random.uniform(y_lb, y_ub)
      # p2_y = random.uniform(y_lb, y_ub)
      # p3_x = random.uniform(x_lb, x_ub)
      # p3_y = random.uniform(y_lb, y_ub)

      #p1_x = 0.6
      #p1_y = 0.1
      obstacle_pos = []
      obstacle_pos.append(elite_map_config["obstacle_pos_x"])
      obstacle_pos.append(random.uniform(y_lb,  y_ub))
      obstacle_pos.append(elite_map_config["obstacle_pos_z"])


      MORSEL_HEIGHT = elite_map_config["morsel_height"]
      obstacle_radius = elite_map_config["obstacle_radius"]
      padding = elite_map_config["obstacle_padding"]


      goal_pos = np.array([p1_x, p1_y, MORSEL_HEIGHT])
      start_pos = np.array([END_EFFECTOR_POS[0], END_EFFECTOR_POS[1], END_EFFECTOR_POS[2]])
      
      
      #default_waypoints = []
      #waypoints = []
      #waypoints.append(start_pos)
      #default_waypoints.append(start_pos)

      #from IPython import embed
      #embed()

#      disturbances = np.random.uniform(-DISTURBANCE_MAX, DISTURBANCE_MAX,NUM_WAYPOINTS)
      disturbances = np.array([0,0,0,0,0])
      scenario = scenario_generate_1point_obstacle(p1_x, p1_y, disturbances, obstacle_pos, elite_map_config)

      #disturbances = np.array([-0.05, -0.05, -0.05, -0.05,-0.05])

      # for w in range(0,NUM_WAYPOINTS): 
      #   waypoint_pos = np.array(start_pos) + (float(w+1)/(NUM_WAYPOINTS+1)) * (np.array(goal_pos) - np.array(start_pos))
      #   waypoints.append(waypoint_pos)
      # waypoints.append(goal_pos)
      # #default_waypoints.append(goal_pos)

      # #initialize 

      # #repair waypoints
      # for ii in range(NUM_WAYPOINTS+1):
      #   collision_waypoint_pre = waypoints[ii]
      #   collision_waypoint_pos = waypoints[ii+1]
      #   if (line_sphere_intersection(collision_waypoint_pre, collision_waypoint_pos, obstacle_pos, obstacle_radius)) == False:
      #     continue
      #   waypoint_pos_fixed_1 = collision_waypoint_pos.copy()
      #   waypoint_pos_fixed_2 = collision_waypoint_pos.copy()
      #   counter = 0 
      #   while line_sphere_intersection(collision_waypoint_pre, waypoint_pos_fixed_1, obstacle_pos, obstacle_radius) and counter < 100:
      #     waypoint_pos_fixed_1[1] = waypoint_pos_fixed_1[1] - 0.01
      #     counter = counter + 1

      #   counter = 0 
      #   while line_sphere_intersection(collision_waypoint_pre, waypoint_pos_fixed_2, obstacle_pos, obstacle_radius) and counter < 100:
      #     waypoint_pos_fixed_2[1] = waypoint_pos_fixed_2[1] + 0.01
      #     counter = counter + 1
      #   # pick waypoints
      #   d1 = np.linalg.norm(waypoint_pos_fixed_1 - collision_waypoint_pre) + np.linalg.norm(goal_pos - waypoint_pos_fixed_1)
      #   d2 = np.linalg.norm(waypoint_pos_fixed_2 - collision_waypoint_pre) + np.linalg.norm(goal_pos - waypoint_pos_fixed_2)
      #   if d1 <= d2: 
      #     waypoints[ii+1] = waypoint_pos_fixed_1.copy()
      #   else:
      #     waypoints[ii+1] = waypoint_pos_fixed_2.copy()

      # default_waypoints = list(waypoints)

      # #apply disturbances
      # for w in range(1,NUM_WAYPOINTS+1): 
      #   waypoints[w][1] = waypoints[w][1] +disturbances[w-1]

      # scenario = Scenario([(p1_x, p1_y)],waypoints,default_waypoints,disturbances)
      # scenario.setObstacle(obstacle_pos, obstacle_radius, knows_obstacle = False, padding = padding)
      # scenario.max_time = 10

        #if ss == 84:
      scenarios.append(scenario)
        #  from IPython import embed
        #  embed()
       #msg_list.append(msg)
    cc = 1
    process = OpenraveSingle(cc,elite_map_config)
    process.start()
    #process.start()
   #    process_list.append(process)
 
    startTime = time.time()
    
    while(len(scenarios)>0):
        if(scenarios):
          eval_scenario = scenarios.pop()
          print(scenario.waypoints)
          print(scenario.points)
          print(scenario.obstacle_pos)
          process.evaluate(eval_scenario, METHOD, BLEND_LEVEL)

    totalTime = time.time() - startTime
    print "Total time: " + str(totalTime)

    print "all done!!!!!!!!!!!!!!!!!!!!"



