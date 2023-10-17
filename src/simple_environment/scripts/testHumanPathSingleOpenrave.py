#!/usr/bin/env python
from simple_environment.scenarios import Scenario
from simple_environment.util.openrave_single import OpenraveSingle
import time
from simple_environment.util.bc_calculate import *
import argparse


if __name__ == "__main__":

    START_ROBOT_POS =  [0.409,0.338]
    END_EFFECTOR_POS = [0.41,0.05,0.95]
    MORSEL_HEIGHT = 0.81 
    NUM_WAYPOINTS = 5
    DISTURBANCE_MAX = 0.05

    NUM_CORES = 1

    #METHOD = 'shared_auton_always'
    METHOD = 'blend'

    #initialize scenarios
    #x_ub = 0.50
    #x_lb = 0.50
    #y_ub = -0.05 
    #y_lb = -0.05
    


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
      x_lb =elite_map_config["x_b"][0]
      x_ub = elite_map_config["x_b"][1]
      y_lb =elite_map_config["y_b"][0]
      y_ub = elite_map_config["y_b"][1]
      
      # initialize scenarios randomly
    scenarios = []
    num_scenarios = 10
    random.seed(1)
    np.random.seed(1)
    
    for ss in range(num_scenarios):


      #for ss in range(0,num_scenarios): 
      p1_x = random.uniform(x_lb, x_ub)
      p2_x = random.uniform(x_lb, x_ub)
     
      p1_x = 0.6
      p2_x = 0.65
      p1_y = 0.125
      p2_y = 0.1
#      p1_y = random.uniform(y_lb, y_ub)
#      p2_y = random.uniform(y_lb, y_ub)

      goal_pos = np.array([p2_x, p2_y, MORSEL_HEIGHT])
      start_pos = np.array([END_EFFECTOR_POS[0], END_EFFECTOR_POS[1], END_EFFECTOR_POS[2]])
      
      
      default_waypoints = []
      waypoints = []
      waypoints.append(start_pos)
      default_waypoints.append(start_pos)


      #disturbances = np.random.uniform(-DISTURBANCE_MAX, DISTURBANCE_MAX,NUM_WAYPOINTS)
      disturbances = np.array([0.02, 0.02, 0.02, 0.02, 0.02])

      for w in range(0,NUM_WAYPOINTS): 
        waypoint_pos = np.array(start_pos) + (float(w+1)/(NUM_WAYPOINTS+1)) * (np.array(goal_pos) - np.array(start_pos))
        default_waypoint_pos = np.array(start_pos) + (float(w+1)/(NUM_WAYPOINTS+1)) * (np.array(goal_pos) - np.array(start_pos))
        default_waypoints.append(default_waypoint_pos)
        #add disturbance
        waypoint_pos[1] = waypoint_pos[1] +disturbances[w]
        waypoints.append(waypoint_pos)
      waypoints.append(goal_pos)
      default_waypoints.append(goal_pos)

      #initialize 
      scenario = Scenario([(p1_x, p1_y), (p2_x, p2_y)],waypoints,default_waypoints,disturbances)
      scenario.max_time = 20
      scenarios.append(scenario)

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
          #print(scenario.obstacle_pos)
          process.evaluate(eval_scenario, METHOD)

    totalTime = time.time() - startTime
    print "Total time: " + str(totalTime)

    print "all done!!!!!!!!!!!!!!!!!!!!"
