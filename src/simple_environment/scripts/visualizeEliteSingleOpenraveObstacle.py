#!/usr/bin/env python
import os
import sys
import argparse
import time

print(os.getcwd())
sys.path.append(os.getcwd())
#from util import bc_calculate


#import torchvision.utils as vutils
import csv

from simple_environment.scenario_generators.scenario_generator import *

from simple_environment.util.bc_calculate import *
from simple_environment.util.openrave_single import OpenraveSingle

#MarioGame = autoclass('engine.core.MarioGame')
#Agent = autoclass('agents.robinBaumgarten.Agent')

"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c','--config', help='path of experiment config file',required=True)
opt = parser.parse_args()
"""

if not os.path.exists("logs"):
    os.mkdir("logs")

#large variation, side
#large variation, line
#cell indices: cell_x = 5, cell_y = 94
#              cell_x = 14, cell_y = 72



global idle_workers, running_workers
idle_workers = []
running_workers = []

global worker_list, process_list
worker_list = []
process_list = []


global NUM_CORES
NUM_CORES = 1

MAP_SIZE_1 = 20
MAP_SIZE_2 = 100


global START_ROBOT_POS
START_ROBOT_POS = (0.409,0.338)

global NUM_SIMULATIONS
NUM_SIMULATIONS = 100

NUM_WAYPOINTS = 5 


#file_name = "MAPELITES_BC_sim1_elites_freq20.csv"
#BC_TYPE = "BC_obstacle_baseline"

file_name = "MAPELITES_BC_sim2_elites_freq20.csv"
BC_TYPE = "BC_obstacle_blend"


ALG_TYPE = "ME"
folder_name = str(BC_TYPE + "/" + ALG_TYPE + "/")

class Elite():
  def __init__(self, sc_id, p1_y, disturbances, obstacle_pos_y, fitness, cell_x, cell_y, elite_map_config):
    #new_points = [(points[0][0],points[0][1]),(points[1][0],points[1][1])]
    self.fitness = fitness
    self.id = sc_id
    self.cell_x = cell_x
    self.cell_y = cell_y

    p1_x = elite_map_config["p1_x"]

    END_EFFECTOR_POS = elite_map_config["start_end_effector_pos"]
    #p1_x = 0.6
    #p1_y = 0.1
    obstacle_pos = []
    obstacle_pos.append(elite_map_config["obstacle_pos_x"])
    obstacle_pos.append(obstacle_pos_y)
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

    self.scenario = scenario_generate_1point_obstacle(p1_x, p1_y, disturbances, obstacle_pos, elite_map_config)



    
def  load_map(feature_ranges, elite_map_config):
  feature_map = FeatureMap(-1, feature_ranges, resolutions = (MAP_SIZE_1, MAP_SIZE_2))
  with open("../output/"+folder_name + file_name) as csvfile:
    all_records = csv.reader(csvfile, delimiter=',')
    for i,one_map in enumerate(all_records):
        elites_per_iteration = []
        if  i == 499:
          for data_point in one_map:
            #i=i+1
            data_point=data_point[1:-1]

            data_point_info=data_point.split(', ')
            sc_id = int(data_point_info[0])
            feature1 = float(data_point_info[10])
            feature2 = float(data_point_info[11])
            fitness = float(data_point_info[2])

            cell_x = feature_map.get_feature_index(0, feature1)
            cell_y = feature_map.get_feature_index(1, feature2)
            cell_x = MAP_SIZE_1-1 - cell_x
            p1_y = (float(data_point_info[3]))
            obstacle_pos_y = (float(data_point_info[4]))
            disturbances = []
            for w in range(NUM_WAYPOINTS): 
              disturbances.append(float(data_point_info[w+5]))
              #disturbances.append(0)
            #cell_x = int(cell_x*MAP_SIZE_1)
            #cell_y = int(cell_y*MAP_SIZE_2)
            #if cell_x >= MAP_SIZE_1 or cell_y >= MAP_SIZE_2: 
            #  continue
            elite = Elite(sc_id, p1_y, disturbances, obstacle_pos_y, fitness, cell_x, cell_y, elite_map_config)
            elites_per_iteration.append(elite)

    elites = elites_per_iteration
    return elites


def find_elite(elites, cell_x, cell_y):
    #from IPython import embed
    #embed()
    for i in range(0,len(elites)):
        if elites[i].cell_x == cell_x and elites[i].cell_y == cell_y:
            #from IPython import embed
            #embed()
            return elites[i]
    sys.exit("Elite not found!") 




	


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-c','--config', help='path of BC config file',required=True)
  parser.add_argument('-x','--cell_x', help='path of experiment cel_x file',type=int,required=True)
  parser.add_argument('-y','--cell_y', help='path of experiment cell_y file',type=int,required=True)
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


  method = elite_map_config["method"]
  blend_level = elite_map_config["blend_level"]
  opt = parser.parse_args()
  random.seed(1)
   

  process = OpenraveSingle(1,elite_map_config)
  cell_x = MAP_SIZE_1-1 - opt.cell_x
  cell_y = opt.cell_y

  elites = load_map(feature_ranges, elite_map_config)

  #from IPython import embed
  #embed()


  elite = find_elite(elites, cell_x, cell_y)
  #from IPython import embed
  #embed()
  scenario = elite.scenario
  process.start()
  scenario.max_time = 15


  NUM_SCENARIOS = 100
  for ss in range(NUM_SCENARIOS):
    startTime = time.time()  
    #eval_scenario = scenarios.pop()
    print(scenario.waypoints)
    print(scenario.points)
    process.evaluate(scenario, method = method, blend_level = blend_level)
    print("Time is: " + str(scenario.time))

  exit()
  
  # elite = []
  # for i in range(len(elites)):
  #   check_elite = elites[i]
  #   if (abs(check_elite.scenario.points[0][1] - check_elite.scenario.points[1][1]) < 0.05) and check_elite.fitness > 9.0 and check_elite.cell_y < 30 and check_elite.cell_x < 20:
  #     print(str(MAP_SIZE_1-1-check_elite.cell_x)+", "+str(check_elite.cell_y))
  #     elite = check_elite
  #     #process.start()
  #     #    process_list.append(process)
  #     scenario = elite.scenario
  #     NUM_SCENARIOS = 1
  #     for ss in range(NUM_SCENARIOS):
  #       startTime = time.time()  
  #       #eval_scenario = scenarios.pop()
  #       print(scenario.waypoints)
  #       print(scenario.points)
  #       process.evaluate(scenario)

  # #start_openrave(elite_map_config)
  #from IPython import embed
  #embed()
  #start_run(elite)
  #terminate_all_workers()
 

