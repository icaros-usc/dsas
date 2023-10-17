#!/usr/bin/env python
import os
import sys
import argparse
import time

print (os.getcwd())
sys.path.append(os.getcwd())

# from util import bc_calculate


# import torchvision.utils as vutils
import csv

from simple_environment.util.openrave_process import OpenraveProcess
from multiprocessing.managers import BaseManager
import random
from simple_environment.scenario_generators.scenario_generator import *


# MarioGame = autoclass('engine.core.MarioGame')
# Agent = autoclass('agents.robinBaumgarten.Agent')

"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c','--config', help='path of experiment config file',required=True)
opt = parser.parse_args()
"""

if not os.path.exists("logs"):
    os.mkdir("logs")

file_name = "RANDOM_BC_sim1000_elites_freq1.csv"

global idle_workers, running_workers
idle_workers = []
running_workers = []

global worker_list, process_list
worker_list = []
process_list = []


global NUM_CORES
NUM_CORES = 14

feature1Range = (0, 0.316)
feature2Range = (0.212, 0.46)
gridsize = 25

global START_ROBOT_POS
START_ROBOT_POS = (0.409, 0.338)

global NUM_SIMULATIONS
NUM_SIMULATIONS = 100


class Elite(Scenario):
    def __init__(self, points, cell_x, cell_y):
        new_points = [(points[0][0], points[0][1]), (points[1][0], points[1][1])]
        Scenario.__init__(self, new_points)
        self.cell_x = cell_x
        self.cell_y = cell_y


class Worker(object):
    def __init__(self, id):
        self.id = id
        self.msg = "null"
        self.status = "inactive"

    def setMsg(self, msg):
        self.msg = msg

    def getId(self):
        return self.id

    def setScenario(self, scenario):
        self.scenario = scenario

    def setEvalTime(self, time):
        self.scenario.time = time

    def getEvalTime(self):
        return self.scenario.time

    def getScenario(self):
        return self.scenario

    def getMsg(self):
        return self.msg


def evaluate_openrave(scenario, worker_id):
    # from IPython import embed
    # embed()
    worker_list[worker_id].setScenario(scenario)
    worker_list[worker_id].setMsg("evaluate")


evaluate = evaluate_openrave


def run_one_trial(elite):
    simulations = 0
    startTime = time.time()
    while simulations < NUM_SIMULATIONS:
        while len(idle_workers) > 0 and simulations < NUM_SIMULATIONS:
            scenario = elite
            worker_id = idle_workers.pop(0)
            # time.sleep(0.5)
            # from IPython import embed
            # embed()
            evaluate(scenario, worker_id)
            running_workers.append(worker_id)
        while len(running_workers) > 0:
            worker_id = running_workers[0]
            if worker_has_finished(worker_id):
                scenario = worker_list[worker_id].getScenario()
                print "Scenario time: " + str(scenario.time)
                # if scenario.time >= 9.99:
                # from IPython import embed
                # embed()
                totalTime = time.time() - startTime
                running_workers.pop(0)
                idle_workers.append(worker_id)
                simulations = simulations + 1

    print "Total time: " + str(totalTime)

    wait_all_workers_finished()


def worker_has_finished(worker_id):
    if worker_list[worker_id].getMsg() in ["success", "error"]:
        return True
    else:
        return False


def wait_all_workers_finished():
    while len(running_workers) > 0:
        worker_id = running_workers[0]
        if worker_has_finished(worker_id):
            running_workers.pop(0)
            idle_workers.append(worker_id)


def terminate_all_workers():
    for cc in range(0, NUM_CORES):
        worker_list[cc].setMsg("terminate")
        process_list[cc].join()


def start_worker(worker_id):
    worker_list[worker_id].setMsg("start")


def start_openrave():
    BaseManager.register("Worker", Worker)
    manager = BaseManager()
    manager.start()
    # starting processes
    for cc in range(0, NUM_CORES):
        worker = manager.Worker(cc)
        worker_list.append(worker)
        # msg_list.append(msg)
        process = OpenraveProcess(cc, worker, START_ROBOT_POS)
        process.start()
        process_list.append(process)
        running_workers.append(worker.getId())
    for cc in range(0, NUM_CORES):
        start_worker(cc)

    wait_all_workers_finished()


def load_map():
    with open("logs/" + file_name) as csvfile:
        all_records = csv.reader(csvfile, delimiter=",")
        for i, one_map in enumerate(all_records):
            elites_per_iteration = []
            for data_point in one_map:
                # i=i+1
                data_point = data_point[1:-1]
                # from IPython import embed
                # embed()
                data_point_info = data_point.split(", ")
                cell_x = (float(data_point_info[7]) - feature1Range[0]) / (
                    feature1Range[1] - feature1Range[0]
                )
                cell_y = (float(data_point_info[8]) - feature2Range[0]) / (
                    feature2Range[1] - feature2Range[0]
                )
                # cell_x = gridsize-1 - int(cell_x*gridsize)
                p1_x = float(data_point_info[3])
                p1_y = float(data_point_info[4])
                p2_x = float(data_point_info[5])
                p2_y = float(data_point_info[6])
                cell_x = int(cell_x * gridsize)
                cell_y = int(cell_y * gridsize)
                if cell_x >= gridsize or cell_y >= gridsize:
                    continue
                elite = Elite([(p1_x, p1_y), (p2_x, p2_y)], cell_x, cell_y)
                elites_per_iteration.append(elite)
        # from IPython import embed
        # embed()
        elites = elites_per_iteration
        return elites


def find_elite(elites, cell_x, cell_y):
    # from IPython import embed
    # embed()
    for i in range(0, len(elites)):
        if elites[i].cell_x == cell_x and elites[i].cell_y == cell_y:
            return elites[i]
    sys.exit("Elite not found!")


def start_run(elite):
    # experiment_toml=experiment_toml["Trials"][0]
    # trial_toml=toml.load(experiment_toml["trial_config"])
    # global EliteMapConfig
    # EliteMapConfig=toml.load(trial_toml["elite_map_config"])
    # TrialName=trial_toml["trial_name"]+"_sim"+str(trial_id)
    run_one_trial(elite)
    print ("Finished One Trial")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--cell_x", help="path of experiment cel_x file", type=int, required=True
    )
    parser.add_argument(
        "-y", "--cell_y", help="path of experiment cell_y file", type=int, required=True
    )

    opt = parser.parse_args()
    random.seed(1)

    # cell_x = gridsize-1 - int(cell_x*gridsize)

    opt = parser.parse_args()
    elites = load_map()
    elite = find_elite(elites, opt.cell_x, opt.cell_y)
    # from IPython import embed
    # embed()
    start_openrave()
    start_run(elite)
    terminate_all_workers()

    # NEXT: fix registration and map

    # #quick code will fix later
    # filename = "trial" + str(NUM_CORES) +".txt"
    # file = open("logs/"+ filename,"w")
    # for ss in range(0, num_scenarios):
    #   file.write(str(ss) + "\t" + str(eval_scenarios[ss].points[0][0])+"\t" +str(eval_scenarios[ss].points[0][1])+"\t" + \
    #               str(eval_scenarios[ss].points[1][0])+"\t" +str(eval_scenarios[ss].points[1][1])+"\t" + str(eval_scenarios[ss].time)+"\n")
    # file.close()
