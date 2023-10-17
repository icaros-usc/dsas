#!/usr/bin/env python
import os
import sys
import argparse
import time

print (os.getcwd())
sys.path.append(os.getcwd())
from simple_environment.util import SearchHelper

# from util import bc_calculate
import pathlib

import sys

import pandas as pd
import numpy as np
from numpy.linalg import eig

# import torchvision.utils as vutils
import toml
import json
import numpy
import json
import numpy
import math
import random
from collections import OrderedDict
import csv
from algorithms import *
from simple_environment.util.SearchHelper import *
from simple_environment.util.bc_calculate import *
from simple_environment.util.openrave_process import OpenraveProcess
from multiprocessing.managers import BaseManager
import random
from multiprocessing import Process

if not os.path.exists("logs"):
    os.mkdir("logs")

global EliteMapConfig
EliteMapConfig = []

global idle_workers, running_workers
idle_workers = []
running_workers = []

global worker_list, process_list
worker_list = []
process_list = []


global RECORD_FREQUENCY
RECORD_FREQUENCY = 20


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

    def setCollided(self, collided):
        self.scenario.collided = collided

    def setEvalTime(self, time):
        self.scenario.time = time

    def getEvalTime(self):
        return self.scenario.time

    def getScenario(self):
        return self.scenario

    def setRobotPositions(self, robot_positions):
        self.scenario.setRobotPositions(robot_positions)

    def getMsg(self):
        return self.msg


def log_individual(algorithm_instance, algorithm_name, scenario, elite_map_config):
    messageReceived = str(scenario.getTime())
    statsList = messageReceived.split(",")
    scenario.statsList = statsList
    identity = scenario.getID()
    #
    NUM_POINTS = elite_map_config["num_points"]
    has_obstacle = elite_map_config["has_obstacle"]
    if NUM_POINTS == 1 and has_obstacle:
        record = (
            [algorithm_name]
            + [scenario.getTime()]
            + [scenario.points[0][1]]
            + [scenario.obstacle_pos[1]]
        )
    elif NUM_POINTS == 2:
        record = (
            [algorithm_name]
            + [scenario.getTime()]
            + [scenario.points[0][0]]
            + [scenario.points[0][1]]
            + [scenario.points[1][0]]
            + [scenario.points[1][1]]
        )
    elif NUM_POINTS == 3:
        record = (
            [algorithm_name]
            + [scenario.getTime()]
            + [scenario.points[0][0]]
            + [scenario.points[0][1]]
            + [scenario.points[1][0]]
            + [scenario.points[1][1]]
            + [scenario.points[2][0]]
            + [scenario.points[2][1]]
        )
    else:
        sys.error("Unknown number of points!")
    # from IPython import embed
    # embed()
    NUM_WAYPOINTS = len(scenario.disturbances)
    for ww in range(NUM_WAYPOINTS):
        record += [scenario.disturbances[ww]]

    record += [scenario.features[0]] + [scenario.features[1]] + [scenario.features[2]]

    algorithm_instance.allRecords.loc[identity] = record
    if algorithm_instance.individuals_evaluated_total % RECORD_FREQUENCY == 0:
        elites = [
            algorithm_instance.feature_map.elite_map[x]
            for x in algorithm_instance.feature_map.elite_map
        ]

        # from IPython import embed
        # embed()

        if len(elites) != 0:

            logFile = open(
                "logs/"
                + algorithm_instance.trial_name
                + "_elites_freq"
                + str(RECORD_FREQUENCY)
                + ".csv",
                "a",
            )
            rowData = []

            for x in elites:
                currElite = [x.getID()]
                currElite += algorithm_instance.allRecords.loc[x.getID()].tolist()

                rowData.append(currElite)
            wr = csv.writer(logFile, dialect="excel")
            wr.writerow(rowData)
            logFile.close()


def evaluate_openrave(scenario, worker_id):
    worker_list[worker_id].setScenario(scenario)
    worker_list[worker_id].setMsg("evaluate")


evaluate = evaluate_openrave


def run_trial(
    num_to_evaluate, algorithm_name, algorithm_config, elite_map_config, trial_name
):
    feature_ranges = []
    NUM_POINTS = elite_map_config["num_points"]
    if NUM_POINTS == 1:
        has_obstacle = elite_map_config["has_obstacle"]
        if has_obstacle is False:
            sys.exit("for only one goal has_obstacle should be set true")
        column_names = ["emitterName", "timeSpent", "P1_y", "obstacle_y"]
    elif NUM_POINTS == 2:
        column_names = ["emitterName", "timeSpent", "P1_x", "P1_y", "P2_x", "P2_y"]
    elif NUM_POINTS == 3:
        column_names = [
            "emitterName",
            "timeSpent",
            "P1_x",
            "P1_y",
            "P2_x",
            "P2_y",
            "P3_x",
            "P3_y",
        ]
    else:
        sys.error("unknown NUM_POINTS")

    NUM_WAYPOINTS = elite_map_config["num_waypoints"]
    for ww in range(NUM_WAYPOINTS):
        column_names += ["dist" + str(ww)]
    bc_names = []
    MAP_SIZE_1 = elite_map_config["map_size_1"]
    MAP_SIZE_2 = elite_map_config["map_size_2"]
    MAP_SIZE_3 = elite_map_config["map_size_3"]
    for bc in elite_map_config["Map"]["Features"]:
        feature_ranges.append((bc["low"], bc["high"]))
        column_names.append(bc["name"])
        bc_names.append(bc["name"])
    feature_map = FeatureMap(
        num_to_evaluate, feature_ranges, resolutions=(MAP_SIZE_1, MAP_SIZE_2, MAP_SIZE_3)
    )
    if algorithm_name == "CMAES":
        print ("Start Running CMAES")
        mutation_power = algorithm_config["mutation_power"]
        population_size = algorithm_config["population_size"]
        algorithm_instance = CMA_ES_Algorithm(
            mutation_power,
            population_size,
            num_to_evaluate,
            feature_map,
            trial_name,
            column_names,
            bc_names,
            elite_map_config,
        )
    elif algorithm_name == "MAPELITES":
        print ("Start Running MAPELITES")
        mutation_power_pos = algorithm_config["mutation_power_pos"]
        mutation_power_disturb = algorithm_config["mutation_power_disturb"]
        initial_population = algorithm_config["initial_population"]
        algorithm_instance = MapElitesAlgorithm(
            mutation_power_pos,
            mutation_power_disturb,
            initial_population,
            num_to_evaluate,
            feature_map,
            trial_name,
            column_names,
            bc_names,
            elite_map_config,
        )
    elif algorithm_name == "RANDOM":
        print ("Start Running RANDOM")
        algorithm_instance = RandomGenerator(
            num_to_evaluate,
            feature_map,
            trial_name,
            column_names,
            bc_names,
            elite_map_config,
        )

    startTime = time.time()
    simulation = 0
    while algorithm_instance.is_running():
        while (
            len(idle_workers) > 0
            and simulation < num_to_evaluate
            and not algorithm_instance.is_blocking()
        ):
            scenario = algorithm_instance.generate_individual()
            worker_id = idle_workers.pop(0)
            simulation = simulation + 1
            print ("starting simulation: " + str(simulation) + "/" + str(num_to_evaluate))
            evaluate(scenario, worker_id)
            running_workers.append(worker_id)

        # print "coverage: " + str(float(len(algorithm_instance.feature_map.elite_map))/ (MAP_SIZE_1 * MAP_SIZE_2))
        QD_score = 0
        for i in range(0, len(algorithm_instance.feature_map.elite_indices)):
            QD_score = (
                QD_score
                + algorithm_instance.feature_map.elite_map[
                    algorithm_instance.feature_map.elite_indices[i]
                ].fitness
            )
        # print "QD_score: " + str(QD_score)
        # next, try changing that to idle
        while len(running_workers) > 0:
            worker_id = running_workers[0]
            if worker_has_finished(worker_id):
                scenario = worker_list[worker_id].getScenario()
                # beta = calc_rationality(scenario, elite_map_config)
                # print("beta is: "+str(beta))
                SSE = calc_variation(scenario, elite_map_config)
                # print("SSE is:" + str(SSE))
                algorithm_instance.return_evaluated_individual(scenario)
                log_individual(
                    algorithm_instance, algorithm_name, scenario, elite_map_config
                )
                running_workers.pop(0)
                idle_workers.append(worker_id)

    totalTime = time.time() - startTime
    print "Total time: " + str(totalTime)

    print "coverage: " + str(
        float(len(algorithm_instance.feature_map.elite_map)) / (MAP_SIZE_1 * MAP_SIZE_2)
    )
    QD_score = 0
    for i in range(0, len(algorithm_instance.feature_map.elite_indices)):
        QD_score = (
            QD_score
            + algorithm_instance.feature_map.elite_map[
                algorithm_instance.feature_map.elite_indices[i]
            ].fitness
        )
    print "QD_score: " + str(QD_score)

    algorithm_instance.allRecords.to_csv("logs/" + trial_name + "all_simulations.csv")

    # see number of restarts
    # print "number of restarts:" + str(algorithm_instance.emitters[0].num_restarts)

    wait_all_workers_finished()
    # print "Individuals clipped: " + str(algorithm_instance.individuals_clipped)


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


def terminate_all_workers(experiment_toml):
    experiment_toml = experiment_toml["Trials"][0]
    NUM_CORES = experiment_toml["num_cores"]
    for cc in range(0, NUM_CORES):
        worker_list[cc].setMsg("terminate")
        process_list[cc].join()


def start_worker(worker_id):
    worker_list[worker_id].setMsg("start")


def start_openrave(elite_map_config, NUM_CORES):
    BaseManager.register("Worker", Worker)
    manager = BaseManager()
    manager.start()
    # starting processes
    for cc in range(0, NUM_CORES):
        worker = manager.Worker(cc)
        worker_list.append(worker)
        # msg_list.append(msg)
        process = OpenraveProcess(cc, worker, elite_map_config)
        process.start()
        process_list.append(process)
        running_workers.append(worker.getId())
    for cc in range(0, NUM_CORES):
        start_worker(cc)

    wait_all_workers_finished()


def start_search(trial_id, experiment_toml):
    experiment_toml = experiment_toml["Trials"][0]
    trial_toml = toml.load(experiment_toml["trial_config"])
    NumSimulations = trial_toml["num_simulations"]
    AlgorithmToRun = trial_toml["algorithm"]
    AlgorithmConfig = toml.load(trial_toml["algorithm_config"])
    global EliteMapConfig
    EliteMapConfig = toml.load(trial_toml["elite_map_config"])
    NUM_CORES = experiment_toml["num_cores"]
    start_openrave(EliteMapConfig, NUM_CORES)
    TrialName = trial_toml["trial_name"] + "_sim" + str(trial_id)
    run_trial(NumSimulations, AlgorithmToRun, AlgorithmConfig, EliteMapConfig, TrialName)
    print ("Finished One Trial")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="path of experiment config file", required=True
    )
    parser.add_argument(
        "-t", "--trialID", help="path of experiment config file", type=int, required=True
    )
    opt = parser.parse_args()

    trial_id = opt.trialID - 1
    # random.seed(4) #seed 4 for not reaching target
    opt = parser.parse_args()
    experiment_toml = toml.load(opt.config)
    start_search(trial_id, experiment_toml)
    terminate_all_workers(experiment_toml)
