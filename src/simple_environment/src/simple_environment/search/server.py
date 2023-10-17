#!/usr/bin/env python
"""Flask server for evaluating solutions.

Based on search/search.py

Usage:
    python search/server.py --help
"""
import argparse
import atexit
import glob
import os
import sys
import time
import traceback

from flask import Flask, request, jsonify
from multiprocessing.managers import BaseManager

from simple_environment.util import bc_calculate
from simple_environment import scenario_generators
from simple_environment.util.SearchHelper import *
from simple_environment.util.openrave_process import OpenraveProcess


def print_header(s):
    print "========== " + s + " =========="


print_header("Working directory: " + os.getcwd())
sys.path.append(os.getcwd())

if not os.path.exists("logs"):
    os.mkdir("logs")

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
        self.scenario = None

    def setMsg(self, msg):
        self.msg = msg

    def getMsg(self):
        return self.msg

    def getId(self):
        return self.id

    def setScenario(self, scenario):
        self.scenario = scenario

    def getScenario(self):
        return self.scenario


def evaluate_openrave(scenario, worker_id):
    worker_list[worker_id].setScenario(scenario)
    worker_list[worker_id].setMsg("evaluate")


evaluate = evaluate_openrave


def worker_has_finished(worker_id):
    """Returns true if the scenario either succeeds or if there is an error and false if
    it is still running."""
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


def terminate_all_workers(experiment_toml, num_cores):
    experiment_toml = experiment_toml["Trials"][0]
    for cc in range(0, num_cores):
        worker_list[cc].setMsg("terminate")
        process_list[cc].join()


def start_worker(worker_id):
    worker_list[worker_id].setMsg("start")


def start_openrave(elite_map_config, num_cores):
    BaseManager.register("Worker", Worker)
    manager = BaseManager()
    manager.start()
    # starting processes
    for cc in range(0, num_cores):
        worker = manager.Worker(cc)
        worker_list.append(worker)
        # msg_list.append(msg)
        process = OpenraveProcess(cc, worker, elite_map_config)
        process.start()
        process_list.append(process)
        running_workers.append(worker.getId())
    for cc in range(0, num_cores):
        start_worker(cc)

    wait_all_workers_finished()


def initialize(experiment_toml, num_cores):
    # Collect configurations.
    experiment_toml = experiment_toml["Trials"][0]
    trial_toml = toml.load(experiment_toml["trial_config"])
    NumSimulations = trial_toml["num_simulations"]
    AlgorithmToRun = trial_toml["algorithm"]
    AlgorithmConfig = toml.load(trial_toml["algorithm_config"])
    elite_map_config = toml.load(trial_toml["elite_map_config"])

    # Start OpenRAVE workers.
    start_openrave(elite_map_config, num_cores)

    return elite_map_config


def evaluate_batch(scenario_params, elite_map_config):
    """Evaluates a batch of scenarios.

    `scenario_params` should be a list of dicts:

    [
        # First entry.
        {
            # Name of a scenario generation function in scenario_generators
            "function": "scenario_generate",
            "features": ["calc_distance_between_objects", "calc_variation"],
            "solution: ...  # Solution given by QD
            "kwargs": {...} # Kwargs to pass to scenario generation function,
        },
        # Other entries.
        ...
    ]

    The return value will also be a list of dicts:

    [
        # In case of an error.
        {
            "status": "error",
            "message": "...",
        },

        # In case of success.
        {
            "status": "success",
            # Other fields -- see code.
            ...
        },
    ]
    """
    print "Evaluating %s solutions" % len(scenario_params)

    start_time = time.time()

    results = [{} for _ in scenario_params]
    num_simulated = 0
    num_to_evaluate = len(scenario_params)

    while num_simulated < num_to_evaluate:
        # The evaluations are done in a batchwise fashion over all the workers.
        # First, we send scenarios to all the workers with the evaluate()
        # function.
        while len(idle_workers) > 0 and num_simulated < num_to_evaluate:
            params = scenario_params[num_simulated]

            # Retrieve scenario generation function.
            try:
                scenario_function = getattr(scenario_generators, params["function"])
            except AttributeError:
                results[num_simulated] = {
                    "status": "error",
                    "message": "Unknown scenario function: " + params["function"],
                }
                num_simulated += 1
                continue

            # Generate the scenario.
            kwargs = params["kwargs"]
            kwargs.update(elite_map_config)
            try:
                scenario = scenario_function(params["solution"], **kwargs)
            except RuntimeError as e:
                results[num_simulated] = {
                    "status": "error",
                    "message": "Scenario creation failed: " + e.message,
                }
                num_simulated += 1
                continue

            # Set an ID so that we can assign the result to the results array
            # later.
            scenario.setID(num_simulated)

            # Increment simulation counter.
            num_simulated += 1

            # Start running the scenario on a worker.
            worker_id = idle_workers.pop(0)
            print "starting simulation: " + str(num_simulated) + "/" + str(
                num_to_evaluate
            ) + " on worker " + str(worker_id)

            evaluate(scenario, worker_id)
            running_workers.append(worker_id)

        # Next, we wait for all the workers to finish and collect their results.
        while len(running_workers) > 0:
            worker_id = running_workers[0]
            if worker_has_finished(worker_id):
                scenario = worker_list[worker_id].getScenario()
                result_id = scenario.getID()
                params = scenario_params[result_id]
                kwargs = params["kwargs"]
                kwargs.update(elite_map_config)

                features = []
                for bc in params["features"]:
                    get_feature = getattr(bc_calculate, bc)
                    features.append(get_feature(scenario, kwargs))

                results[result_id] = {
                    "status": str(worker_list[worker_id].getMsg()),
                    "features": features,
                    "fitness": scenario.getTime(),
                }

                if "collab" in params["function"]:
                    results[result_id].update(
                        {
                            # Surrogate model data.
                            "points": scenario.points,
                            "obstacles": scenario.obstacles,
                            "human_trajectory": scenario.human_trajectory,
                            "robot_trajectory": scenario.robot_trajectory,
                        }
                    )
                else:
                    results[result_id].update(
                        {
                            # Surrogate model data.
                            "points": scenario.points,
                            "robot_trajectory": scenario.robot_trajectory,
                        }
                    )

                running_workers.pop(0)
                idle_workers.append(worker_id)

    total_time = time.time() - start_time
    print "Total time: " + str(total_time)

    wait_all_workers_finished()

    # Delete cmovetraj_*.txt potentially generated by the planner
    for f in glob.glob("cmovetraj_*.txt"):
        os.remove(f)

    return results


def main():
    #
    # Parse arguments.
    #

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="path of experiment config file", required=True
    )
    parser.add_argument(
        "-p", "--port", help="Port for server.", type=int, default=5000, required=False
    )
    parser.add_argument(
        "-nc", "--n_cpus", help="Number of cpus.", type=int, default=1, required=False
    )
    parser.add_argument(
        "--test", help="Run a test on evaluate_batch", action="store_true", required=False
    )

    opt = parser.parse_args()
    experiment_toml = toml.load(opt.config)

    if opt.test:
        print_header("Running test mode")

    print_header("Initializing workers")
    elite_map_config = initialize(experiment_toml, opt.n_cpus)

    #
    # Server setup.
    #

    app = Flask(__name__)

    @app.route("/")
    def server_hello_world():
        return "Hello, World! This confirms that the Flask server is running."

    @app.route("/ncores")
    def server_ncores():
        return str(opt.n_cpus)

    @app.route("/log", methods=["POST"])
    def server_log():
        data = request.form
        print "==> LOG: %s" % data["message"]
        return "Logged: %s" % data["message"]

    @app.route("/evaluate", methods=["POST"])
    def server_evaluate():
        data = request.form
        print "/evaluate received POST request: %s" % data
        scenario_params = json.loads(data["scenario_params"])
        print "scenario_params: %s" % scenario_params
        results = evaluate_batch(scenario_params, elite_map_config)
        return jsonify(results)

    # Catches errors. See here:
    # https://code-maven.com/python-flask-catch-exception
    @app.errorhandler(Exception)
    def server_error(err):
        print "Server error!"
        app.logger.exception(err)
        return traceback.format_exc(err), 500

    #
    # Register cleanup function -- doesn't seem to work though.
    #

    def cleanup():
        print_header("Terminating workers")
        terminate_all_workers(experiment_toml, opt.n_cpus)
        print_header("Done")

    atexit.register(cleanup)

    if opt.test:
        # Run test mode.
        results = evaluate_batch(
            [
                {
                    "function": "scenario_generate",
                    "kwargs": {
                        "p1_x": 0.05,
                        "p1_y": 0.05,
                        "p2_x": 0.15,
                        "p2_y": 0.15,
                        "disturbances": [0.0, 0.0, 0.0, 0.0, 0.0,],
                    },
                },
            ]
            * 10,
            elite_map_config,
        )
        print results
    else:
        # Start the server -- not the best way to run things; see:
        # - https://flask.palletsprojects.com/en/1.1.x/api/#flask.Flask.run
        # - https://stackoverflow.com/questions/69455452/run-flask-app-with-arguments-in-command-line
        app.run(host="0.0.0.0", port=opt.port)


if __name__ == "__main__":
    main()
