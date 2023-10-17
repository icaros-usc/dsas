#!/usr/bin/env python
import rospy
from simple_environment.scenario import Scenario
from openrave_process import OpenraveProcess
from multiprocessing.managers import BaseManager
import random
from multiprocessing import Process
from std_msgs.msg import String
import time


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


if __name__ == "__main__":

    NUM_CORES = 6

    # initialize scenarios
    x_ub = 0.50
    x_lb = 0.60
    y_ub = 0.13
    y_lb = 0.03

    # initialize scenarios randomly
    scenarios = []
    num_scenarios = 10
    random.seed(1)
    for ss in range(0, num_scenarios):
        p1_x = random.uniform(x_lb, x_ub)
        p1_y = random.uniform(y_lb, y_ub)
        p2_x = random.uniform(x_lb, x_ub)
        p2_y = random.uniform(y_lb, y_ub)
        scenario = Scenario([(p1_x, p1_y), (p2_x, p2_y)])
        # scenario = Scenario([(0.59, 0.03), (0.59, 0.13)], 5)
        scenarios.append(scenario)

    BaseManager.register("Worker", Worker)

    manager = BaseManager()
    manager.start()

    idle_workers = []
    running_workers = []

    # starting processes
    worker_list = []
    process_list = []
    for cc in range(0, NUM_CORES):
        worker = manager.Worker(cc)
        worker_list.append(worker)
        # msg_list.append(msg)
        process = OpenraveProcess(cc, worker)
        process.start()
        process_list.append(process)
        worker.setMsg("start")
        running_workers.append(worker.getId())

    # initialize random search processes
    # all_finished = False
    while len(running_workers) > 0:
        # for cc in range(0,NUM_CORES):
        #  if worker_list[cc].getMsg()== "success":
        worker_id = running_workers[0]
        if worker_list[worker_id].getMsg() == "success":
            running_workers.pop(0)
            idle_workers.append(worker_id)
    # while msg_list[0].getMsg()!="success" or msg_list[1].getMsg()!="success" :
    #  time.sleep(1)

    startTime = time.time()

    eval_scenarios = []
    while len(eval_scenarios) < num_scenarios:
        while len(idle_workers) > 0:
            if scenarios:
                worker_id = idle_workers.pop(0)
                eval_scenario = scenarios.pop()
                worker_list[worker_id].setScenario(eval_scenario)
                worker_list[worker_id].setMsg("evaluate")
                running_workers.append(worker_id)
            else:
                break

        while len(running_workers) > 0:
            worker_id = running_workers[0]
            if worker_list[worker_id].getMsg() == "success":
                running_workers.pop(0)
                scenario = worker_list[worker_id].getScenario()
                print "\n\n\n\nretrieving scenario" + str(scenario)
                eval_scenarios.append(scenario)
                idle_workers.append(worker_id)

    totalTime = time.time() - startTime
    print "Total time: " + str(totalTime)

    print "all done!!!!!!!!!!!!!!!!!!!!"
    # finalize random search processes
    while len(running_workers) > 0:
        # for cc in range(0,NUM_CORES):
        #  if worker_list[cc].getMsg()== "success":
        worker_id = running_workers[0]
        if worker_list[worker_id].getMsg() == "success":
            running_workers.pop(0)
            idle_workers.append(worker_id)

    for cc in range(0, NUM_CORES):
        worker_list[cc].setMsg("terminate")
        process_list[cc].join()

    filename = "trial" + str(NUM_CORES) + ".txt"

    # from IPython import embed
    # embed()
    file = open("logs/" + filename, "w")
    for ss in range(0, num_scenarios):
        file.write(
            str(ss)
            + "\t"
            + str(eval_scenarios[ss].points[0][0])
            + "\t"
            + str(eval_scenarios[ss].points[0][1])
            + "\t"
            + str(eval_scenarios[ss].points[1][0])
            + "\t"
            + str(eval_scenarios[ss].points[1][1])
            + "\t"
            + str(eval_scenarios[ss].time)
            + "\n"
        )
    file.close()
