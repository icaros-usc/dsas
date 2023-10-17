#!/usr/bin/env python
from simple_environment.scenarios import Scenario
from simple_environment.util.openrave_process import OpenraveProcess
from multiprocessing.managers import BaseManager
import time
from simple_environment.util.bc_calculate import *


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

    def setRobotPositions(self, robot_positions):
        self.scenario.setRobotPositions(robot_positions)

    def getMsg(self):
        return self.msg


if __name__ == "__main__":

    START_ROBOT_POS = [0.409, 0.338]
    END_EFFECTOR_POS = [0.41, 0.05, 0.95]
    MORSEL_HEIGHT = 0.81
    NUM_WAYPOINTS = 5
    DISTURBANCE_MAX = 0.05

    NUM_CORES = 1

    # initialize scenarios
    x_ub = 0.7
    x_lb = 0.5
    y_ub = 0.2
    y_lb = -0.1
    elite_map_config = toml.load(
        "src/simple_environment/src/simple_environment/search/config/elite_map/BC.tml"
    )

    # initialize scenarios randomly
    scenarios = []
    num_scenarios = 5
    random.seed(1)
    np.random.seed(1)
    # for ss in range(0,num_scenarios):
    # p1_x = random.uniform(x_lb, x_ub)
    # p1_y = random.uniform(y_lb, y_ub)
    # p2_x = random.uniform(x_lb, x_ub)
    # p2_y = random.uniform(y_lb, y_ub)
    # p3_x = random.uniform(x_lb, x_ub)
    # p3_y = random.uniform(y_lb, y_ub)
    p3_x = 0.7
    p3_y = -0.1
    p2_x = 0.7
    p2_y = 0.05
    p1_x = 0.7
    p1_y = 0.2

    goal_pos = np.array([p2_x, p2_y, MORSEL_HEIGHT])
    start_pos = np.array([END_EFFECTOR_POS[0], END_EFFECTOR_POS[1], END_EFFECTOR_POS[2]])

    default_waypoints = []
    waypoints = []
    waypoints.append(start_pos)
    default_waypoints.append(start_pos)

    # disturbances = np.random.uniform(-DISTURBANCE_MAX, DISTURBANCE_MAX,NUM_WAYPOINTS)
    disturbances = np.array([-0.05, -0.05, -0.05, -0.05, -0.05])
    for w in range(0, NUM_WAYPOINTS):
        waypoint_pos = np.array(start_pos) + (float(w + 1) / (NUM_WAYPOINTS + 1)) * (
            np.array(goal_pos) - np.array(start_pos)
        )
        default_waypoint_pos = np.array(start_pos) + (
            float(w + 1) / (NUM_WAYPOINTS + 1)
        ) * (np.array(goal_pos) - np.array(start_pos))
        default_waypoints.append(default_waypoint_pos)
        # add disturbance
        waypoint_pos[1] = waypoint_pos[1] + disturbances[w]
        waypoints.append(waypoint_pos)
    waypoints.append(goal_pos)
    default_waypoints.append(goal_pos)

    # initialize
    scenario = Scenario(
        [(p1_x, p1_y), (p2_x, p2_y), (p3_x, p3_y)],
        waypoints,
        default_waypoints,
        disturbances,
    )
    # scenario = Scenario([(0.59, 0.03), (0.59, 0.13)], 5)
    for ss in range(num_scenarios):
        scenarios.append(scenario)

    # from IPython import embed
    # embed()

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
        process = OpenraveProcess(cc, worker, elite_map_config)
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

    beta = calc_rationality(scenario, elite_map_config)
    print ("beta is: " + str(beta))

    # file = open("logs/"+ filename,"w")
    # for ss in range(0, num_scenarios):
    #   file.write(str(ss) + "\t" + str(eval_scenarios[ss].points[0][0])+"\t" +str(eval_scenarios[ss].points[0][1])+"\t" + \
    #             str(eval_scenarios[ss].points[1][0])+"\t" +str(eval_scenarios[ss].points[1][1])+"\t" + str(eval_scenarios[ss].time)+"\n")
    # file.close()
