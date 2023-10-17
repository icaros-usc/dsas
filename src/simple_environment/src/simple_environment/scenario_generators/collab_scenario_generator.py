import copy
import glob
import os
import pickle
import random
import time
from inspect import getmembers, isfunction

import matplotlib.pyplot as plt
from multiprocessing.managers import BaseManager

import numpy as np
import toml

from simple_environment.scenarios import CollabScenario
from simple_environment.util import bc_calculate
from simple_environment.util.mdp_human import MDPHuman
from simple_environment.util.openrave_process import OpenraveProcess
from simple_environment.util.openrave_single import OpenraveSingle


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


def collab_scenario_generate(solution, **scenario_kwargs):
    """Generate a collaborative scenario given the goal positions and noise distribution.
    The human is assumed to always move from start -> goal 1 -> start -> goal 2 -> start
    -> goal 3. Waypoints are only generated between the start and the goal and not on the
    way back.

    Args:
        solution: Solution given by QD search. `human_mode` guides how the solution
            parsed. In case of "waypoints", the first `num_goals * 2` elements are treated
            as (x, y) for each goal and the rest are treated as the waypoint disturbances.
            In case of "mdp" the first `num_goals * 2` elements are treated as (x, y) for
            each goal and the rest are treated as (x, y, r) for the obstacle if available.
        scenario_kwargs: Following keyword args (some of the mandatory ones are not
            actually mandatory, but it would take time to refactor and test it):
            - morsel_height: Height of the goal object.
            - num_waypoints: (mandatory if human_mode is "waypoints") Number of waypoints
                between the start and the goal. num_waypoints waypoints will be added for
                each goal.
            - start_human_pos: Starting human position in 3D (we only care about x, y in
                this project, so the z is currently ignored and set to morsel height).
            - human_mode: (optional) Either "mdp" or "waypoints". (default: "mdp")
            - num_goals: (optional) Number of goal objects. (default: 3)
            - one_shot: (optional) True if the scenario should end after robot reaches one
                goal. (default: False)
            - start_robot_pos: (mandatory if human_mode is "mdp")

            # MDP parameters when human_mode == "mdp" or "stochastic" (see MDPHuman for
            #   docs)
            - softmax (default: True)
            - discount_factor (default: 0.9999)
            - reward_goal (default: 1)
            - reward_move (default: -0.01)
            - reward_stay (default: -0.01)
            - vi_max_iters (default: 5000)
            - softmax_temperature (default: 1000)

    Returns:
        CollabScenario with the given parameters.
    """
    mandatory_kwargs = ["morsel_height", "start_human_pos"]
    human_mode = scenario_kwargs.get("human_mode", "mdp")
    if human_mode == "mdp":
        mandatory_kwargs.append("start_robot_pos")
    elif human_mode == "waypoints":
        mandatory_kwargs.append("num_waypoints")

    for kwarg in mandatory_kwargs:
        if kwarg not in scenario_kwargs:
            raise ValueError(
                "{} is mandatory but not specified in scenario_kwargs".format(kwarg)
            )

    morsel_height = scenario_kwargs["morsel_height"]
    human_pos = scenario_kwargs["start_human_pos"][:2] + [morsel_height]

    num_goals = scenario_kwargs.get("num_goals", 3)
    # Goal sequence is 0 -> 1 -> ...
    points = [(solution[3 * i + 1], solution[3 * i + 2]) for i in range(num_goals)]
    goals = np.array([[x, y, morsel_height] for x, y in points])

    mdp_human = None
    waypoints = None
    default_waypoints = None
    obstacles = None
    human_vel_coeff = 0.5
    if human_mode == "mdp":
        robot_pos = scenario_kwargs["start_robot_pos"]
        robot_obstacle = robot_pos[:2] + [0.05]  # 1 cell radius in the MDP grid
        obstacles = np.array([robot_obstacle]).tolist()

        mdp_start_time = time.time()
        # Parameters are not tuned
        mdp_params = dict(
            softmax=True,  # This only controls how human acts, not belief update
            discount_factor=0.9999,
            reward_goal=1,
            reward_move=-0.01,
            reward_stay=-0.01,
            vi_max_iters=5000,
            softmax_temperature=1000,
        )
        for k in mdp_params:
            if k in scenario_kwargs:
                mdp_params[k] = scenario_kwargs[k]

        mdp_human = MDPHuman(
            grid_bounds=np.array([[0.1, 2.0], [-0.15, 2.0]]),
            cell_size=np.array([0.05, 0.05]),
            start_location=np.array(human_pos[:2]),
            goal_locations=np.array(points),
            obstacles=np.array(obstacles),
            inaccessible_func=lambda center: (
                center[0] <= robot_pos[0] and center[1] <= robot_pos[1]
            ),
            **mdp_params
        )

        # print "MDP construction time: {} s".format(time.time() - mdp_start_time)
        # print ""
        # print mdp_human.visualize_grid(human_path=True)
        # mdp_human.visualize_grid_img(human_path=True, show_vf=True)

    elif human_mode == "waypoints":
        num_waypoints = scenario_kwargs["num_waypoints"]
        disturbances = solution[3 * num_goals :]
        if len(disturbances) == num_waypoints:
            disturbances *= num_goals

        disturbances = np.array(disturbances).reshape((-1, num_waypoints))

        default_waypoints = []
        waypoints = []

        # Go from start -> goal1 -> goal2 -> goal3 with num_waypoints between each point
        for i, g in enumerate(goals):
            wps = np.linspace(human_pos, g, num_waypoints + 2)
            default_waypoints.append(wps.copy())

            # Add noise to the y-coord of the waypoints between the start and the goal
            wps[1:-1][:, 1] += disturbances[i]
            waypoints.append(wps)

        default_waypoints = np.concatenate(default_waypoints).reshape(
            (num_goals, num_waypoints + 2, -1)
        )
        waypoints = np.concatenate(waypoints).reshape((num_goals, num_waypoints + 2, -1))
        
    elif human_mode == "stochastic":
        robot_pos = scenario_kwargs["start_robot_pos"]
        robot_obstacle = robot_pos[:2] + [0.05]  # 1 cell radius in the MDP grid
        obstacles = np.array([robot_obstacle]).tolist()

        mdp_start_time = time.time()
        # Parameters are not tuned
        mdp_params = dict(
            softmax=True,  # This only controls how human acts, not belief update
            stochastic=True,
            discount_factor=0.9999,
            reward_goal=1,
            reward_move=-0.01,
            reward_stay=-0.01,
            vi_max_iters=5000,
        )
        for k in mdp_params:
            if k in scenario_kwargs:
                mdp_params[k] = scenario_kwargs[k]

        mdp_params["softmax_temperature"] = 10 ** solution[3 * num_goals]
        human_vel_coeff = solution[3 * num_goals + 1]
        print "Temp: {}; Vel: {}".format(solution[3 * num_goals], solution[3 * num_goals + 1])

        mdp_human = MDPHuman(
            grid_bounds=np.array([[0.1, 2.0], [-0.15, 2.0]]),
            cell_size=np.array([0.05, 0.05]),
            start_location=np.array(human_pos[:2]),
            goal_locations=np.array(points),
            obstacles=np.array(obstacles),
            inaccessible_func=lambda center: (
                    center[0] <= robot_pos[0] and center[1] <= robot_pos[1]
            ),
            **mdp_params
        )

        # print "MDP construction time: {} s".format(time.time() - mdp_start_time)
        # print ""
        # print mdp_human.visualize_grid(human_path=True)
        # mdp_human.visualize_grid_img(human_path=True, show_vf=True)

    scenario = CollabScenario(
        points,
        waypoints,
        default_waypoints,
        human_pos,
        one_shot=scenario_kwargs.get("one_shot", False),
        mdp_human=mdp_human,
        obstacles=obstacles,
        human_vel_coeff=human_vel_coeff,
    )
    return scenario


def test_collab_scenario_generate():
    random.seed(1)
    np.random.seed(1)
    # method = "shared_auton"

    import argparse

    parser = argparse.ArgumentParser(description="Test collab scenario.")
    parser.add_argument(
        "--one_shot",
        action="store_true",
        help="Set if the scenario should end after robot reaches one goal.",
    )
    args = parser.parse_args()

    # initialize scenarios
    sol = [
        1.3334516286849976,
        0.6798813343048096,
        -0.06518277525901794,
        -5.029469013214111,
        0.138273224234581,
        0.688606858253479,
        -0.14719024300575256,
        0.15000610053539276,
        0.5621386170387268,
        3,
        0.5,
    ]  # Scenario 1

    num_scenarios = 1
    elite_map_config = toml.load("search/config/elite_map/BC_collab.tml")

    try:
        scenario = collab_scenario_generate(
            sol, one_shot=args.one_shot, **elite_map_config
        )
    except RuntimeError as e:
        print "Scenario creation failed: " + e.message
        return
    scenarios = [copy.deepcopy(scenario) for _ in range(num_scenarios)]

    cc = 1
    single = True
    if single:
        process = OpenraveSingle(
            cc, elite_map_config, simulate_user=True, simulate_robot=True
        )
        process.start()
    else:
        BaseManager.register("Worker", Worker)
        manager = BaseManager()
        manager.start()
        worker = manager.Worker(cc)
        process = OpenraveProcess(cc, worker, elite_map_config)
        process.start()
        worker.setMsg("start")
        while worker.getMsg() not in ["success", "error"]:
            pass

    start_time = time.time()

    while len(scenarios) > 0:
        eval_scenario = scenarios.pop()
        print eval_scenario.waypoints
        print eval_scenario.points
        if single:
            process.evaluate(eval_scenario)
        else:
            worker.setScenario(eval_scenario)
            worker.setMsg("evaluate")
            while worker.getMsg() not in ["success", "error"]:
                pass
            eval_scenario = worker.getScenario()

        print "Objective: {}".format(eval_scenario.getTime())
        print "Measures:"
        for bc_name, bc_func in getmembers(bc_calculate, isfunction):
            print "{}: {}".format(bc_name, bc_func(eval_scenario, None))

    total_time = time.time() - start_time
    print "Total time: " + str(total_time)

    # with open("human_traj.pkl", "wb") as f:
    #     pickle.dump(eval_scenario.human_trajectory, f)
    # with open("robot_traj.pkl", "wb") as f:
    #     pickle.dump(eval_scenario.robot_trajectory, f)

    # size = (32, 32)
    # bounding_box = [[0.1, 1.0], [-0.15, 0.8]]
    #
    # def _compute_occupancy_grid(size, bounding_box, trajectory, scale=2):
    #     grid = np.zeros(size)
    #
    #     n = len(trajectory)
    #     for timestamp, pt in trajectory:
    #         i0 = _compute_index(size[0], bounding_box[0], pt[0])
    #         i1 = _compute_index(size[1], bounding_box[1], pt[1])
    #         grid[i0, i1] += 1.0 / (1.0 + grid[i0, i1]) ** scale
    #     grid /= n
    #
    #     return grid
    #
    # def _compute_index(size, bounds, pos, eps=1e-9):
    #     interval_size = bounds[1] - bounds[0]
    #     index = (size * (pos - bounds[0]) + eps) / interval_size
    #     return np.clip(index, 0, size - 1).astype(np.int32)
    #
    # with open("human_traj.pkl", "rb") as f:
    #     human_trajectory = pickle.load(f)
    # with open("robot_traj.pkl", "rb") as f:
    #     robot_trajectory = pickle.load(f)
    #
    # human_grid = _compute_occupancy_grid(size, bounding_box, human_trajectory, scale=4)
    # robot_grid = _compute_occupancy_grid(size, bounding_box, robot_trajectory)
    #
    # fig, ax = plt.subplots()
    # plt.axis("off")
    # plt.imshow(human_grid)
    # fig.savefig("human_grid.png")
    # fig.savefig("human_grid.svg")
    # fig.savefig("human_grid.pdf")
    #
    # fig, ax = plt.subplots()
    # plt.axis("off")
    # plt.imshow(robot_grid)
    # fig.savefig("robot_grid.png")
    # fig.savefig("robot_grid.svg")
    # fig.savefig("robot_grid.pdf")

    print "all done!!!!!!!!!!!!!!!!!!!!"

    # Delete cmovetraj_*.txt potentially generated by the planner
    for f in glob.glob("cmovetraj_*.txt"):
        os.remove(f)


if __name__ == "__main__":
    test_collab_scenario_generate()
