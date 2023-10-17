import copy
import glob
import os
import random
import sys
import time
from inspect import getmembers, isfunction

import numpy as np
import toml

from simple_environment.scenarios import Scenario
from simple_environment.util import bc_calculate
from simple_environment.util.collision_detector import line_sphere_intersection
from simple_environment.util.openrave_single import OpenraveSingle


# boundary_value = 5.12
# nz = 32

# need to make a tml file for these

# def scenario_generate_1point(p1_x,p1_y, disturbances, elite_map_config):
#     #print "hello!"
#     #p1_x = random.uniform(x_lb, x_ub)
#     #p1_y = random.uniform(y_lb, y_ub)
#     #p2_x = random.uniform(x_lb, x_ub)
#     #p2_y = random.uniform(y_lb, y_ub)
#     print("hello!!!!!")
#     MORSEL_HEIGHT = elite_map_config["morsel_height"]
#     NUM_WAYPOINTS = elite_map_config["num_waypoints"]
#     END_EFFECTOR_POS = elite_map_config["start_end_effector_pos"]
#     points = [(p1_x, p1_y)]
#     goal_pos = np.array([p1_x, p1_y, MORSEL_HEIGHT])
#     start_pos = np.array([END_EFFECTOR_POS[0], END_EFFECTOR_POS[1], END_EFFECTOR_POS[2]])


#     default_waypoints = []
#     waypoints = []
#     waypoints.append(start_pos)
#     default_waypoints.append(start_pos)
#     for w in range(0,NUM_WAYPOINTS):
#       waypoint_pos = np.array(start_pos) + (float(w+1)/(NUM_WAYPOINTS+1)) * (np.array(goal_pos) - np.array(start_pos))
#       default_waypoint_pos = np.array(start_pos) + (float(w+1)/(NUM_WAYPOINTS+1)) * (np.array(goal_pos) - np.array(start_pos))
#       default_waypoints.append(default_waypoint_pos)
#       #add disturbance
#       #if w %2 == 0:
#       waypoint_pos[1] = waypoint_pos[1] + disturbances[w]
#       #else:
#       #  waypoint_pos[1] = waypoint_pos[1] + 0.0
#       waypoints.append(waypoint_pos)
#     waypoints.append(goal_pos)
#     default_waypoints.append(goal_pos)

#     scenario = Scenario(points,waypoints, default_waypoints, disturbances)

#     obstacle_pos = elite_map_config["obstacle_pos"]
#     obstacle_radius = elite_map_config["obstacle_radius"]

#     scenario.setObstacle(obstacle_pos, obstacle_radius)
#     return scenario


def scenario_generate_3points(
    p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, disturbances, elite_map_config
):
    # print "hello!"
    # p1_x = random.uniform(x_lb, x_ub)
    # p1_y = random.uniform(y_lb, y_ub)
    # p2_x = random.uniform(x_lb, x_ub)
    # p2_y = random.uniform(y_lb, y_ub)
    MORSEL_HEIGHT = elite_map_config["morsel_height"]
    NUM_WAYPOINTS = elite_map_config["num_waypoints"]
    END_EFFECTOR_POS = elite_map_config["start_end_effector_pos"]
    points = [(p1_x, p1_y), (p2_x, p2_y), (p3_x, p3_y)]
    goal_pos = np.array([p2_x, p2_y, MORSEL_HEIGHT])
    start_pos = np.array([END_EFFECTOR_POS[0], END_EFFECTOR_POS[1], END_EFFECTOR_POS[2]])

    default_waypoints = []
    waypoints = []
    waypoints.append(start_pos)
    default_waypoints.append(start_pos)
    for w in range(0, NUM_WAYPOINTS):
        waypoint_pos = np.array(start_pos) + (float(w + 1) / (NUM_WAYPOINTS + 1)) * (
            np.array(goal_pos) - np.array(start_pos)
        )
        default_waypoint_pos = np.array(start_pos) + (
            float(w + 1) / (NUM_WAYPOINTS + 1)
        ) * (np.array(goal_pos) - np.array(start_pos))
        default_waypoints.append(default_waypoint_pos)
        # add disturbance
        # if w %2 == 0:
        waypoint_pos[1] = waypoint_pos[1] + disturbances[w]
        # else:
        #  waypoint_pos[1] = waypoint_pos[1] + 0.0
        waypoints.append(waypoint_pos)
    waypoints.append(goal_pos)
    default_waypoints.append(goal_pos)

    scenario = Scenario(points, waypoints, default_waypoints, disturbances)
    return scenario


def scenario_generate_1point_obstacle(
    p1_x, p1_y, disturbances, obstacle_pos, elite_map_config
):
    MORSEL_HEIGHT = elite_map_config["morsel_height"]
    NUM_WAYPOINTS = elite_map_config["num_waypoints"]
    END_EFFECTOR_POS = elite_map_config["start_end_effector_pos"]
    points = [(p1_x, p1_y)]
    goal_pos = np.array([p1_x, p1_y, MORSEL_HEIGHT])
    start_pos = np.array([END_EFFECTOR_POS[0], END_EFFECTOR_POS[1], END_EFFECTOR_POS[2]])
    has_obstacle = elite_map_config["has_obstacle"]
    knows_obstacle = elite_map_config["knows_obstacle"]
    obstacle_radius = elite_map_config["obstacle_radius"]
    obstacle_padding = elite_map_config["obstacle_padding"]

    if has_obstacle is False:
        sys.exit("we should not be here!")

    default_waypoints = []
    waypoints = []
    waypoints.append(start_pos)
    for w in range(0, NUM_WAYPOINTS):
        waypoint_pos = np.array(start_pos) + (float(w + 1) / (NUM_WAYPOINTS + 1)) * (
            np.array(goal_pos) - np.array(start_pos)
        )
        # add disturbance
        # if w %2 == 0:
        # else:
        #  waypoint_pos[1] = waypoint_pos[1] + 0.0
        waypoints.append(waypoint_pos)
    waypoints.append(goal_pos)
    # repair waypoints
    for ii in range(NUM_WAYPOINTS):
        collision_waypoint_pre = waypoints[ii]
        collision_waypoint_pos = waypoints[ii + 1]
        if not (
            line_sphere_intersection(
                collision_waypoint_pre,
                collision_waypoint_pos,
                obstacle_pos,
                obstacle_radius,
                obstacle_padding,
            )
        ):
            continue
        waypoint_pos_fixed_1 = collision_waypoint_pos.copy()
        waypoint_pos_fixed_2 = collision_waypoint_pos.copy()
        counter = 0
        while (
            line_sphere_intersection(
                collision_waypoint_pre,
                waypoint_pos_fixed_1,
                obstacle_pos,
                obstacle_radius,
                obstacle_padding,
            )
            and counter < 100
        ):
            waypoint_pos_fixed_1[1] = waypoint_pos_fixed_1[1] - 0.01
            counter = counter + 1

        counter = 0
        while (
            line_sphere_intersection(
                collision_waypoint_pre,
                waypoint_pos_fixed_2,
                obstacle_pos,
                obstacle_radius,
                obstacle_padding,
            )
            and counter < 100
        ):
            waypoint_pos_fixed_2[1] = waypoint_pos_fixed_2[1] + 0.01
            counter = counter + 1
        # pick waypoints
        d1 = np.linalg.norm(
            waypoint_pos_fixed_1 - collision_waypoint_pre
        ) + np.linalg.norm(goal_pos - waypoint_pos_fixed_1)
        d2 = np.linalg.norm(
            waypoint_pos_fixed_2 - collision_waypoint_pre
        ) + np.linalg.norm(goal_pos - waypoint_pos_fixed_2)
        if d1 <= d2:
            waypoints[ii + 1] = waypoint_pos_fixed_1.copy()
        else:
            waypoints[ii + 1] = waypoint_pos_fixed_2.copy()

    default_waypoints = list(waypoints)

    # apply disturbances
    for w in range(1, NUM_WAYPOINTS + 1):
        waypoints[w][1] = waypoints[w][1] + disturbances[w - 1]

    scenario = Scenario(points, waypoints, default_waypoints, disturbances, max_time=15)
    scenario.setObstacle(
        obstacle_pos,
        obstacle_radius,
        knows_obstacle=knows_obstacle,
        obstacle_padding=obstacle_padding,
    )

    return scenario


def scenario_generate(solution, **scenario_kwargs):
    # print "hello!"
    # p1_x = random.uniform(x_lb, x_ub)
    # p1_y = random.uniform(y_lb, y_ub)
    # p2_x = random.uniform(x_lb, x_ub)
    # p2_y = random.uniform(y_lb, y_ub)
    MORSEL_HEIGHT = scenario_kwargs["morsel_height"]
    num_waypoints = scenario_kwargs["num_waypoints"]
    END_EFFECTOR_POS = scenario_kwargs["start_end_effector_pos"]
    num_goals = scenario_kwargs.get("num_goals", 2)
    points = [(solution[2 * i], solution[2 * i + 1]) for i in range(num_goals)]
    goal_pos = np.array([points[0][0], points[0][1], MORSEL_HEIGHT])
    start_pos = np.array([END_EFFECTOR_POS[0], END_EFFECTOR_POS[1], END_EFFECTOR_POS[2]])

    disturbances = solution[2 * num_goals:]
    waypoints = np.linspace(start_pos, goal_pos, num_waypoints + 2)
    default_waypoints = waypoints.copy()

    # Add noise to the y-coord of the waypoints between the start and the goal
    waypoints[1:-1, 1] += disturbances

    scenario = Scenario(points, waypoints, default_waypoints, disturbances)
    return scenario


def test_scenario_generate():
    random.seed(1)
    np.random.seed(1)
    # method = "shared_auton"

    # initialize scenarios
    sol = [
        0.657062828540802,
        0.1574791520833969,
        0.5200009942054749,
        0.126586173549294472,
        0.01,
        0.02,
        0.0,
        0.0,
        0.0,
    ]

    num_scenarios = 1
    elite_map_config = toml.load("search/config/elite_map/BC_blend.tml")

    scenario = scenario_generate(sol, **elite_map_config)
    scenarios = [copy.deepcopy(scenario) for _ in range(num_scenarios)]

    cc = 1
    process = OpenraveSingle(
        cc, elite_map_config, simulate_user=True, simulate_robot=True
    )
    process.start()

    start_time = time.time()

    while len(scenarios) > 0:
        eval_scenario = scenarios.pop()
        print eval_scenario.waypoints
        print eval_scenario.points
        process.evaluate(eval_scenario)

        print "Objective: {}".format(eval_scenario.getTime())
        print "Measures:"
        for bc_name, bc_func in getmembers(bc_calculate, isfunction):
            print "{}: {}".format(bc_name, bc_func(eval_scenario, None))

    total_time = time.time() - start_time
    print "Total time: " + str(total_time)

    print "all done!!!!!!!!!!!!!!!!!!!!"

    # Delete cmovetraj_*.txt potentially generated by the planner
    for f in glob.glob("cmovetraj_*.txt"):
        os.remove(f)


if __name__ == "__main__":
    test_scenario_generate()
