from simple_environment.util.SearchHelper import *
from simple_environment.scenario_generators.scenario_generator import *
from simple_environment.util import bc_calculate
import numpy as np
import sys


class CMA_ME_Algorithm:
    def __init__(
        self,
        mutation_power,
        initial_population,
        population_size,
        num_to_evaluate,
        feature_map,
        trial_name,
        column_names,
        bc_names,
        emitter_type,
        elite_map_config,
    ):
        self.allRecords = pd.DataFrame(columns=column_names)

        self.initial_population = initial_population
        self.num_to_evaluate = num_to_evaluate

        self.individuals_evaluated_total = 0
        self.individuals_dispatched = 0

        self.feature_map = feature_map
        self.mutation_power = mutation_power
        self.population_size = population_size

        self.trial_name = trial_name
        self.bc_names = bc_names

        self.emitter_type = emitter_type
        self.emitters = None

        self.elite_map_config = elite_map_config

    def is_running(self):
        return self.individuals_evaluated_total < self.num_to_evaluate

    def is_blocking(self):
        if self.emitters == None:
            return False

        for i in range(0, len(self.emitters)):
            if self.emitters[i].is_blocking() is False:
                return False
        return True

    def generate_individual(self):
        ind = None
        if self.individuals_dispatched < self.initial_population:
            ind = Individual()
            if self.individuals_evaluated_total < self.initial_population:
                unscaled_params = np.random.normal(0.0, 1.0, num_params)
                ind.param_vector = unscaled_params
            ind.emitter_id = -1
        else:
            if self.emitters == None:
                self.emitters = []
                if self.emitter_type == "rnd":
                    self.emitters += [
                        RandomDirectionEmitter(
                            self.mutation_power,
                            self.population_size,
                            self.feature_map,
                            self.elite_map_config,
                        )
                        for i in range(5)
                    ]
                elif self.emitter_type == "imp":
                    self.emitters += [
                        ImprovementEmitter(
                            self.mutation_power,
                            self.population_size,
                            self.feature_map,
                            self.elite_map_config,
                        )
                        for i in range(5)
                    ]
                else:
                    sys.exit("Error: unknown emitter type. Exiting program.")
                # self.emitters += [OptimizingEmitter(self.mutation_power, self.feature_map) for i in range(1)]

            pos = 0
            emitter = self.emitters[0]
            for i in range(1, len(self.emitters)):
                if self.emitters[i].individuals_released < emitter.individuals_released:
                    emitter = self.emitters[i]
                    pos = i
            ind = emitter.generate_individual()
            ind.emitter_id = pos

        self.individuals_dispatched += 1
        return ind

    def return_evaluated_individual(self, ind):
        ind.setID(self.individuals_evaluated_total)

        self.individuals_evaluated_total += 1

        #        self.all_records.loc[ind.ID] = ["CMA-ME"]+[ind.param_vector]+ind.statsList+list(ind.features)

        if ind.emitter_id == -1:
            self.feature_map.add(ind)
        else:
            self.emitters[ind.emitter_id].return_evaluated_individual(ind)


class ImprovementEmitter:
    def __init__(self, mutation_power, population_size, feature_map, elite_map_config):
        self.population_size = population_size
        self.sigma = mutation_power
        self.individuals_dispatched = 0
        self.individuals_evaluated = 0
        self.individuals_released = 0

        self.parents = []
        self.population = []
        self.feature_map = feature_map
        self.elite_map_config = elite_map_config

        # X_LB = self.elite_map_config["x_b"][0]
        # X_UB = self.elite_map_config["x_b"][1]
        # Y_LB = self.elite_map_config["y_b"][0]
        # Y_UB = self.elite_map_config["y_b"][1]

        self.reset()

    def reset(self):

        self.mutation_power = self.sigma

        NUM_WAYPOINTS = self.elite_map_config["num_waypoints"]
        self.num_points = self.elite_map_config["num_points"]
        self.num_params = self.num_points * 2 + NUM_WAYPOINTS

        if len(self.feature_map.elite_map) == 0:
            X_LB = self.elite_map_config["x_b"][0]
            X_UB = self.elite_map_config["x_b"][1]
            Y_LB = self.elite_map_config["y_b"][0]
            Y_UB = self.elite_map_config["y_b"][1]
            if self.num_points == 1:
                p1_x = (X_LB + X_UB) / 2
                p1_y = (Y_LB + Y_UB) / 2
                self.mean = np.append([p1_x, p1_y], np.zeros(NUM_WAYPOINTS))
            elif self.num_points == 2:
                p1_x = (X_LB + X_UB) / 2
                p1_y = (Y_LB + Y_UB) / 2
                p2_x = (X_LB + X_UB) / 2
                p2_y = (Y_LB + Y_UB) / 2
                self.mean = np.append([p1_x, p1_y, p2_x, p2_y], np.zeros(NUM_WAYPOINTS))
            else:
                p1_x = (X_LB + X_UB) / 2
                p1_y = (Y_LB + Y_UB) / 2
                p2_x = (X_LB + X_UB) / 2
                p2_y = (Y_LB + Y_UB) / 2
                p3_x = (X_LB + X_UB) / 2
                p3_y = (Y_LB + Y_UB) / 2
                self.mean = np.append(
                    [p1_x, p1_y, p2_x, p2_y, p3_x, p3_y], np.zeros(NUM_WAYPOINTS)
                )
        else:
            elite = self.feature_map.get_random_elite()
            self.mean = np.append(np.array(elite.points).flatten(), elite.disturbances)

        # Setup evolution path variables
        self.pc = np.zeros((self.num_params,), dtype=np.float_)
        self.ps = np.zeros((self.num_params,), dtype=np.float_)

        # Setup the covariance matrix
        scale_vector = np.append(
            np.ones(2 * self.num_points), 0.5 * np.ones(NUM_WAYPOINTS)
        )
        self.C = DecompMatrix(self.num_params, scaled=True, scale_vector=scale_vector)

        # Reset the individuals dispatched
        self.individuals_evaluated = 0

    def check_stop(self, parents):
        if self.C.condition_number > 1e14:
            return True

        area = self.mutation_power * math.sqrt(max(self.C.eigenvalues))
        if area < 1e-11:
            return True
        if abs(parents[0].fitness - parents[-1].fitness) < 1e-12:
            return True

        return False

    def is_blocking(self):
        if self.individuals_dispatched > self.population_size:
            return True
        else:
            return False

    # def generate_individual(self):
    #     unscaled_params = np.random.normal(0.0, self.mutation_power, num_params) * np.sqrt(self.C.eigenvalues)
    #     unscaled_params = np.matmul(self.C.eigenbasis, unscaled_params)
    #     unscaled_params = self.mean + np.array(unscaled_params)
    #     ind = Individual()
    #     ind.param_vector = unscaled_params

    #     self.individuals_disbatched += 1

    #     return ind

    def is_off_limits(self, unscaled_params):
        X_LB = self.elite_map_config["x_b"][0]
        X_UB = self.elite_map_config["x_b"][1]
        Y_LB = self.elite_map_config["y_b"][0]
        Y_UB = self.elite_map_config["y_b"][1]
        NUM_WAYPOINTS = self.elite_map_config["num_waypoints"]
        DISTURBANCE_MAX = self.elite_map_config["disturbance_max"]

        if self.num_points == 1:
            p1_x = unscaled_params[0]
            p1_y = unscaled_params[1]
            disturbances = unscaled_params[2:]
            off_limits = False
            if p1_x < X_LB or p1_x > X_UB or p1_y < Y_LB or p1_y > Y_UB:
                off_limits = True
            for w in range(NUM_WAYPOINTS):
                if (
                    disturbances[w] > DISTURBANCE_MAX
                    or disturbances[w] < -DISTURBANCE_MAX
                ):
                    off_limits = True

        elif self.num_points == 2:
            p1_x = unscaled_params[0]
            p1_y = unscaled_params[1]
            p2_x = unscaled_params[2]
            p2_y = unscaled_params[3]
            disturbances = unscaled_params[4:]
            off_limits = False
            if (
                p1_x < X_LB
                or p1_x > X_UB
                or p1_y < Y_LB
                or p1_y > Y_UB
                or p2_x < X_LB
                or p2_x > X_UB
                or p2_y < Y_LB
                or p2_y > Y_UB
            ):
                off_limits = True
            for w in range(NUM_WAYPOINTS):
                if (
                    disturbances[w] > DISTURBANCE_MAX
                    or disturbances[w] < -DISTURBANCE_MAX
                ):
                    off_limits = True
        elif self.num_points == 3:
            p1_x = unscaled_params[0]
            p1_y = unscaled_params[1]
            p2_x = unscaled_params[2]
            p2_y = unscaled_params[3]
            p3_x = unscaled_params[4]
            p3_y = unscaled_params[5]

            disturbances = unscaled_params[6:]
            off_limits = False
            if (
                p1_x < X_LB
                or p1_x > X_UB
                or p1_y < Y_LB
                or p1_y > Y_UB
                or p2_x < X_LB
                or p2_x > X_UB
                or p2_y < Y_LB
                or p2_y > Y_UB
                or p3_x < X_LB
                or p3_x > X_UB
                or p3_y < Y_LB
                or p3_y > Y_UB
            ):
                off_limits = True
            for w in range(NUM_WAYPOINTS):
                if (
                    disturbances[w] > DISTURBANCE_MAX
                    or disturbances[w] < -DISTURBANCE_MAX
                ):
                    off_limits = True
        else:
            sys.error("Unknown number of points!")

        return off_limits

    def generate_individual(self):
        NUM_WAYPOINTS = self.elite_map_config["num_waypoints"]

        self.individuals_dispatched = self.individuals_dispatched + 1
        self.individuals_released = self.individuals_released + 1

        num_params_1 = 2 * self.num_points

        perturbation = np.random.normal(0.0, self.mutation_power, self.num_params)
        unscaled_params = perturbation * np.sqrt(self.C.eigenvalues)
        unscaled_params = np.matmul(self.C.eigenbasis, unscaled_params)
        unscaled_params = self.mean + np.array(unscaled_params)
        disturbances = unscaled_params[num_params_1:]

        while self.is_off_limits(unscaled_params):

            perturbation = np.random.normal(0.0, self.mutation_power, self.num_params)
            unscaled_params = perturbation * np.sqrt(self.C.eigenvalues)
            unscaled_params = np.matmul(self.C.eigenbasis, unscaled_params)
            unscaled_params = self.mean + np.array(unscaled_params)
            disturbances = unscaled_params[num_params_1:]
        if self.num_points == 1:
            p1_x = unscaled_params[1]
            p1_y = unscaled_params[2]
            scenario = scenario_generate_1point(
                p1_x, p1_y, disturbances, self.elite_map_config
            )
        elif self.num_points == 2:
            p1_x = unscaled_params[0]
            p1_y = unscaled_params[1]
            p2_x = unscaled_params[2]
            p2_y = unscaled_params[3]
            scenario = scenario_generate(
                p1_x, p1_y, p2_x, p2_y, disturbances, self.elite_map_config
            )
        elif self.num_points == 3:
            p1_x = unscaled_params[0]
            p1_y = unscaled_params[1]
            p2_x = unscaled_params[2]
            p2_y = unscaled_params[3]
            p3_x = unscaled_params[4]
            p3_y = unscaled_params[5]
            scenario = scenario_generate_3points(
                p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, disturbances, self.elite_map_config
            )
        else:
            sys.error("unknown number of points")

        return scenario

    def return_evaluated_individual(self, scenario):
        self.individuals_evaluated += 1
        NUM_WAYPOINTS = self.elite_map_config["num_waypoints"]

        scenario.fitness = scenario.getTime()

        scenario.features = []
        for bc in self.elite_map_config["Map"]["Features"]:
            get_feature = bc["name"]
            get_feature = getattr(bc_calculate, get_feature)
            feature_value = get_feature(scenario, self.elite_map_config)
            scenario.features.append(feature_value)

        self.population.append(scenario)

        if self.feature_map.add(scenario):
            self.parents.append(scenario)

        if len(self.population) < self.population_size:
            return

        num_parents = len(self.parents)
        needs_restart = num_parents == 0

        # for cur in self.population:
        # Only update if there are parents
        if num_parents > 0:

            # Sort by fitness
            parents = sorted(self.parents, key=lambda x: x.delta)[::-1]
            # print('----', parents[0].fitness)

            # implementation of CMA-ME

            # Create fresh weights for the number of elites found
            weights = [
                math.log(num_parents + 0.5) - math.log(i + 1) for i in range(num_parents)
            ]
            total_weights = sum(weights)
            weights = np.array([w / total_weights for w in weights])

            # print('mean', self.mean)
            # Dynamically update these parameters
            mueff = sum(weights) ** 2 / sum(weights ** 2)
            cc = (4 + mueff / self.num_params) / (
                self.num_params + 4 + 2 * mueff / self.num_params
            )
            cs = (mueff + 2) / (self.num_params + mueff + 5)
            c1 = 2 / ((self.num_params + 1.3) ** 2 + mueff)
            cmu = min(
                1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((self.num_params + 2) ** 2 + mueff)
            )
            damps = (
                1 + 2 * max(0, math.sqrt((mueff - 1) / (self.num_params + 1)) - 1) + cs
            )
            chiN = self.num_params ** 0.5 * (
                1 - 1 / (4 * self.num_params) + 1.0 / (21 * self.num_params ** 2)
            )

            # Recombination of the new mean
            old_mean = self.mean
            new_mean = sum(
                np.append(np.array(ind.points).flatten(), ind.disturbances) * w
                for ind, w in zip(parents, weights)
            )
            self.mean = np.array(new_mean)

            # Update the evolution path
            y = self.mean - old_mean
            z = np.matmul(self.C.invsqrt, y)
            self.ps = (1 - cs) * self.ps + (
                math.sqrt(cs * (2 - cs) * mueff) / self.mutation_power
            ) * z
            left = (
                sum(x ** 2 for x in self.ps)
                / self.num_params
                / (
                    1
                    - (1 - cs) ** (2 * self.individuals_evaluated / self.population_size)
                )
            )
            right = 2 + 4.0 / (self.num_params + 1)
            hsig = 1 if left < right else 0

            self.pc = (1 - cc) * self.pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y

            # Adapt the covariance matrix
            c1a = c1 * (1 - (1 - hsig ** 2) * cc * (2 - cc))
            self.C.C *= 1 - c1a - cmu
            self.C.C += c1 * np.outer(self.pc, self.pc)
            for k, w in enumerate(weights):
                param_vector = np.append(
                    np.array(parents[k].points).flatten(), parents[k].disturbances
                )
                dv = param_vector - old_mean
                self.C.C += w * cmu * np.outer(dv, dv) / (self.mutation_power ** 2)
            # print('C', np.amin(self.C.C))

            # Update the covariance matrix decomposition and inverse
            if self.check_stop(parents):
                needs_restart = True
            else:
                self.C.update_eigensystem()
            # Update sigma
            cn, sum_square_ps = cs / damps, sum(x ** 2 for x in self.ps)
            self.mutation_power *= math.exp(
                min(1, cn * (sum_square_ps / self.num_params - 1) / 2)
            )

        if needs_restart:
            self.reset()

        self.individuals_dispatched = 0
        # Reset the population
        del self.population[:]
        del self.parents[:]


class CMA_ES_Algorithm:
    def __init__(
        self,
        mutation_power,
        population_size,
        num_to_evaluate,
        feature_map,
        trial_name,
        column_names,
        bc_names,
        elite_map_config,
    ):
        self.population_size = population_size
        self.num_parents = self.population_size // 2
        self.feature_map = feature_map
        self.allRecords = pd.DataFrame(columns=column_names)
        self.sigma = mutation_power
        self.mutation_power = mutation_power
        # self.sigma_2 = mutation_power_2
        # self.mutation_power_2 = mutation_power_2
        self.num_to_evaluate = num_to_evaluate
        self.num_of_points = elite_map_config["num_points"]

        self.individuals_evaluated = 0
        self.individuals_evaluated_total = 0
        self.individuals_dispatched = 0

        self.trial_name = trial_name
        self.bc_names = bc_names

        self.elite_map_config = elite_map_config

        X_LB = self.elite_map_config["x_b"][0]
        X_UB = self.elite_map_config["x_b"][1]
        Y_LB = self.elite_map_config["y_b"][0]
        Y_UB = self.elite_map_config["y_b"][1]
        NUM_WAYPOINTS = self.elite_map_config["num_waypoints"]
        self.num_points = self.elite_map_config["num_points"]
        self.num_params = self.num_points * 2 + NUM_WAYPOINTS

        if self.num_points == 1:
            p1_x = (X_LB + X_UB) / 2
            p1_y = (Y_LB + Y_UB) / 2
            self.mean = np.append([p1_x, p1_y], np.zeros(NUM_WAYPOINTS))
        elif self.num_points == 2:
            p1_x = (X_LB + X_UB) / 2
            p1_y = (Y_LB + Y_UB) / 2
            p2_x = (X_LB + X_UB) / 2
            p2_y = (Y_LB + Y_UB) / 2
            self.mean = np.append([p1_x, p1_y, p2_x, p2_y], np.zeros(NUM_WAYPOINTS))
        else:
            p1_x = (X_LB + X_UB) / 2
            p1_y = (Y_LB + Y_UB) / 2
            p2_x = (X_LB + X_UB) / 2
            p2_y = (Y_LB + Y_UB) / 2
            p3_x = (X_LB + X_UB) / 2
            p3_y = (Y_LB + Y_UB) / 2
            self.mean = np.append(
                [p1_x, p1_y, p2_x, p2_y, p3_x, p3_y], np.zeros(NUM_WAYPOINTS)
            )

        self.best = None
        self.population = []

        self.successful_individuals = []

        # Setup recombination weights
        self.weights = [
            math.log(self.num_parents + 0.5) - math.log(i + 1)
            for i in range(self.num_parents)
        ]
        total_weights = sum(self.weights)
        self.weights = np.array([w / total_weights for w in self.weights])
        self.mueff = sum(self.weights) ** 2 / sum(self.weights ** 2)

        # Setup strategy parameters for adaptation
        self.cc = (4 + self.mueff / self.num_params) / (
            self.num_params + 4 + 2 * self.mueff / self.num_params
        )
        self.cs = (self.mueff + 2) / (self.num_params + self.mueff + 5)
        self.c1 = 2 / ((self.num_params + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2
            * (self.mueff - 2 + 1 / self.mueff)
            / ((self.num_params + 2) ** 2 + self.mueff),
        )
        self.damps = (
            1
            + 2 * max(0, math.sqrt((self.mueff - 1) / (self.num_params + 1)) - 1)
            + self.cs
        )
        self.chiN = self.num_params ** 0.5 * (
            1 - 1 / (4 * self.num_params) + 1.0 / (21 * self.num_params ** 2)
        )

        # Setup evolution path variables
        self.pc = np.zeros((self.num_params,), dtype=np.float_)
        self.ps = np.zeros((self.num_params,), dtype=np.float_)

        # Setup the covariance matrix
        scale_vector = np.append(
            np.ones(2 * self.num_points), 0.5 * np.ones(NUM_WAYPOINTS)
        )
        self.C = DecompMatrix(self.num_params, scaled=True, scale_vector=scale_vector)

    def reset(self):
        self.mutation_power = self.sigma

        if self.best:
            # if False:
            self.mean = np.append(
                np.array(self.best.points).flatten(), self.best.disturbances
            )
        else:
            X_LB = self.elite_map_config["x_b"][0]
            X_UB = self.elite_map_config["x_b"][1]
            Y_LB = self.elite_map_config["y_b"][0]
            Y_UB = self.elite_map_config["y_b"][1]
            if self.num_points == 1:
                p1_x = (X_LB + X_UB) / 2
                p1_y = (Y_LB + Y_UB) / 2
                self.mean = np.append([p1_x, p1_y], np.zeros(NUM_WAYPOINTS))
            elif self.num_points == 2:
                p1_x = (X_LB + X_UB) / 2
                p1_y = (Y_LB + Y_UB) / 2
                p2_x = (X_LB + X_UB) / 2
                p2_y = (Y_LB + Y_UB) / 2
                self.mean = np.append([p1_x, p1_y, p2_x, p2_y], np.zeros(NUM_WAYPOINTS))
            else:
                p1_x = (X_LB + X_UB) / 2
                p1_y = (Y_LB + Y_UB) / 2
                p2_x = (X_LB + X_UB) / 2
                p2_y = (Y_LB + Y_UB) / 2
                p3_x = (X_LB + X_UB) / 2
                p3_y = (Y_LB + Y_UB) / 2
                self.mean = np.append(
                    [p1_x, p1_y, p2_x, p2_y, p3_x, p3_y], np.zeros(NUM_WAYPOINTS)
                )

        # Setup evolution path variables
        self.pc = np.zeros((self.num_params,), dtype=np.float_)
        self.ps = np.zeros((self.num_params,), dtype=np.float_)

        # Setup the covariance matrix
        NUM_WAYPOINTS = self.elite_map_config["num_waypoints"]
        scale_vector = np.append(
            np.ones(2 * self.num_points), 0.5 * np.ones(NUM_WAYPOINTS)
        )
        self.C = DecompMatrix(self.num_params, scaled=True, scale_vector=scale_vector)

        # Reset individual counts
        self.individuals_evaluated = 0

        self.population_size *= 2
        # self.population_size = np.min(self.population_size, self.num_to_evaluate - self.individuals_evaluated_total)
        self.num_parents = self.population_size // 2

    def check_stop(self, parents):
        if self.C.condition_number > 1e14:
            print("large condition number")
            return True

        area = self.mutation_power * math.sqrt(max(self.C.eigenvalues))
        if area < 1e-11:
            print("small area")
            return True
        if abs(parents[0].fitness - parents[-1].fitness) < 1e-12:
            print("flat fitness!")
            print("num of parents:" + str(len(parents)))
            print("best fitness:" + str(parents[0].fitness))
            print("worst fitness:" + str(parents[-1].fitness))
            return True

        return False

    def is_blocking(self):
        if self.individuals_dispatched > self.population_size:
            return True
        else:
            return False

    def is_running(self):
        return self.individuals_evaluated_total < self.num_to_evaluate

    def is_off_limits(self, unscaled_params):
        X_LB = self.elite_map_config["x_b"][0]
        X_UB = self.elite_map_config["x_b"][1]
        Y_LB = self.elite_map_config["y_b"][0]
        Y_UB = self.elite_map_config["y_b"][1]
        NUM_WAYPOINTS = self.elite_map_config["num_waypoints"]
        DISTURBANCE_MAX = self.elite_map_config["disturbance_max"]

        if self.num_points == 1:
            p1_x = unscaled_params[0]
            p1_y = unscaled_params[1]
            disturbances = unscaled_params[2:]
            off_limits = False
            if p1_x < X_LB or p1_x > X_UB or p1_y < Y_LB or p1_y > Y_UB:
                off_limits = True
            for w in range(NUM_WAYPOINTS):
                if (
                    disturbances[w] > DISTURBANCE_MAX
                    or disturbances[w] < -DISTURBANCE_MAX
                ):
                    off_limits = True

        elif self.num_points == 2:
            p1_x = unscaled_params[0]
            p1_y = unscaled_params[1]
            p2_x = unscaled_params[2]
            p2_y = unscaled_params[3]
            disturbances = unscaled_params[4:]
            off_limits = False
            if (
                p1_x < X_LB
                or p1_x > X_UB
                or p1_y < Y_LB
                or p1_y > Y_UB
                or p2_x < X_LB
                or p2_x > X_UB
                or p2_y < Y_LB
                or p2_y > Y_UB
            ):
                off_limits = True
            for w in range(NUM_WAYPOINTS):
                if (
                    disturbances[w] > DISTURBANCE_MAX
                    or disturbances[w] < -DISTURBANCE_MAX
                ):
                    off_limits = True
        elif self.num_points == 3:
            p1_x = unscaled_params[0]
            p1_y = unscaled_params[1]
            p2_x = unscaled_params[2]
            p2_y = unscaled_params[3]
            p3_x = unscaled_params[4]
            p3_y = unscaled_params[5]

            disturbances = unscaled_params[6:]
            off_limits = False
            if (
                p1_x < X_LB
                or p1_x > X_UB
                or p1_y < Y_LB
                or p1_y > Y_UB
                or p2_x < X_LB
                or p2_x > X_UB
                or p2_y < Y_LB
                or p2_y > Y_UB
                or p3_x < X_LB
                or p3_x > X_UB
                or p3_y < Y_LB
                or p3_y > Y_UB
            ):
                off_limits = True
            for w in range(NUM_WAYPOINTS):
                if (
                    disturbances[w] > DISTURBANCE_MAX
                    or disturbances[w] < -DISTURBANCE_MAX
                ):
                    off_limits = True
        else:
            sys.error("Unknown number of points!")

        return off_limits

    def generate_individual(self):
        NUM_WAYPOINTS = self.elite_map_config["num_waypoints"]

        self.individuals_dispatched = self.individuals_dispatched + 1

        num_params_1 = 2 * self.num_points

        perturbation = np.random.normal(0.0, self.mutation_power, self.num_params)
        unscaled_params = perturbation * np.sqrt(self.C.eigenvalues)
        unscaled_params = np.matmul(self.C.eigenbasis, unscaled_params)
        unscaled_params = self.mean + np.array(unscaled_params)
        disturbances = unscaled_params[num_params_1:]

        while self.is_off_limits(unscaled_params):

            perturbation = np.random.normal(0.0, self.mutation_power, self.num_params)
            unscaled_params = perturbation * np.sqrt(self.C.eigenvalues)
            unscaled_params = np.matmul(self.C.eigenbasis, unscaled_params)
            unscaled_params = self.mean + np.array(unscaled_params)
            disturbances = unscaled_params[num_params_1:]
        if self.num_points == 1:
            p1_x = unscaled_params[1]
            p1_y = unscaled_params[2]
            scenario = scenario_generate_1point(
                p1_x, p1_y, disturbances, self.elite_map_config
            )
        elif self.num_points == 2:
            p1_x = unscaled_params[0]
            p1_y = unscaled_params[1]
            p2_x = unscaled_params[2]
            p2_y = unscaled_params[3]
            scenario = scenario_generate(
                p1_x, p1_y, p2_x, p2_y, disturbances, self.elite_map_config
            )
        elif self.num_points == 3:
            p1_x = unscaled_params[0]
            p1_y = unscaled_params[1]
            p2_x = unscaled_params[2]
            p2_y = unscaled_params[3]
            p3_x = unscaled_params[4]
            p3_y = unscaled_params[5]
            scenario = scenario_generate_3points(
                p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, disturbances, self.elite_map_config
            )
        else:
            sys.error("unknown number of points")

        return scenario

    def return_evaluated_individual(self, scenario):
        scenario.setID(self.individuals_evaluated_total)
        self.individuals_evaluated += 1
        self.individuals_evaluated_total += 1
        NUM_WAYPOINTS = self.elite_map_config["num_waypoints"]

        scenario.fitness = scenario.getTime()

        scenario.features = []
        for bc in self.elite_map_config["Map"]["Features"]:
            get_feature = bc["name"]
            get_feature = getattr(bc_calculate, get_feature)
            feature_value = get_feature(scenario, self.elite_map_config)
            scenario.features.append(feature_value)

        self.feature_map.add(scenario)
        self.population.append(scenario)

        if self.best == None or self.best.fitness < scenario.fitness:
            self.best = scenario
        if len(self.population) > self.population_size:
            sys.error("we should not be here!")
        elif len(self.population) == self.population_size:
            # for cur in self.population:

            # Sort by fitness
            parents = sorted(self.population, key=lambda x: x.fitness)[::-1]
            parents = parents[: self.num_parents]
            # print('----', parents[0].fitness)

            # Recombination of the new mean
            old_mean = self.mean
            new_mean = sum(
                np.append(np.array(ind.points).flatten(), ind.disturbances) * w
                for ind, w in zip(parents, self.weights)
            )
            self.mean = np.array(new_mean)
            # implementation of CMA-ME

            # print('mean', self.mean)

            # Update the evolution path
            y = self.mean - old_mean
            z = np.matmul(self.C.invsqrt, y)
            self.ps = (1 - self.cs) * self.ps + (
                math.sqrt(self.cs * (2 - self.cs) * self.mueff) / self.mutation_power
            ) * z
            left = (
                sum(x ** 2 for x in self.ps)
                / self.num_params
                / (
                    1
                    - (1 - self.cs)
                    ** (2 * self.individuals_evaluated / self.population_size)
                )
            )
            right = 2 + 4.0 / (self.num_params + 1)
            hsig = 1 if left < right else 0

            self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(
                self.cc * (2 - self.cc) * self.mueff
            ) * y

            # Adapt the covariance matrix
            c1a = self.c1 * (1 - (1 - hsig ** 2) * self.cc * (2 - self.cc))
            self.C.C *= 1 - c1a - self.cmu
            self.C.C += self.c1 * np.outer(self.pc, self.pc)
            for k, w in enumerate(self.weights):
                param_vector = np.append(
                    np.array(parents[k].points).flatten(), parents[k].disturbances
                )
                dv = param_vector - old_mean
                self.C.C += w * self.cmu * np.outer(dv, dv) / (self.mutation_power ** 2)
            # print('C', np.amin(self.C.C))

            # Updated the covariance matrix decomposition and inverse
            # Update the covariance matrix decomposition and inverse
            self.C.update_eigensystem()
            # Update sigma
            cn, sum_square_ps = self.cs / self.damps, sum(x ** 2 for x in self.ps)
            self.mutation_power *= math.exp(
                min(1, cn * (sum_square_ps / self.num_params - 1) / 2)
            )

            # Check if we need a restart then restart.
            if self.check_stop(parents):
                print("restarting!!")
                self.reset()

            self.individuals_dispatched = 0
            # Reset the population
            del self.population[:]


class MapElitesAlgorithm:
    def __init__(
        self,
        mutation_power_pos,
        mutation_power_disturb,
        initial_population,
        num_to_evaluate,
        feature_map,
        trial_name,
        column_names,
        bc_names,
        elite_map_config,
    ):
        self.mutation_power_pos = mutation_power_pos
        self.mutation_power_disturb = mutation_power_disturb
        self.initial_population = initial_population
        self.num_to_evaluate = num_to_evaluate
        self.trial_name = trial_name
        self.individuals_evaluated = 0
        self.individuals_evaluated_total = self.individuals_evaluated
        self.individuals_dispatched = 0
        self.feature_map = feature_map
        self.allRecords = pd.DataFrame(columns=column_names)
        self.bc_names = bc_names
        self.elite_map_config = elite_map_config
        self.num_points = elite_map_config["num_points"]
        self.has_obstacle = elite_map_config["has_obstacle"]

        if self.has_obstacle == True:
            self.obstacle_pos_x = self.elite_map_config["obstacle_pos_x"]
            self.obstacle_pos_z = self.elite_map_config["obstacle_pos_z"]
            self.p1_x = self.elite_map_config["p1_x"]

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def is_blocking(self):
        if (
            self.individuals_dispatched >= self.initial_population
            and self.individuals_evaluated == 0
        ):
            return True
        else:
            return False

    def generate_individual(self):
        self.individuals_dispatched = self.individuals_dispatched + 1
        X_LB = self.elite_map_config["x_b"][0]
        X_UB = self.elite_map_config["x_b"][1]
        Y_LB = self.elite_map_config["y_b"][0]
        Y_UB = self.elite_map_config["y_b"][1]
        NUM_WAYPOINTS = self.elite_map_config["num_waypoints"]
        DISTURBANCE_MAX = self.elite_map_config["disturbance_max"]
        if self.individuals_dispatched <= self.initial_population:
            disturbances = np.random.uniform(
                -DISTURBANCE_MAX, DISTURBANCE_MAX, NUM_WAYPOINTS
            )

            if self.num_points == 1 and self.has_obstacle == True:
                p1_y = random.uniform(Y_LB, Y_UB)
                obstacle_pos_y = random.uniform(Y_LB, Y_UB)
                obstacle_pos = [self.obstacle_pos_x, obstacle_pos_y, self.obstacle_pos_z]
                scenario = scenario_generate_1point_obstacle(
                    self.p1_x, p1_y, disturbances, obstacle_pos, self.elite_map_config
                )
            elif self.num_points == 2:
                p1_x = random.uniform(X_LB, X_UB)
                p1_y = random.uniform(Y_LB, Y_UB)
                p2_x = random.uniform(X_LB, X_UB)
                p2_y = random.uniform(Y_LB, Y_UB)
                scenario = scenario_generate(
                    p1_x, p1_y, p2_x, p2_y, disturbances, self.elite_map_config
                )
            elif self.num_points == 3:
                p1_x = random.uniform(X_LB, X_UB)
                p1_y = random.uniform(Y_LB, Y_UB)
                p2_x = random.uniform(X_LB, X_UB)
                p2_y = random.uniform(Y_LB, Y_UB)
                p3_x = random.uniform(X_LB, X_UB)
                p3_y = random.uniform(Y_LB, Y_UB)
                scenario = scenario_generate_3points(
                    p1_x,
                    p1_y,
                    p2_x,
                    p2_y,
                    p3_x,
                    p3_y,
                    disturbances,
                    self.elite_map_config,
                )
            else:
                sys.error("unknown number of points")
        else:
            parent = self.feature_map.get_random_elite()
            exceeded_limits = True
            while exceeded_limits:
                new_disturbances = np.zeros(NUM_WAYPOINTS)
                for w in range(NUM_WAYPOINTS):
                    new_disturbances[w] = (
                        parent.disturbances[w] + self.mutation_power_disturb * gaussian()
                    )
                exceeded_limits = False
                for w in range(NUM_WAYPOINTS):
                    if (
                        new_disturbances[w] > DISTURBANCE_MAX
                        or new_disturbances[w] < -DISTURBANCE_MAX
                    ):
                        exceeded_limits = True
            if self.num_points == 1 and self.has_obstacle == True:
                p1_y = parent.points[0][1] + self.mutation_power_pos * gaussian()
                obstacle_pos_y = (
                    parent.obstacle_pos[1] + self.mutation_power_pos * gaussian()
                )

                while (
                    p1_y < Y_LB
                    or p1_y > Y_UB
                    or obstacle_pos_y < Y_LB
                    or obstacle_pos_y > Y_UB
                ):
                    p1_y = parent.points[0][1] + self.mutation_power_pos * gaussian()
                    obstacle_pos_y = (
                        parent.obstacle_pos[1] + self.mutation_power_pos * gaussian()
                    )

                obstacle_pos = [self.obstacle_pos_x, obstacle_pos_y, self.obstacle_pos_z]

                scenario = scenario_generate_1point_obstacle(
                    self.p1_x, p1_y, new_disturbances, obstacle_pos, self.elite_map_config
                )
            elif self.num_points == 2:
                p1_x = parent.points[0][0] + self.mutation_power_pos * gaussian()
                p1_y = parent.points[0][1] + self.mutation_power_pos * gaussian()
                p2_x = parent.points[1][0] + self.mutation_power_pos * gaussian()
                p2_y = parent.points[1][1] + self.mutation_power_pos * gaussian()  #
                while (
                    p1_x < X_LB
                    or p1_x > X_UB
                    or p1_y < Y_LB
                    or p1_y > Y_UB
                    or p2_x < X_LB
                    or p2_x > X_UB
                    or p2_y < Y_LB
                    or p2_y > Y_UB
                ):
                    p1_x = parent.points[0][0] + self.mutation_power_pos * gaussian()
                    p1_y = parent.points[0][1] + self.mutation_power_pos * gaussian()
                    p2_x = parent.points[1][0] + self.mutation_power_pos * gaussian()
                    p2_y = parent.points[1][1] + self.mutation_power_pos * gaussian()  #
                scenario = scenario_generate(
                    p1_x, p1_y, p2_x, p2_y, new_disturbances, self.elite_map_config
                )
            elif self.num_points == 3:
                p1_x = parent.points[0][0] + self.mutation_power_pos * gaussian()
                p1_y = parent.points[0][1] + self.mutation_power_pos * gaussian()
                p2_x = parent.points[1][0] + self.mutation_power_pos * gaussian()
                p2_y = parent.points[1][1] + self.mutation_power_pos * gaussian()
                p3_x = parent.points[2][0] + self.mutation_power_pos * gaussian()
                p3_y = parent.points[2][1] + self.mutation_power_pos * gaussian()  #

                while (
                    p1_x < X_LB
                    or p1_x > X_UB
                    or p1_y < Y_LB
                    or p1_y > Y_UB
                    or p2_x < X_LB
                    or p2_x > X_UB
                    or p2_y < Y_LB
                    or p2_y > Y_UB
                    or p3_x < X_LB
                    or p3_x > X_UB
                    or p3_y < Y_LB
                    or p3_y > Y_UB
                ):
                    p1_x = parent.points[0][0] + self.mutation_power_pos * gaussian()
                    p1_y = parent.points[0][1] + self.mutation_power_pos * gaussian()
                    p2_x = parent.points[1][0] + self.mutation_power_pos * gaussian()
                    p2_y = parent.points[1][1] + self.mutation_power_pos * gaussian()  #
                    p3_x = parent.points[2][0] + self.mutation_power_pos * gaussian()
                    p3_y = parent.points[2][1] + self.mutation_power_pos * gaussian()  #
                scenario = scenario_generate_3points(
                    p1_x,
                    p1_y,
                    p2_x,
                    p2_y,
                    p3_x,
                    p3_y,
                    new_disturbances,
                    self.elite_map_config,
                )
            else:
                sys.error("Unknown number of points!")

        return scenario

    def return_evaluated_individual(self, scenario):

        scenario.setID(self.individuals_evaluated)
        self.individuals_evaluated += 1
        self.individuals_evaluated_total += 1
        scenario.features = []
        for bc in self.elite_map_config["Map"]["Features"]:
            get_feature = bc["name"]
            get_feature = getattr(bc_calculate, get_feature)
            feature_value = get_feature(scenario, self.elite_map_config)
            scenario.features.append(feature_value)

        scenario.fitness = scenario.getTime()
        self.feature_map.add(scenario)


class IsolineDDAlgorithm:
    def __init__(
        self,
        mutation_power_pos_1,
        mutation_power_pos_2,
        mutation_power_disturb_1,
        mutation_power_disturb_2,
        initial_population,
        num_to_evaluate,
        feature_map,
        trial_name,
        column_names,
        bc_names,
        elite_map_config,
    ):
        self.mutation_power_pos_1 = mutation_power_pos_1
        self.mutation_power_disturb_1 = mutation_power_disturb_1

        self.mutation_power_pos_2 = mutation_power_pos_2
        self.mutation_power_disturb_2 = mutation_power_disturb_2

        self.initial_population = initial_population
        self.num_to_evaluate = num_to_evaluate
        self.trial_name = trial_name
        self.individuals_evaluated = 0
        self.individuals_evaluated_total = self.individuals_evaluated
        self.individuals_dispatched = 0
        self.feature_map = feature_map
        self.allRecords = pd.DataFrame(columns=column_names)
        self.bc_names = bc_names
        self.elite_map_config = elite_map_config
        self.num_points = elite_map_config["num_points"]
        self.has_obstacle = elite_map_config["has_obstacle"]

        if self.has_obstacle == True:
            self.obstacle_pos_x = self.elite_map_config["obstacle_pos_x"]
            self.obstacle_pos_z = self.elite_map_config["obstacle_pos_z"]
            self.p1_x = self.elite_map_config["p1_x"]

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def is_blocking(self):
        if (
            self.individuals_dispatched >= self.initial_population
            and self.individuals_evaluated == 0
        ):
            return True
        else:
            return False

    def generate_individual(self):
        self.individuals_dispatched = self.individuals_dispatched + 1
        X_LB = self.elite_map_config["x_b"][0]
        X_UB = self.elite_map_config["x_b"][1]
        Y_LB = self.elite_map_config["y_b"][0]
        Y_UB = self.elite_map_config["y_b"][1]
        NUM_WAYPOINTS = self.elite_map_config["num_waypoints"]
        DISTURBANCE_MAX = self.elite_map_config["disturbance_max"]
        if self.individuals_dispatched <= self.initial_population:
            disturbances = np.random.uniform(
                -DISTURBANCE_MAX, DISTURBANCE_MAX, NUM_WAYPOINTS
            )

            if self.num_points == 2:
                p1_x = random.uniform(X_LB, X_UB)
                p1_y = random.uniform(Y_LB, Y_UB)
                p2_x = random.uniform(X_LB, X_UB)
                p2_y = random.uniform(Y_LB, Y_UB)
                scenario = scenario_generate(
                    p1_x, p1_y, p2_x, p2_y, disturbances, self.elite_map_config
                )
            elif self.num_points == 3:
                p1_x = random.uniform(X_LB, X_UB)
                p1_y = random.uniform(Y_LB, Y_UB)
                p2_x = random.uniform(X_LB, X_UB)
                p2_y = random.uniform(Y_LB, Y_UB)
                p3_x = random.uniform(X_LB, X_UB)
                p3_y = random.uniform(Y_LB, Y_UB)
                scenario = scenario_generate_3points(
                    p1_x,
                    p1_y,
                    p2_x,
                    p2_y,
                    p3_x,
                    p3_y,
                    disturbances,
                    self.elite_map_config,
                )
            else:
                sys.error("unknown number of points")
        else:
            parent_1 = self.feature_map.get_random_elite()
            parent_2 = self.feature_map.get_random_elite()
            exceeded_limits = True
            while exceeded_limits:
                new_disturbances = np.zeros(NUM_WAYPOINTS)
                for w in range(NUM_WAYPOINTS):
                    new_disturbances[w] = (
                        parent_1.disturbances[w]
                        + self.mutation_power_disturb_1 * gaussian()
                        + (parent_2.disturbances[w] - parent_1.disturbances[w])
                        * self.mutation_power_disturb_2
                        * gaussian()
                    )
                exceeded_limits = False
                for w in range(NUM_WAYPOINTS):
                    if (
                        new_disturbances[w] > DISTURBANCE_MAX
                        or new_disturbances[w] < -DISTURBANCE_MAX
                    ):
                        exceeded_limits = True
            if self.num_points == 2:
                parent1_p1_x = parent_1.points[0][0]
                parent1_p1_y = parent_1.points[0][1]
                parent1_p2_x = parent_1.points[1][0]
                parent1_p2_y = parent_1.points[1][1]

                parent2_p1_x = parent_2.points[0][0]
                parent2_p1_y = parent_2.points[0][1]
                parent2_p2_x = parent_2.points[1][0]
                parent2_p2_y = parent_2.points[1][1]

                p1_x = (
                    parent1_p1_x
                    + self.mutation_power_pos_1 * gaussian()
                    + (parent2_p1_x - parent1_p1_x)
                    * self.mutation_power_pos_2
                    * gaussian()
                )
                p1_y = (
                    parent1_p1_y
                    + self.mutation_power_pos_1 * gaussian()
                    + (parent2_p1_y - parent1_p1_y)
                    * self.mutation_power_pos_2
                    * gaussian()
                )
                p2_x = (
                    parent1_p2_x
                    + self.mutation_power_pos_1 * gaussian()
                    + (parent2_p2_x - parent1_p2_x)
                    * self.mutation_power_pos_2
                    * gaussian()
                )
                p2_y = (
                    parent1_p2_y
                    + self.mutation_power_pos_1 * gaussian()
                    + (parent2_p2_y - parent1_p2_y)
                    * self.mutation_power_pos_2
                    * gaussian()
                )

                while (
                    p1_x < X_LB
                    or p1_x > X_UB
                    or p1_y < Y_LB
                    or p1_y > Y_UB
                    or p2_x < X_LB
                    or p2_x > X_UB
                    or p2_y < Y_LB
                    or p2_y > Y_UB
                ):
                    p1_x = (
                        parent1_p1_x
                        + self.mutation_power_pos_1 * gaussian()
                        + (parent2_p1_x - parent1_p1_x)
                        * self.mutation_power_pos_2
                        * gaussian()
                    )
                    p1_y = (
                        parent1_p1_y
                        + self.mutation_power_pos_1 * gaussian()
                        + (parent2_p1_y - parent1_p1_y)
                        * self.mutation_power_pos_2
                        * gaussian()
                    )
                    p2_x = (
                        parent1_p2_x
                        + self.mutation_power_pos_1 * gaussian()
                        + (parent2_p2_x - parent1_p2_x)
                        * self.mutation_power_pos_2
                        * gaussian()
                    )
                    p2_y = (
                        parent1_p2_y
                        + self.mutation_power_pos_1 * gaussian()
                        + (parent2_p2_y - parent1_p2_y)
                        * self.mutation_power_pos_2
                        * gaussian()
                    )
                scenario = scenario_generate(
                    p1_x, p1_y, p2_x, p2_y, new_disturbances, self.elite_map_config
                )

            elif self.num_points == 3:
                parent1_p1_x = parent_1.points[0][0]
                parent1_p1_y = parent_1.points[0][1]
                parent1_p2_x = parent_1.points[1][0]
                parent1_p2_y = parent_1.points[1][1]
                parent1_p3_x = parent_1.points[2][0]
                parent1_p3_y = parent_1.points[2][1]

                parent2_p1_x = parent_2.points[0][0]
                parent2_p1_y = parent_2.points[0][1]
                parent2_p2_x = parent_2.points[1][0]
                parent2_p2_y = parent_2.points[1][1]
                parent2_p3_x = parent_2.points[2][0]
                parent2_p3_y = parent_2.points[2][1]

                p1_x = (
                    parent1_p1_x
                    + self.mutation_power_pos_1 * gaussian()
                    + (parent2_p1_x - parent1_p1_x)
                    * self.mutation_power_pos_2
                    * gaussian()
                )
                p1_y = (
                    parent1_p1_y
                    + self.mutation_power_pos_1 * gaussian()
                    + (parent2_p1_y - parent1_p1_y)
                    * self.mutation_power_pos_2
                    * gaussian()
                )
                p2_x = (
                    parent1_p2_x
                    + self.mutation_power_pos_1 * gaussian()
                    + (parent2_p2_x - parent1_p2_x)
                    * self.mutation_power_pos_2
                    * gaussian()
                )
                p2_y = (
                    parent1_p2_y
                    + self.mutation_power_pos_1 * gaussian()
                    + (parent2_p2_y - parent1_p2_y)
                    * self.mutation_power_pos_2
                    * gaussian()
                )
                p3_x = (
                    parent1_p3_x
                    + self.mutation_power_pos_1 * gaussian()
                    + (parent2_p3_x - parent1_p3_x)
                    * self.mutation_power_pos_2
                    * gaussian()
                )
                p3_y = (
                    parent1_p3_y
                    + self.mutation_power_pos_1 * gaussian()
                    + (parent2_p3_y - parent1_p3_y)
                    * self.mutation_power_pos_2
                    * gaussian()
                )

                while (
                    p1_x < X_LB
                    or p1_x > X_UB
                    or p1_y < Y_LB
                    or p1_y > Y_UB
                    or p2_x < X_LB
                    or p2_x > X_UB
                    or p2_y < Y_LB
                    or p2_y > Y_UB
                    or p3_x < X_LB
                    or p3_x > X_UB
                    or p3_y < Y_LB
                    or p3_y > Y_UB
                ):
                    p1_x = (
                        parent1_p1_x
                        + self.mutation_power_pos_1 * gaussian()
                        + (parent2_p1_x - parent1_p1_x)
                        * self.mutation_power_pos_2
                        * gaussian()
                    )
                    p1_y = (
                        parent1_p1_y
                        + self.mutation_power_pos_1 * gaussian()
                        + (parent2_p1_y - parent1_p1_y)
                        * self.mutation_power_pos_2
                        * gaussian()
                    )
                    p2_x = (
                        parent1_p2_x
                        + self.mutation_power_pos_1 * gaussian()
                        + (parent2_p2_x - parent1_p2_x)
                        * self.mutation_power_pos_2
                        * gaussian()
                    )
                    p2_y = (
                        parent1_p2_y
                        + self.mutation_power_pos_1 * gaussian()
                        + (parent2_p2_y - parent1_p2_y)
                        * self.mutation_power_pos_2
                        * gaussian()
                    )
                    p3_x = (
                        parent1_p3_x
                        + self.mutation_power_pos_1 * gaussian()
                        + (parent2_p3_x - parent1_p3_x)
                        * self.mutation_power_pos_2
                        * gaussian()
                    )
                    p3_y = (
                        parent1_p3_y
                        + self.mutation_power_pos_1 * gaussian()
                        + (parent2_p3_y - parent1_p3_y)
                        * self.mutation_power_pos_2
                        * gaussian()
                    )
                scenario = scenario_generate_3points(
                    p1_x,
                    p1_y,
                    p2_x,
                    p2_y,
                    p3_x,
                    p3_y,
                    new_disturbances,
                    self.elite_map_config,
                )
            else:
                sys.error("Unknown number of points!")

        return scenario

    def return_evaluated_individual(self, scenario):

        scenario.setID(self.individuals_evaluated)
        self.individuals_evaluated += 1
        self.individuals_evaluated_total += 1
        scenario.features = []
        for bc in self.elite_map_config["Map"]["Features"]:
            get_feature = bc["name"]
            get_feature = getattr(bc_calculate, get_feature)
            feature_value = get_feature(scenario, self.elite_map_config)
            scenario.features.append(feature_value)

        scenario.fitness = scenario.getTime()
        self.feature_map.add(scenario)


class RandomGenerator:
    def __init__(
        self,
        num_to_evaluate,
        feature_map,
        trial_name,
        column_names,
        bc_names,
        elite_map_config,
    ):
        self.num_to_evaluate = num_to_evaluate
        self.trial_name = trial_name
        self.individuals_evaluated = 0
        self.individuals_evaluated_total = 0
        self.feature_map = feature_map
        self.allRecords = pd.DataFrame(columns=column_names)
        self.bc_names = bc_names
        self.elite_map_config = elite_map_config
        self.num_points = elite_map_config["num_points"]

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def generate_individual(self):
        X_LB = self.elite_map_config["x_b"][0]
        X_UB = self.elite_map_config["x_b"][1]
        Y_LB = self.elite_map_config["y_b"][0]
        Y_UB = self.elite_map_config["y_b"][1]
        DISTURBANCE_MAX = self.elite_map_config["disturbance_max"]
        NUM_WAYPOINTS = self.elite_map_config["num_waypoints"]
        disturbances = np.random.uniform(-DISTURBANCE_MAX, DISTURBANCE_MAX, NUM_WAYPOINTS)
        if self.num_points == 2:
            p1_x = random.uniform(X_LB, X_UB)
            p1_y = random.uniform(Y_LB, Y_UB)
            p2_x = random.uniform(X_LB, X_UB)
            p2_y = random.uniform(Y_LB, Y_UB)
            scenario = scenario_generate(
                p1_x, p1_y, p2_x, p2_y, disturbances, self.elite_map_config
            )
        elif self.num_points == 3:
            p1_x = random.uniform(X_LB, X_UB)
            p1_y = random.uniform(Y_LB, Y_UB)
            p2_x = random.uniform(X_LB, X_UB)
            p2_y = random.uniform(Y_LB, Y_UB)
            p3_x = random.uniform(X_LB, X_UB)
            p3_y = random.uniform(Y_LB, Y_UB)
            scenario = scenario_generate_3points(
                p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, disturbances, self.elite_map_config
            )
        else:
            sys.error("Unknown number of points!")

        return scenario

    def is_blocking(self):
        return False

    def return_evaluated_individual(self, scenario):

        scenario.setID(self.individuals_evaluated)
        self.individuals_evaluated += 1
        self.individuals_evaluated_total += 1
        scenario.features = []
        for bc in self.elite_map_config["Map"]["Features"]:
            get_feature = bc["name"]
            get_feature = getattr(bc_calculate, get_feature)
            feature_value = get_feature(scenario, self.elite_map_config)
            scenario.features.append(feature_value)

        scenario.fitness = scenario.getTime()
        self.feature_map.add(scenario)
