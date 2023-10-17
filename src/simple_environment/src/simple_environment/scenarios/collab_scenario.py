class CollabScenario:
    def __init__(
        self,
        goals,
        waypoints,
        default_waypoints,
        human_pos,
        max_time=100,
        one_shot=False,
        mdp_human=None,
        obstacles=None,
        human_vel_coeff=0.5,
    ):
        """Scenario for collaborative tasks.

        Args:
            goals (List): (x,y) locations of n goals.
            waypoints (np.ndarray): Array of shape (n_goals, num_waypoints, 3) giving the
                list of waypoints that the human will follow from the starting point to
                each goal.
            default_waypoints (np.ndarray): Similar to waypoints but without added noise.
            human_pos (np.ndarray): Starting position of the human.
            max_time (int): Maximum length of the scenario evaluation. (default 10)
            one_shot (bool): True if the scenario should end after robot reaches one goal.
            mdp_human (MDPHuman): MDP model to use for human policy.
            obstacles (List): (x,y,r) of the obstacles.
            human_vel_coeff (float): Coefficient that is multiplied with human velocity
                (default 0.5).
        """
        self.points = goals
        self.waypoints = waypoints
        self.default_waypoints = default_waypoints
        if waypoints is not None:
            self.disturbances = waypoints - default_waypoints
        else:
            self.disturbances = None
        self.obstacles = obstacles
        self.human_vel_coeff = human_vel_coeff

        self.max_time = max_time
        self.one_shot = one_shot
        self.mdp_human = mdp_human

        self.robot_position = None
        self.robot_trajectory = []
        self.human_position = human_pos
        self.human_trajectory = []
        self.time = -1
        self.repaired = False
        self.repaired_ind = []
        self.emitter_id = -1
        self.has_obstacle = False
        self.knows_obstacle = False
        self.obstacle_pos = -1
        self.obstacle_radius = -1
        self.obstacle_padding = -1
        self.collided = 0
        self.max_wrong_goal_prob = 0
        self.human_wait_time = 0
        self.robot_wait_time = 0

    def setCollided(self):
        self.collided = 1

    def setObstacle(
        self, obstacle_pos, obstacle_radius, knows_obstacle=True, obstacle_padding=0.0
    ):
        self.has_obstacle = True
        self.obstacle_pos = obstacle_pos
        self.knows_obstacle = knows_obstacle
        self.obstacle_radius = obstacle_radius
        self.obstacle_padding = obstacle_padding

    def setEmitterID(self, emitterID):
        self.emitterID = emitterID

    def getEmitterID(self):
        return self.emitterID

    def setRepaired(self, repaired_ind):
        self.repaired = True
        self.repaired_ind = repaired_ind

    def isRepaired(self):
        return self.repaired

    def setTime(self, time):
        self.time = time

    def setRobotPositions(self, robot_positions):
        self.robot_positions = robot_positions

    def add_robot_pos(self, timestamp, robot_pos):
        self.robot_trajectory.append((timestamp, robot_pos))

    def add_human_pos(self, timestamp, human_pos):
        self.human_trajectory.append((timestamp, human_pos))

    def getTime(self):
        return self.time

    def setID(self, id):
        self.id = id

    def getID(self):
        return self.id

    def add_wrong_goal_prob(self, goal_dist, correct_goal):
        for i, gp in enumerate(goal_dist):
            if i != correct_goal:
                if gp > self.max_wrong_goal_prob:
                    self.max_wrong_goal_prob = gp

    def set_human_wait_time(self, human_wait_time):
        self.human_wait_time = human_wait_time

    def set_robot_wait_time(self, robot_wait_time):
        self.robot_wait_time = robot_wait_time
