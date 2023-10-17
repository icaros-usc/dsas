class Scenario:
    def __init__(self, points, waypoints, default_waypoints, disturbances, max_time=10):
        self.points = points
        self.max_time = max_time
        self.waypoints = waypoints
        self.default_waypoints = default_waypoints
        self.disturbances = disturbances
        self.robot_positions = -1
        self.time = -1
        self.repaired = False
        self.repaired_ind = []
        self.emitterID = -1
        self.has_obstacle = False
        self.knows_obstacle = False
        self.obstacle_pos = -1
        self.obstacle_radius = -1
        self.obstacle_padding = -1
        self.collided = 0

        self.robot_trajectory = []

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

    def getTime(self):
        return self.time

    def setID(self, id):
        self.id = id

    def getID(self):
        return self.id

    def add_robot_pos(self, timestamp, robot_pos):
        self.robot_trajectory.append((timestamp, robot_pos))
