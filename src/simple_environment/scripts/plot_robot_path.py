import adapy, argparse, logging, numpy, os, openravepy, prpy, rospy
from catkin.find_in_workspaces import find_in_workspaces
from std_msgs.msg import String

from prpy.planning.base import PlanningError
from adapy.adarobot import ADARobot

import random
import toml

from simple_environment.scenario_generators.scenario_generator import *

from simple_environment.actions.bite_serving import BiteServing
from simple_environment.actions.bypassable_action import ActionException

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

project_name = 'simple_environment'
logger = logging.getLogger(project_name)

start_config = np.array([-2.38, -0.1959051, -0.34364456 , -0.688872349, 0,
        3.55386329])


URDF_PATH = 'package://ada_description/robots/mico.urdf'
SRDF_PATH = 'package://ada_description/robots/mico.srdf'



class OpenraveSingle():


    def __init__(self,id,elite_map_config,simulate_user = True):
      self.id = id
      self.elite_map_config = elite_map_config
      self.simulate_user = simulate_user
      #self.messageClass = messageClass

   

    def start(self):
      self.init_environment("starting_process")

    def evaluate(self, scenario):
       evalTime = self.evaluate(scenario)
       #print(scenario.robot_positions)  

       #print(self.message.etScenario())
       scenario.setEvalTime(evalTime)
       scenario.setRobotPositions(scenario.robot_positions)



    def setup(self, sim=False, viewer=None, debug=True):
        # load the robot and environment for meal serving

        # find the openrave environment file
        data_base_path = find_in_workspaces(
            search_dirs=['share'],
            project=project_name,
            path='data',
            first_match_only=True)
        if len(data_base_path) == 0:
            raise Exception('Unable to find environment path. Did you source devel/setup.bash?')
        env_path = os.path.join(data_base_path[0], 'environments', 'table.env.xml')

        # Initialize logging
        if debug:
            openravepy.RaveInitialize(True, level=openravepy.DebugLevel.Debug)
        else:
            openravepy.RaveInitialize(True, level=openravepy.DebugLevel.Info)
        openravepy.misc.InitOpenRAVELogging()
        prpy.logger.initialize_logging()

        # Load the environment and robot
        self.env, self.robot = adapy.initialize(attach_viewer=viewer, sim=sim, env_path=env_path)

        # Set the active manipulator on the robot
        self.robot.arm.SetActive()


        #else:
        START_ROBOT_POS = self.elite_map_config["start_robot_pos"]
        robot_pose = numpy.array([[1., 0., 0., START_ROBOT_POS[0]],
                                  [0., 1., 0., START_ROBOT_POS[1]],
                                  [0., 0., 1., 0.795],
                                  [0., 0., 0., 1.]])

        with self.env:
            self.robot.SetTransform(robot_pose)


        point_1 = [-2.38,-0.1959051,-0.34364456,-0.68887235,0.,3.55386329, 0.15,0.15]
        point_2 = [-2.23414425,-0.20351582,-0.29449167,-1.18136489,0.64684566,2.95822193,0.15,0.15]
        self.robot.SetDOFValues(point_1)


        # Use or_urdf to load ADA from URDF and SRDF.

    # Use or_urdf to load ADA from URDF and SRDF.
        with self.env:
            or_urdf = openravepy.RaveCreateModule(self.env, 'urdf')
            ada_name = or_urdf.SendCommand(
                'load {:s} {:s}'.format(URDF_PATH, SRDF_PATH))

        self.robot2 = self.env.GetRobot(ada_name)
        if self.robot2 is None:
            raise AdaPyException('Failed loading ADA with or_urdf.')

        # Bind AdaPy-specific functionality on the robot.
        prpy.bind_subclass(self.robot2, ADARobot, sim=sim)
        # Set the active manipulator on the robot
        self.robot2.arm.SetActive()


        #else:
        START_ROBOT_POS = self.elite_map_config["start_robot_pos"]
        robot_pose = numpy.array([[1., 0., 0., START_ROBOT_POS[0]],
                                  [0., 1., 0., START_ROBOT_POS[1]],
                                  [0., 0., 1., 0.795],
                                  [0., 0., 0., 1.]])

        with self.env:
            self.robot2.SetTransform(robot_pose)

        links = self.robot.GetLinks()
        for i in range(0,len(links)):
            link = links[i]
            geometries = link.GetGeometries()
            for g in range(0,len(geometries)):
                geometry = geometries[g]
               # print("hello!")
               # geometry.SetVisible(False)
                geometry.SetTransparency(0.5)




        self.robot2.SetDOFValues(point_2)


        self.ResetTrial()
        from IPython import embed
        embed()




    def ResetTrial(self):
      #self.env.GetViewer().Reset()
      # set the robot to the start configuration for next trial
      logger.info('Resetting Robot')
      if self.robot.simulated:
        #  indices, values = self.robot.configurations.get_configuration('ada_meal_scenario_servingConfiguration')
          self.robot.arm.SetDOFValues(start_config)
          values = self.robot.GetDOFValues()
          values[6] = 0.15
          values[7] = 0.15
          self.robot.SetDOFValues(values)

      else:
        #first try to plan to serving
        try:
          self.robot.PlanToNamedConfiguration('ada_meal_scenario_servingConfiguration', execute=True)
        except PlanningError, e:
          logger.info('Failed to plan to start config')
          #if it doesn't work, unload controllers


    def setup_trial_recording(record_next_trial, file_directory_user):
        # creates user directory if we will be recording
        if record_next_trial and not os.path.exists(file_directory_user):
            os.makedirs(file_directory_user)

    def evaluate(self, scenario, method = 'shared_auton_always', blend_level  = 'agressive'):
      times = []
      for i in range(0,1):
        try:
          # robot set initial position
          self.robot.arm.SetDOFValues(start_config)

          #print(self.robot.arm.GetEndEffectorTransform())
          self.manip = self.robot.GetActiveManipulator()
          action = BiteServing()



          action.execute(self.manip, self.env, method=method, blend_level = blend_level, ui_device = 'kinova', state_pub=self.state_pub,detection_sim = True, record_trial=False, scenario = scenario, plot_trajectory = False, simulate_user = self.simulate_user)


        except ActionException, e:
          logger.info('Failed to complete bite serving: %s' % str(e))


        self.ResetTrial()
        times.append(scenario.getTime())

      print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Min time is: " + str(min(times))
      return min(times)




    def init_environment(self,rospyname):
        self.state_pub = rospy.Publisher('ada_tasks',String, queue_size=10)
        rospy.init_node(rospyname, anonymous=True)


        self.setup(sim=True, viewer='qtcoin', debug=True)

         

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', help='path of BC config file',required=True)
    opt = parser.parse_args()

    elite_map_config=toml.load(opt.config)

    waypoints = [np.array([0.41, 0.05, 0.95]),
     np.array([0.44972744, 0.05800751, 0.92666667]),
     np.array([0.48945489, 0.0980931 , 0.90333333]),
     np.array([0.52918233, 0.10728433, 0.88      ]),
     np.array([0.56890977, 0.12846371, 0.85666667]),
     np.array([0.60863722, 0.1464666 , 0.83333333]),
     np.array([0.64836466, 0.18314335, 0.81      ])]
    default_waypoints = [np.array([0.41, 0.05, 0.95]),
     np.array([0.44972744, 0.07219056, 0.92666667]),
     np.array([0.48945489, 0.09438112, 0.90333333]),
     np.array([0.52918233, 0.11657167, 0.88      ]),
     np.array([0.56890977, 0.13876223, 0.85666667]),
     np.array([0.60863722, 0.16095279, 0.83333333]),
     np.array([0.64836466, 0.18314335, 0.81      ])]
    p1_x = 0.578106677158656
    p1_y = 0.18618648097156895
    p2_x = 0.6483646624970655
    p2_y = 0.18314334709926336

    disturbances = [-0.014183051902421799,
      0.003711986861540669,
     -0.009287342108160079,
     -0.010298516499131317,
     -0.014486189467392812]

    scenario = scenario_generate(p1_x,p1_y,p2_x,p2_y, disturbances, elite_map_config)


    opt = parser.parse_args()
    random.seed(1)
           

    process = OpenraveSingle(1,elite_map_config)
    process.start()
    scenario.max_time = 10

