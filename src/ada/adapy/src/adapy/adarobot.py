PACKAGE = 'adapy'
import logging
import prpy
from prpy import Cloned
from prpy.base.robot import Robot
import os

logger = logging.getLogger(PACKAGE)

CONFIGURATIONS_PATH = 'config/configurations.yaml'
TSR_PATHS = [
    'config/glass_grasp_tsr.yaml',
    'config/glass_move_tsr.yaml',
]

class ADARobot(Robot):
    def __init__(self, sim):
        """ Create an ADARobot.

        @param sim: Set to True to run the robot in simulation.
        """
        from mico import Mico
        from micohand import MicoHand
        from util import AdaPyException, find_adapy_resource

        self.simulated = sim
        self.talker_simulated = sim

        # We need to hard-code the name. Otherwise, it defaults to
        # "mico-modified".
        Robot.__init__(self, robot_name='ada')

        # Convenience attributes for accessing self components.
        self.arm = self.GetManipulator('Mico')
        self.arm.hand = self.arm.GetEndEffector()
        self.manipulators = [ self.arm ]

        # Bind robot-specific subclasses.
        prpy.bind_subclass(self.arm, Mico, sim=sim)
        prpy.bind_subclass(self.arm.hand, MicoHand, sim=sim,
                           manipulator=self.arm)

        # TODO: Load an IdealController controller in simulation.

        # Load default named configurations from YAML.
        self.configurations.add_group('arm', self.arm.GetIndices())
        try:
            self.configurations.load_yaml(
                find_adapy_resource(CONFIGURATIONS_PATH))
        except IOError as e:
            logger.warning('Failed loading named configurations from "%s": %s',
                           configurations_path, e.message)

        # If in sim, set the robot DOFs to not be in collision
        if sim:
            inds,dofs = self.configurations.get_configuration('home')
            self.SetDOFValues(values=dofs,dofindices=inds)

        # Load default TSRs from YAML.
        for tsr_path_relative in TSR_PATHS:
            try:
                tsr_path = find_adapy_resource(tsr_path_relative)
                self.tsrlibrary.load_yaml(tsr_path)
            except IOError as e:
                raise ValueError(
                    'Failed loading TSRs from "{:s}": {:s}'.format(
                        tsr_path, e.message))

        # Create ros_control clients for execution on real hardware.
        if not sim:
            from .controller_client import (ControllerManagerClient,
                                            JointStateClient)
            from .trajectory_client import FollowJointTrajectoryClient

            self._trajectory_joint_names = [
                self.GetJointFromDOFIndex(i).GetName()
                for i in xrange(self.GetDOF())
            ]

            self._jointstate_client = JointStateClient(
                self, topic_name='/joint_states'
            )
            self._controller_client = ControllerManagerClient(
                ns='/controller_manager'
            )
            self._velocity_joint_mode_controller = self._controller_client.request(
                'velocity_joint_mode_controller'
            )
            self._velocity_joint_mode_controller.switch()

            self._trajectory_switcher = self._controller_client.request(
                'traj_controller'
            )
            self._trajectory_client = FollowJointTrajectoryClient(
                ns='/traj_controller/follow_joint_trajectory'
            )

        # Initialize the default planning pipeline.
        from prpy.planning import Sequence, Ranked, FirstSupported
        from prpy.planning import (
            BiRRTPlanner,
            CBiRRTPlanner,
            IKPlanner,
            GreedyIKPlanner,
            NamedPlanner,
            SBPLPlanner,
            SnapPlanner,
            IKPlanner,
            TSRPlanner,
            VectorFieldPlanner
        )

        self.snap_planner = SnapPlanner()
        self.greedyik_planner = GreedyIKPlanner()
        self.cbirrt_planner = CBiRRTPlanner()
        self.vectorfield_planner = VectorFieldPlanner()

        # Hide TrajOpt logging.
        os.environ.setdefault('TRAJOPT_LOG_THRESH', 'WARN')

        # Trajectory optimizer.
        #try:
        #    from or_trajopt import TrajoptPlanner
        #    self.trajopt_planner = TrajoptPlanner()
        #except ImportError:
        #    self.trajopt_planner = None
        #    logger.warning('Failed creating TrajoptPlanner. Is the or_trajopt'
        #                   ' package in your workspace and built?')


        ik_planners = Sequence(
            self.snap_planner,
         #   self.trajopt_planner,
        )
        planner_for_ik = FirstSupported(
            ik_planners,
            NamedPlanner(delegate_planner=ik_planners)
        )
        self.ik_planner = IKPlanner(delegate_planner=planner_for_ik)

        actual_planner = Sequence(
            self.snap_planner,
            self.ik_planner,
     #       self.trajopt_planner,
            self.vectorfield_planner,
            #self.greedyik_planner,
            self.cbirrt_planner
        )
        self.planner = FirstSupported(
            actual_planner,
            NamedPlanner(delegate_planner=actual_planner)
        )

        self.simplifier = None
        #from prpy.planning.retimer import ParabolicSmoother, HauserParabolicSmoother
#        self.smoother = Sequence(
#            ParabolicSmoother(),
#            HauserParabolicSmoother()
#        )
        from prpy.planning.retimer import HauserParabolicSmoother
        self.smoother = HauserParabolicSmoother(do_blend=True, do_shortcut=True)
        self.retimer = HauserParabolicSmoother(do_blend=True, do_shortcut=False)

        from prpy.action import ActionLibrary
        self.actions = ActionLibrary()

        if not self.talker_simulated:
            # Initialize herbpy ROS Node
            import rospy
            if not rospy.core.is_initialized():
                rospy.init_node('adapy', anonymous=True)
                logger.debug('Started ROS node with name "%s".', rospy.get_name())

            #import talker.msg
            #from actionlib import SimpleActionClient
            #self._say_action_client = SimpleActionClient('say', talker.msg.SayAction)


    def CloneBindings(self, parent):
        """ Copies fields added in Python to allow OpenRave environment cloning.

        This is an internal function that you should not call directly.
        See https://github.com/personalrobotics/prpy/blob/master/README.md#cloning-bound-subclasses
        for the rationale for this function.

        @param parent: The ADARobot to copy fields from.
        """
        super(ADARobot, self).CloneBindings(parent)

        self.arm = Cloned(parent.arm)
        self.manipulators = [ self.arm ]
        self.planner = parent.planner

    def ExecuteTrajectory(self, traj, defer=False, timeout=None, switch=True,
                          unswitch=None, **kwargs):
        """ Executes a time trajectory on the robot.

        This function directly executes a timed OpenRAVE trajectory on the
        robot. If you have a geometric path, such as those returned by a
        geometric motion planner, you should first time the path using
        PostProcessPath. Alternatively, you could use the ExecutePath helper
        function to time and execute the path in one function call.

        If timeout = None (the default), this function does not return until
        execution has finished. Termination occurs if the trajectory is
        successfully executed or if a fault occurs (in this case, an exception
        will be raised). If timeout is a float (including timeout = 0), this
        function will return None once the timeout has ellapsed, even if the
        trajectory is still being executed.

        NOTE: We suggest that you either use timeout=None or defer=True. If
        trajectory execution times out, there is no way to tell whether
        execution was successful or not. Other values of timeout are only
        supported for legacy reasons.

        This function returns the trajectory that was actually executed on the
        robot, including controller error. If this is not available, the input
        trajectory will be returned instead.

        If switch = True, this function switches to the ros_control controllers
        necessary to execute traj. If unswitch = True, it also undoes this
        switch after the trajectory has finished executing. If unswitch is
        unspecified, it defaults to the same value as switch.

        @param traj: timed trajectory to execute
        @type  traj: openravepy.Trajectory
        @param defer: execute asynchronously and return a trajectory Future
        @type  defer: bool
        @param timeout: maximum time to wait for execution to finish
        @type  timeout: float or None
        @param switch: switch to the controllers necessary to execute traj
        @type  switch: bool
        @param unswitch: revert the controllers after executing traj
        @type  unswitch: bool or None
        @return trajectory executed on the robot
        @rtype  openravepy.Trajectory or TrajectoryFuture
        """
        from .util import or_to_ros_trajectory, pad_ros_trajectory
        from rospy import Time
        from prpy import exceptions

        if self.simulated:
            return Robot.ExecuteTrajectory(self, traj, defer=defer, timeout=timeout,
                                           **kwargs)


        # Check that the current configuration of the robot matches the
        # initial configuration specified by the trajectory.
#        if not prpy.util.IsAtTrajectoryStart(self, traj):
#            raise exceptions.TrajectoryNotExecutable(
#                'Trajectory started from different configuration than robot.')

        # If there was only one waypoint, at this point we are done!
        if traj.GetNumWaypoints() == 1:
            return traj

        if unswitch is None:
            unswitch = switch

        traj_msg = or_to_ros_trajectory(self, traj)

        # sometimes points are removed from message
        # ensure this trajectory is useful to send before continuing
        if len(traj_msg.points) < 1:
            return traj


        # The trajectory_controller expects the full set of DOFs to be
        # specified in every trajectory. We pad the trajectory by adding
        # constant values for any missing joints.
        #print traj_msg
        #print "\n\n\n\n\n\n"
        #print traj.serialize()

        #from IPython import embed
        #embed()
        with self.GetEnv():
            pad_ros_trajectory(self, traj_msg, self._trajectory_joint_names)

        if switch:
            self._trajectory_switcher.switch()

        traj_future = self._trajectory_client.execute(traj_msg)

        if unswitch:
            traj_future.add_done_callback(
                lambda _: self._trajectory_switcher.unswitch
            )

        #for i in range(0,len(traj_msg.points)):
        #    print traj_msg.points[i].positions
        #    print '\n'


        from trajectory_client import TrajectoryExecutionFailed
        if defer:
            return traj_future
        else:
            try:
                traj_future.result(timeout)
                return traj
            except TrajectoryExecutionFailed as e:
                logger.exception('Trajectory execution failed.')


    def Say(self, words, block=True):
        """ Speak 'words' through text-to-speech.

        On the live robot this will use the talker action service;
        in simulation it will use espeak. By default the function will block
        until the synthesized speech has finished playing.

        @param words: the string to say
        @type  words: string
        @param block: whether the function blocks until the utterance is done
        @type  block: bool
        """
        if self.talker_simulated:
            import subprocess
            try:
                proc = subprocess.Popen(['espeak', '-s', '160', '"{0}"'.format(words)])
                if block:
                    proc.wait()
            except OSError as e:
                logger.error('Unable to speak. Make sure "espeak" is installed locally.\n%s' % str(e))
        else:
            import talker.msg
            goal = talker.msg.SayGoal(text=words)
            self._say_action_client.send_goal(goal)
            if block:
                self._say_action_client.wait_for_result()



    def SwitchToTeleopController(self):
        """ Set the robot into velocity control mode for teleop.

        Note that this does not by itself implement any teleop controls, it
        only puts the robot into velocity control mode to allow teleop
        controllers to send velocity commands at interactive rates rather than
        having to go through the trajectory/planning pipeline.
        """
        for arm in self.manipulators:
            #turn on velocity controllers for each arm
            for controller_name in arm.velocity_controller_names:
                velocity_controller = self._controller_client.request(controller_name)
                velocity_controller.switch()
            #turn on velocity controller for each hand
            for controller_name in arm.hand.velocity_controller_names:
                velocity_controller = self._controller_client.request(controller_name)
                velocity_controller.switch()


    #def SwitchToTrajController(self):
    #    self._velocity_joint_mode_controller.switch()
