import numpy
import openravepy
from std_msgs.msg import Float64
import rospy
import threading
from prpy import util
from prpy.util import Watchdog
from prpy.exceptions import PrPyException
from prpy.base.endeffector import EndEffector

class MicoHand(EndEffector):

    def __init__(self, sim, manipulator):
        """ End effector wrapper for the MICO hand.

        This class wraps the MICO end effector. 
        @param sim Whether the hand is simulated
        @param manipulator The manipulator the hand is attached to
        """
        EndEffector.__init__(self, manipulator)

        robot = manipulator.GetRobot()
        env = robot.GetEnv()

        self.simulated = sim

        low_lims, hi_lims = robot.GetDOFLimits()
        self.limits = [low_lims[self.GetIndices()], hi_lims[self.GetIndices()]]

        with env:
            accel_limits = robot.GetDOFAccelerationLimits()
            accel_limits[self.GetIndices()] = 1.
            robot.SetDOFAccelerationLimits(accel_limits)

        if sim:
            robot = manipulator.GetRobot()
            self.controller = robot.AttachController(
                self.GetName(), '', self.GetIndices(), 0, True)

        #store names of finger controllers
        num_dofs = len(self.GetIndices())
        self.velocity_controller_names = ['vel_f' + str(i) + '_controller' for i in range(1,num_dofs+1)]
        self.velocity_topic_names = [controller_name + '/command' for controller_name in self.velocity_controller_names]

        self.velocity_publishers = [rospy.Publisher(topic_name, Float64, queue_size=1) for topic_name in self.velocity_topic_names]
        self.velocity_publisher_lock = threading.Lock()

        self.servo_watchdog = Watchdog(timeout_duration=0.25, handler=self.SendVelocitiesToMico, args=[[0.]*num_dofs])

    def CloneBindings(self, parent):
        super(MicoHand, self).CloneBindings(parent)

        self.simulated = True

    def MoveHand(self, f1, f2, f3=None, timeout=None):
        """ Change the hand preshape. 

        This function blocks until trajectory
        execution finishes. This can be changed by changing the timeout
        parameter to a maximum number of seconds. Pass zero to return
        instantly.

        @param f1 Finger 1 angle
        @param f2 Finger 2 angle
        @param timeout Blocking execution timeout, in seconds
        """

        from openravepy import PlannerStatus

        robot = self.GetParent()

        with robot.GetEnv():
            sp = openravepy.Robot.SaveParameters
            with robot.CreateRobotStateSaver(sp.ActiveDOF):
                robot.SetActiveDOFs(self.GetIndices())
                cspec = robot.GetActiveConfigurationSpecification('linear')
                current_preshape = robot.GetActiveDOFValues()

            # Default any None's to the current DOF values.
            desired_preshape = current_preshape.copy()
            if f1 is not None: desired_preshape[0] = f1
            if f2 is not None: desired_preshape[1] = f2
            if f3 is not None: desired_preshape[2] = f3

        # Create a two waypoint trajectory to the target configuration.
        traj = openravepy.RaveCreateTrajectory(robot.GetEnv(), '')
        traj.Init(cspec)
        traj.Insert(0, current_preshape)
        traj.Insert(1, desired_preshape)

        # Time the trajectory so we can execute it.
        result = openravepy.planningutils.RetimeTrajectory(
            traj, False, 1., 1., 'LinearTrajectoryRetimer')

        if result not in [ PlannerStatus.HasSolution,
                           PlannerStatus.InterruptedWithSolution ]:
            raise PrPyException('Failed timing finger trajectory.')

        # Execute the trajectory.
        return robot.ExecuteTrajectory(traj)
       
    def OpenHand(self, value=0., timeout=None):
        """ Open the hand.

        @param timeout Blocking execution timeout, in seconds
        """
        if self.simulated:
            robot = self.manipulator.GetRobot()
            p = openravepy.KinBody.SaveParameters

            with robot.CreateRobotStateSaver(p.ActiveDOF | p.ActiveManipulator):
                self.manipulator.SetActive()
                robot.task_manipulation.ReleaseFingers()

            if timeout:
                robot.WaitForController(timeout)

            return None
        else:
            num_dofs = len(self.GetIndices())
            if num_dofs == 2:
                return self.MoveHand(f1=value, f2=value, timeout=timeout)
            else:
                return self.MoveHand(f1=value, f2=value, f3=value, timeout=timeout)


    def CloseHand(self, value=0.8, timeout=None):
        """ Close the hand.

        @param timeout Blocking execution timeout, in seconds
        """
        num_dofs = len(self.GetIndices())
        if num_dofs == 2:
            return self.MoveHand(f1=value, f2=value, timeout=timeout)
        else:
            return self.MoveHand(f1=value, f2=value, f3=value, timeout=timeout)

    def CloseHandTight(self, value=1.2, timeout=None):
        """ Close the hand tightly.

        @param timeout Blocking execution timeout, in seconds
        """
        return self.CloseHand(value=value, timeout=timeout)


    def Servo(self, velocities):
        """ Servo with an instantaneous vector of joint velocities.

        @param velocities Instantaneous joint velocities in radians per second
        """
        num_dof = len(self.GetIndices())

        if len(velocities) != num_dof:
            raise ValueError(
                'Incorrect number of joint velocities.'
                ' Expected {:d}; got {:d}.'.format(num_dof, len(velocities)))

        if self.simulated:
            #self.GetRobot().GetController().Reset(0)
            #self.servo_simulator.SetVelocity(velocities)
            #TODO get simulator for this
            dofs_to_set = self.GetDOFValues() + velocities*0.05

            #make sure within limits
            dofs_to_set = numpy.maximum(dofs_to_set, self.limits[0]+1e-8)
            dofs_to_set = numpy.minimum(dofs_to_set, self.limits[1]-1e-8)

            self.SetDOFValues(dofs_to_set)
        else:
            self.SendVelocitiesToMico(velocities)
            #reset watchdog timer
            self.servo_watchdog.reset()


    def SendVelocitiesToMico(self, velocities):
        """ Send velocities to the motor controller.

        Publish individual joint velocities to the velocity publishers.
        @param velocities List of velocities 

        """
        with self.velocity_publisher_lock:
            for velocity_publisher,velocity in zip(self.velocity_publishers, velocities):
                velocity_publisher.publish(velocity)
