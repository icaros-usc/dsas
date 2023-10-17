import numpy as np
import openravepy
from prpy.base.manipulator import Manipulator
from std_msgs.msg import Float64
import rospy
import threading
from prpy.util import Watchdog


class Mico(Manipulator):
    def __init__(self, sim, iktype=openravepy.IkParameterization.Type.Transform6D):
        """Initialize Mico manipulator.

        Args:
            sim (bool): a boolean variable indicating whether we want to run it in
                simulation or not
            iktype (openravepy.IkParameterization.Type): the type of IK
        """
        Manipulator.__init__(self)

        self.simulated = sim
        self.iktype = iktype

        robot = self.GetRobot()
        env = robot.GetEnv()

        with env:
            dof_indices = self.GetIndices()
            accel_limits = robot.GetDOFAccelerationLimits()
            accel_limits[dof_indices[0]] = 1.65
            accel_limits[dof_indices[1]] = 1.76
            accel_limits[dof_indices[2]] = 1.70
            accel_limits[dof_indices[3]] = 1.80
            accel_limits[dof_indices[4]] = 1.70
            accel_limits[dof_indices[5]] = 1.77
            robot.SetDOFAccelerationLimits(accel_limits)

            low_lims, hi_lims = robot.GetDOFLimits()
            self.limits = [low_lims[dof_indices], hi_lims[dof_indices]]

        # Load or_nlopt_ik as the IK solver. Unfortunately, IKFast doesn't work
        # on the Mico.
        if iktype is not None:
            with env:
                self.iksolver = openravepy.RaveCreateIkSolver(env, "TracIK")
                if self.iksolver is None:
                    raise Exception("Could not create the ik solver")
                set_ik_succeeded = self.SetIKSolver(self.iksolver)
                if not set_ik_succeeded:
                    raise Exception("could not set the ik solver")

        # Load simulation controllers.
        if sim:
            from prpy.simulation import ServoSimulator

            self.controller = robot.AttachController(
                self.GetName(), "", self.GetArmIndices(), 0, True
            )
            self.servo_simulator = ServoSimulator(self, rate=20, watchdog_timeout=0.1)
        else:
            # if not simulation, create publishers for each joint
            self.velocity_controller_names = [
                "vel_j" + str(i) + "_controller" for i in range(1, 7)
            ]
            self.velocity_topic_names = [
                controller_name + "/command"
                for controller_name in self.velocity_controller_names
            ]

            self.velocity_publishers = [
                rospy.Publisher(topic_name, Float64, queue_size=1)
                for topic_name in self.velocity_topic_names
            ]
            self.velocity_publisher_lock = threading.Lock()

            # create watchdog to send zero velocity
            self.servo_watchdog = Watchdog(
                timeout_duration=0.25,
                handler=self.SendVelocitiesToMico,
                args=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            )

    def CloneBindings(self, parent):
        super(Mico, self).CloneBindings(parent)

        self.simulated = True
        self.iktype = parent.iktype

        self.servo_simulator = None

        # TODO: This is broken on nlopt_ik
        # if parent.iktype is not None:
        #     self.iksolver = openravepy.RaveCreateIkSolver(env, "NloptIK")
        #     self.SetIKSolver(self.iksolver)

    def Servo(self, velocities):
        """
        Servo with an instantaneous vector of joint velocities.
        @param velocities: Instantaneous joint velocities in radians per second
        @type velocities: [float] or numpy.array
        """
        num_dof = len(self.GetArmIndices())

        if len(velocities) != num_dof:
            raise ValueError(
                "Incorrect number of joint velocities."
                " Expected {:d}; got {:d}.".format(num_dof, len(velocities))
            )

        if self.simulated:
            # self.GetRobot().GetController().Reset(0)
            # self.servo_simulator.SetVelocity(velocities)
            dofs_to_set = self.GetDOFValues() + velocities * 0.025

            # make sure within limits
            dofs_to_set = np.maximum(dofs_to_set, self.limits[0] + 1e-8)
            dofs_to_set = np.minimum(dofs_to_set, self.limits[1] - 1e-8)

            self.SetDOFValues(dofs_to_set)
        else:
            self.SendVelocitiesToMico(velocities)
            # reset watchdog timer
            self.servo_watchdog.reset()

    def SendVelocitiesToMico(self, velocities):
        """
        Send the velocities to Mico publisher
        @param velocities: Instantaneous joint velocities in radians per second
        @type velocities: [float] or numpy.array
        """
        with self.velocity_publisher_lock:
            for velocity_publisher, velocity in zip(self.velocity_publishers, velocities):
                velocity_publisher.publish(velocity)
