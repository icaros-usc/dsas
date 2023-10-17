import numpy, openravepy

class RenderTrajectory:
    """
    Context manager for rendering trajectories in the OpenRAVE viewer. This
    renders a trajectory as a smooth curve. The curve is are removed when the exit
    handler is called.
    @param robot robot executing the trajectory
    @param traj input trajectory
    @param num_samples number of samples to use for interpolation
    @param linewidth The width of the rendered line representing the trajecotry
    @param color interpolated line color
    @param render If true, perform the render
    """
    def __init__(self, robot, traj, num_samples=20, linewidth=2, color=[1, 0, 0, 1], render=True):
        self.env = robot.GetEnv()
        self.robot = robot
        self.handles = list()
        self.render = render

        # Rendering options.
        self.num_samples = num_samples
        self.linewidth = linewidth
        self.color = numpy.array(color, dtype='float')

        # Clone and retime the trajectory.
        self.traj = openravepy.RaveCreateTrajectory(self.env, 'GenericTrajectory')
        self.traj.Clone(traj, 0)
        openravepy.planningutils.RetimeTrajectory(self.traj)

    def __enter__(self):
        
        if not self.render:
            return

        with self.env:
            with self.robot.CreateRobotStateSaver():
                config_spec = self.traj.GetConfigurationSpecification()
                manipulators = self.robot.GetTrajectoryManipulators(self.traj)

                for manipulator in manipulators:
                    arm_indices = manipulator.GetArmIndices()

                    # Skip manipulators that don't have render_offset set.
                    if hasattr(manipulator, "render_offset"):
                        render_offset = manipulator.render_offset

                        if render_offset is None:
                            continue
                    else:
                        render_offset = [0., 0., 0., 1.]

                    # Evenly interpolate joint values throughout the entire trajectory.
                    interpolated_points = list()
                    for t in numpy.linspace(0, self.traj.GetDuration(), self.num_samples):
                        waypoint = self.traj.Sample(t)
                        joint_values = config_spec.ExtractJointValues(waypoint, self.robot, arm_indices)
                        manipulator.SetDOFValues(joint_values)
                        hand_pose = manipulator.GetEndEffectorTransform()
                        render_position = numpy.dot(hand_pose, render_offset)
                        interpolated_points.append(render_position[0:3])

                    # Render a line through the interpolated points.
                    interpolated_points = numpy.array(interpolated_points)
                    if len(interpolated_points) > 0:
                        handle = self.env.drawlinestrip(interpolated_points, self.linewidth, self.color)
                        handle.SetShow(True)
                        self.handles.append(handle)

    def __exit__(self, *exc_info):
        with self.env:
            del self.handles

class RenderPoses(object):
    """
    Render axis at a poses
    @param poses The poses to render
    @param env The OpenRAVE environment
    @param length The length of each axis
    @param render If false, this class does nothing
    """
    def __init__(self, poses, env, length=0.2, render=True):
        self.env = env
        self.poses = poses
        self.length = length
        self.render = render

    def __enter__(self):
        if self.render:
            self.handles = []
            for pose in self.poses:
                self.handles.append(openravepy.misc.DrawAxes(self.env, pose, dist=self.length))

    def __exit__(self, exc_type, exc_value, traceback):
        self.handles = []

class RenderTSRList(RenderPoses):
    """
    Render samples from a list of tsrs. 
    @param tsr_list A list of TSRChain objects (typically output from a call to tsrlibrary)
    @param env The OpenRAVE environment
    @param num_samples The number of samples to render
    @param render If false, this class does nothing
    """
    def __init__(self, tsr_list, env, num_samples=25, length=0.2, render=True):
        import random
        poses = []
        for idx in range(num_samples):
            tsr_chain_idx = random.randint(0, len(tsr_list) - 1)
            tsr_chain = tsr_list[tsr_chain_idx]
            poses.append(tsr_chain.sample())
        RenderPoses.__init__(self, poses, env, length=length, render=render)

class RenderVector(object):
    '''
    Render a vector in an openrave environment
    @param start_pt The start point of the vector
    @param direction The direction of the vector to render
    @param length The length of the rendered vector
    @param env The OpenRAVE environment
    @param render If false, this class does nothing
    '''
    def __init__(self, start_pt, direction, length, env, render=True):
        self.env = env
        self.start_point = start_pt
        self.end_point = start_pt + numpy.array(direction)*length
        self.render = render
        
    def __enter__(self):
        if self.render:
            self.h = self.env.drawarrow(self.start_point, self.end_point)

    def __exit__(self, exc_type, exc_value, traceback):
        self.h = []

