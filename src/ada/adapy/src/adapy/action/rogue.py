import logging, openravepy, prpy
from prpy.action import ActionMethod
from prpy.planning.base import PlanningError
from contextlib import contextmanager
from prpy.util import FindCatkinResource, GetPointFrom
import numpy, cPickle, time, os.path

logger = logging.getLogger('adapy')

@ActionMethod
def PointAt(robot, focus, manip=None, render=False):
    """ Make a robot point at a 3d position or object.

    Note that this is identical to Point except that the focus can be an object
    or 3d position.

    @param robot The robot performing the point
    @param focus The 3-D coordinate in space or object
                 that is being pointed at
    @param manip The manipulator to perform the point with.
    @param render Render tsr samples during planning
    """
    pointing_coord = GetPointFrom(focus)
    return Point(robot, pointing_coord, manip, render)

@ActionMethod
def PresentAt(robot, focus, manip=None, render=True):
    """ Make a robot 'present' at a 3d position or object.

    Note that this is identical to Present except that the focus can be an
    object or 3d position.

    @param robot The robot performing the presentation
    @param focus The 3-D coordinate in space or object that
                 is being presented
    @param manip The manipulator to perform the presentation with.
    @param render Render tsr samples during planning
    """
    presenting_coord = GetPointFrom(focus)
    return Present(robot, presenting_coord, manip, render)

@ActionMethod
def SweepAt(robot, start, end, manip=None, margin=0.3, render=True):
    """ Make a robot 'sweep' its arm from one object or position to another.

    Note that this is identical to Sweep except start and end can be designated
    by objects.

    @param robot The robot performing the sweep
    @param start The object or 3-D position that marks the start
    @param end The object or 3-D position that marks the end
    @param manip The manipulator to perform the sweep
    @param margin The distance between the start object and the hand,
                  so the vertical space between the hand and objects.
                  This must be enough to clear the objects themselves.
    @param render Render tsr samples during planning
    """
    start_coord = GetPointFrom(start)
    end_coord = GetPointFrom(end)
    return Sweep(robot, start_coord, end_coord, manip, margin, render)

def Point(robot, focus, manip=None, render=False):
    """ Make a robot point at a 3d position.

    @param robot The robot performing the point
    @param focus The 3-D coordinate in space
    @param manip The manipulator to perform the point with.
                 This must be the right arm
    @param render Render tsr samples during planning
    """
    if manip is None:
        manip = robot.GetActiveManipulator()

    focus_trans = numpy.eye(4, dtype='float')
    focus_trans[0:3, 3] = focus

    with robot.GetEnv():
        point_tsr = robot.tsrlibrary(None, 'point', focus_trans, manip)

    p = openravepy.KinBody.SaveParameters
    with robot.CreateRobotStateSaver(p.ActiveManipulator | p.ActiveDOF):
        robot.SetActiveManipulator(manip)
        robot.SetActiveDOFs(manip.GetArmIndices())
        with prpy.viz.RenderTSRList(point_tsr, robot.GetEnv(), render=render):
            robot.PlanToTSR(point_tsr, execute=True)

    robot.arm.hand.CloseHand()

def Present(robot, focus, manip=None, render=True):
    """ Make a robot 'present' at a 3d position.

    The 'present' gesture is like pointing, except all the fingers are open
    and the palm is held up.

    @param robot The robot performing the presentation
    @param focus The 3-D coordinate in space or object that
                 is being presented
    @param manip The manipulator to perform the presentation with.
                 This must be the right arm.
    @param render Render tsr samples during planning
    """
    if manip is None:
        manip = robot.GetActiveManipulator()

    focus_trans = numpy.eye(4, dtype='float')
    focus_trans[0:3, 3] = focus

    with robot.GetEnv():
        present_tsr = robot.tsrlibrary(None, 'present', focus_trans, manip)

    p = openravepy.KinBody.SaveParameters
    with robot.CreateRobotStateSaver(p.ActiveManipulator | p.ActiveDOF):
        robot.SetActiveManipulator(manip)
        robot.SetActiveDOFs(manip.GetArmIndices())
        with prpy.viz.RenderTSRList(present_tsr, robot.GetEnv(), render=render):
            robot.PlanToTSR(present_tsr, execute=True)

    #should be closehand() but that doesnt work
    robot.arm.hand.CloseHand()

def Sweep(robot, start_coords, end_coords, manip=None, margin=0.3, render=True):
    """ Make a robot sweep its arm from one 3d position to another.

    Roughly, this instructs the robot to start by pointing at one position and
    then transition to pointing at another.

    @param robot The robot performing the sweep
    @param start The object or 3-d position that marks the start
    @param end The object of 3-d position that marks the end
    @param manip The manipulator to perform the sweep
    @param margin The distance between the start object and the hand,
                  so the vertical space between the hand and objects.
                  This must be enough to clear the objects themselves.
    @param render Render tsr samples during planning
    """

    if manip is None:
        manip = robot.GetActiveManipulator()

    #ee_offset : such that the hand, not wrist, is above the object
    #hand_pose : places the hand above the start location
    if manip.GetName() == 'Mico':
        hand = manip.hand
        ee_offset = -0.15
        hand_pose = numpy.array([[ 0,  0, -1, (start_coords[0]+ee_offset)],
                                 [ 0,  1,  0,  start_coords[1]],
                                 [ 1,  0,  0, (start_coords[2]+margin)],
                                 [ 0,  0,  0, 1]])
    else:
        raise prpy.exceptions.PrPyException('Manipulator does not have an \
                 associated hand')

    end_trans = numpy.eye(4, dtype='float')
    end_trans[0:3, 3] = end_coords
    hand.CloseHand()

    q = openravepy.KinBody.SaveParameters
    with robot.CreateRobotStateSaver(q.ActiveManipulator | q.ActiveDOF):
        robot.SetActiveManipulator(manip)
        robot.SetActiveDOFs(manip.GetArmIndices())
        manip.PlanToEndEffectorPose(hand_pose)

    #TSR to sweep to end position
    with robot.GetEnv():
        sweep_tsr = robot.tsrlibrary(None, 'sweep', end_trans, manip)

    p = openravepy.KinBody.SaveParameters
    with robot.CreateRobotStateSaver(p.ActiveManipulator | p.ActiveDOF):
        robot.SetActiveManipulator(manip)
        robot.SetActiveDOFs(manip.GetArmIndices())
        with prpy.viz.RenderTSRList(sweep_tsr, robot.GetEnv(), render=render):
             robot.PlanToTSR(sweep_tsr, execute=True)

@ActionMethod
def Exhibit(robot, obj, manip=None, distance=0.1, wait=2, render=True):
    """ Make a robot 'exhibit' an object (pick it up and put it back down).

    @param robot The robot performing the exhibit
    @param obj The object being exhibited
    @param manip The maniplator to perform the exhibit
    @param distance The distance the object will be lifted up
    @param wait The amount of time the object will be held up in seconds
    @param render Render tsr samples during planning
    """

    with robot.GetEnv():
        if manip is None:
            manip = robot.GetActiveManipulator()
    robot.Grasp(obj)

    p = openravepy.KinBody.SaveParameters
    with robot.CreateRobotStateSaver(p.ActiveManipulator | p.ActiveDOF):
        robot.SetActiveManipulator(manip)
        robot.SetActiveDOFs(manip.GetArmIndices())

        #Lift the object
        lift_tsr = robot.tsrlibrary(obj, 'lift', manip, distance=distance)
        with prpy.viz.RenderTSRList(lift_tsr, robot.GetEnv(), render=render):
            robot.PlanToTSR(lift_tsr, execute=True)

        #Wait for 'time'
        time.sleep(wait)

        #'Unlift' the object, so place it back down
        unlift_tsr = robot.tsrlibrary(obj, 'lift', manip, distance=-distance)
        with prpy.viz.RenderTSRList(unlift_tsr, robot.GetEnv(), render=render):
            robot.PlanToTSR(unlift_tsr, execute=True)
