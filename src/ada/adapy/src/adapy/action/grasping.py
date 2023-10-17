import logging, openravepy, prpy
from prpy.action import ActionMethod
from prpy.planning.base import PlanningError
from contextlib import contextmanager

logger = logging.getLogger('adapy')

@ActionMethod
def Grasp(robot, obj, manip=None, tsrlist=None, render=True, **kw_args):
    """ Have a robot grasp an object.

    @param robot The robot performing the grasp
    @param obj The object to grasp
    @param manip The manipulator to perform the grasp with
       (if None active manipulator is used)
    @param tsrlist A list of TSRCHain objects to use for planning to grasp pose
       (if None, the 'grasp' tsr from tsrlibrary is used)
    @param render Render tsr samples
    """
    if manip is None:
        manip = robot.GetActiveManipulator()

    # Get the grasp tsr
    with robot.GetEnv():
        if tsrlist is None:
            tsrlist = robot.tsrlibrary(obj, 'grasp')

    # Plan to the grasp
    with prpy.viz.RenderTSRList(tsrlist, robot.GetEnv(), render=render):
        manip.PlanToTSR(tsrlist, execute=True)

    manip.hand.CloseHand()

    # Manipulator must be active for grab to work properly
    p = openravepy.KinBody.SaveParameters
    with robot.CreateRobotStateSaver(p.ActiveManipulator):
        robot.SetActiveManipulator(manip)
        robot.Grab(obj)
