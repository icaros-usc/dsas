import numpy
from prpy.tsr.tsrlibrary import TSRFactory
from prpy.tsr.tsr import TSR, TSRChain

@TSRFactory('ada', 'fuze_bottle', 'grasp')
def fuze_grasp(robot, fuze, manip=None):
    '''
    @param robot The robot performing the grasp
    @param fuze The bottle to grasp
    @param manip The manipulator to perform the grasp, if None
       the active manipulator on the robot is used
    '''

    if manip is None:
        manip_idx = robot.GetActiveManipulatorIndex()
    else:
        manip.SetActive()
        manip_idx = manip.GetRobot().GetActiveManipulatorIndex()

    T0_w = fuze.GetTransform()

    ee_to_palm = 0.15
    palm_to_bottle_center = 0.045
    total_offset = ee_to_palm + palm_to_bottle_center
    Tw_e = numpy.array([[ 0., 0., -1., -total_offset],
                        [ 0., 1.,  0., 0.],
                        [ 1., 0.,  0., 0.08], #bottle height
                        [ 0., 0.,  0., 1.]])


    Bw = numpy.zeros((6,2))
    Bw[2,:] = [0.0, 0.02]  # Allow a little vertical movement
    Bw[5,:] = [-numpy.pi/2, numpy.pi/2]  # Allow any orientation

    grasp_tsr = TSR(T0_w = T0_w, Tw_e = Tw_e, Bw = Bw, manip = manip_idx)
    grasp_chain = TSRChain(sample_start=False, sample_goal = True,
                           constrain=False, TSR = grasp_tsr)

    return [grasp_chain]
