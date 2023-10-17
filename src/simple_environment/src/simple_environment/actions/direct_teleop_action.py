#creates a direct teleop action

from bypassable_action import ActionException, BypassableAction
from ada_teleoperation.AdaTeleopHandler import AdaTeleopHandler, is_done_func_button_hold
from ada_teleoperation.DataRecordingUtils import TrajectoryData

class DirectTeleopAction(BypassableAction):
    def __init__(self, bypass=False):
        BypassableAction.__init__(self, 'DIRECT_TELEOP', bypass=bypass)
        
    def _run(self, manip, ui_device, filename_trajdata=None):
        robot = manip.GetRobot()
        env = robot.GetEnv()

        if filename_trajdata:
          traj_data_recording = TrajectoryData(filename_trajdata)
        else:
          traj_data_recording = None

        ada_teleop = AdaTeleopHandler(env, robot, teleop_interface=ui_device, num_input_dofs=2, use_finger_mode=False)
        ada_teleop.execute_direct_teleop(is_done_func=is_done_func_button_hold, traj_data_recording=traj_data_recording)



