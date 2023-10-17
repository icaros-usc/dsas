# ada_teleoperation
Teleoperation code using adapy, including direct teleoperation mimicking Kinova's system.

You can run direct teleoperation with

'''
rosrun ada_teleoperation AdaDirectTeleop.py -input=INPUT -joy_dofs=DOFS
'''

Where INPUT specifies the device to read inputs from (either mouse, kinova, or hydra), and DOFS specifies the number of degrees of freedom of the joystick (either 2 or 3). The controller also keeps track of modes, including one mode for moving the fingers.

Alternatively, you can utilize the AdaTeleopHandler to send twists and finger velocities to the arm with single commands, e.g. for a shared autonomy system.
