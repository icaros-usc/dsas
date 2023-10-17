# ada_meal_scenario
A set of scripts for a meal serving scenario using Ada. 

## generate_saved_trajectories.py
If the configuration of any fixed points (e.g. servingConfiguration) are changed, this must be run to cache new trajectories

## runBiteServing.py (Marshmallow demo):

Note: in all terminals, first `source devel/setup.bash`. Each of the following commands happens in its own terminal.

1. Set up a roscore
	```bash
	roscore
	```
	
1. Make sure the Kinova USB joystick is plugged in, then launch it
	```bash
	roslaunch ada_launch kinova_joystick.launch
	```

1. Open rviz
	```bash
	rosrun rviz rviz
	```

1. Start the ADA controller (if running on real robot)
	```bash
	roslaunch ada_launch default.launch
	```

1. Make sure the depth camera is attached and powered on, then launch it
	```bash
	roslaunch openni2_launch openni2.launch
	```
	Note: The error "Unsupported color video mode" in red is ok
	
	At this point, the camera feed should be showing in rviz (subscribe to the /camera/depth/image topic).

1. Start the morsel detector
	```bash
	cd src/morsel/detector/
	python biteserver.py structureio_settings.json verbose.json
	```

1. Start the demo
	```bash
	rosrun ada_meal_scenario runBiteServing.py --real --viewer=interactivemarker
	```
	
At this point, a GUI should appear that allows you to select the operation mode. Select a mode (e.g., shared autonomy) and an input device (e.g., Kinova USB), then press the start next trial button.



