^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package adapy
^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.1.3 (2015-04-15)
------------------
* commented out spammy trajectory printing
* Merge branch 'bugfix/AdaRobotCleanup' of https://github.com/personalrobotics/ada into bugfix/AdaRobotCleanup
* added configurations and changed mico.urdf file
* Fall back on HauserParabolicSmoother.
* Better cleanup for duplicate waypoints.
* changes in the mico.urdf
* added camera model in the mico.urdf, changed constraints in controllers, modifications in exceptions returned by adarobot
* Fixed ADA's planners.
* Removed smoothing hack.
* Added the Stefanos hack.
* Fixed the retract configuration (again).
* Fixed partial-DOF trajectory execution.
* Removed hand (copied and pasted from PrPy).
* Fixed retract configuration to not be in a joint limit.
* Fixed a typo in futures.
* added service call to velocity_joint_mode_controller
* Fixing small bug when calling ExecuteTrajectory on the robot class
* Merge branch 'bugfix/AdaRobotCleanup' of github.com:personalrobotics/ada into bugfix/AdaRobotCleanup
  Conflicts:
  ada_description/robots/mico.urdf
* Renamed Mico model to mico.urdf and mico.srdf.
* Updated joint limits in mico-modified.
* Added JointStateClient.
* Got trajectories running on ADA.
* Fixed a bunch of typos.
* Fixed bugs.
* Fixed simulation mode.
* Implemented ExecuteTrajectory.
* Split ros_control into multiple files.
* More refactoring.
* Created ModeContextManager.
* Added documentation.
* Implemented FollowJointTrajectoryClient.
* Made TrajectoryFuture inherit from Future.
* Implemented a base Future class.
* Added the GreedyIKPlanner.
* Use package:// URI's to load the ADA model.
* Switching to URDF and SRDF.
* Cleaning up __init__.
* Merge branch 'master' into bugfix/AdaRobotCleanup
  Conflicts:
  adapy/src/adapy/adarobot.py
* Working on ros_control trajectory execution in Python.
* Added Snap and NamedPlanner.
* More AdaPy cleanup.
* Fixed a typo.
* Cleanup ADARobot.
* Contributors: Jennifer King, Michael Koval, Stefanos Nikolaidis

0.1.0 (2015-04-08)
------------------
* fixed bug in adarobot that could not load the tsr library
  renamed robot name to ADA in tsr.yaml files
* Changed default flag to True
* Adapy now supports loading urdf model
* Fixed joint-limits
* Removed named planner for now
* Updated adarobot to load planners in the same way as in herbpy master
* changed planner sequence to be similar to herbpy
* Moved test.py to scripts/.
* Deleted .pyc files and added them go .gitignore.
* Added console.py to AdaPy, to mimic HerbPy.
* added named configurations
* Contributors: Michael Koval, Stefanos Nikolaidis
