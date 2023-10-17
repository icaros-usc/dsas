^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package ada_description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.1.3 (2015-04-15)
------------------

=======
* added slight rotation to j2 to bring it in line with reality
* added configurations and changed mico.urdf file
* changes in the mico.urdf
* added camera model in the mico.urdf, changed constraints in controllers, modifications in exceptions returned by adarobot
* Tweaked finger joint limits.
* Removed Mico .robot.xml files.
* Fixed joint limits on fingers.
* Ignore collision between finger links.
* Fixed joint limits in URDF.
* Renamed Mico model to mico.urdf and mico.srdf.
* Removed transmisison from mico-modified.urdf.
* Fixed mico-modified mimic joints.
* Updated joint limits in mico-modified.
* Got trajectories running on ADA.
* Changed finger limits to match the .robot.xml
* Fixed <limits> in mico-modified-updated.
  - Copied joint limits from the .robot.xml file. Chris calibrated these
  on the real hardware by moving ADA to each joint limit.
  - Set velocity limits from the Mico datasheet.
  - Set effort limits from the Mico datasheet (used continuous ratings).
  - Did NOT update the finger position, velocity, or effort limits.
* Switching to URDF and SRDF.
* Delete mico-modified-old.urdf
* Delete mico.urdf


* Contributors: Michael Koval, Pyry Matikainen, Stefanos Nikolaidis


0.1.0 (2015-04-08)
------------------
* fixed bug in adarobot that could not load the tsr library
  renamed robot name to ADA in tsr.yaml files
* Delete mico-modified.urdf~
* added updated urdf and srdf model
* Fixed joint-limits
* Contributors: Stefanos Nikolaidis
