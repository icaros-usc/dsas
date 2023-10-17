^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package or_nlopt_ik
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.1.1 (2015-04-15)
------------------
* fixed bug with activeDofIndices sequence (it was assumed they start from 0)
* Cleaned up logging (`#3 <https://github.com/personalrobotics/or_nlopt_ik/issues/3>`_ and `#4 <https://github.com/personalrobotics/or_nlopt_ik/issues/4>`_).
* added thresholds for error checking
* Merge pull request `#2 <https://github.com/personalrobotics/or_nlopt_ik/issues/2>`_ from personalrobotics/collision_check
  Checking self and environment collisions
* collisions working
* first pass at collision check
* reformat and cleanup for my own understanding
* Contributors: Michael Koval, Stefanos Nikolaidis, mklingen

0.1.0 (2015-04-08)
------------------
* latest changes in precision settings for demo working version
* Merge branch 'master' of https://github.com/personalrobotics/or_nlopt_ik
* Added check for solution error with respect to threshold
* bandaid on num free parameters interface
* Changed precision (xtol) to be proportional to min of DOF resolution values.
* fixed bug when robot's transform is not identity
* added condition for continuous joints
* Included gradients in the objective function
* speed of about 9ms with new quaternion distance function
* Moved FK to the objective function
* tuned the constraint parameters to make it faster
* Moved limit initialization in the init function`
* Incorporated all of Mike's comments
* Incorporated more of Mike's comments. Need to write RobotSaver to work properly.
* Incorporated part of Mike's comments
* basic version of nlopt solver for ada
* Renamed to or_nlopt_ik.
* Initial commit.
* Contributors: Stefanos Nikolaidis, mklingen
