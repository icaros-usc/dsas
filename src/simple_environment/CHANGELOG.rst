^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package ada_meal_scenario
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
0.3.0 (2016-05-23)
------------------
* April 2016 working demo
* Supports Kinova fork holder


0.2.1 (2015-04-15)
------------------
* latest changes
* nicely tuned offsets
* nice fine-tuned offsets
* recreated trajectories, set offsets to work with structureIO and added sleep
* changes using mike's branch
* Contributors: Stefanos Nikolaidis

* nicely tuned offsets
* nice fine-tuned offsets
* recreated trajectories, set offsets to work with structureIO and added sleep
* changes using mike's branch
* Contributors: Stefanos Nikolaidis

0.2.0 (2015-04-12)
------------------
* Merge branch 'feature/action_framework'
  Conflicts:
  scripts/runBiteServing_FSM.py
* adding files to generate trajectories and load from trajectories
* adding FSM scripts to master
* Now it can detect multiple morsels
* changed traj_serving.xml, changed manip.Plan to robot.Plan,. added time.sleep
* Fixing bugs. Update location trajectories are written to.
* Adding script to generate saved trajectories automatically. Restructuring directory to make imports cleaner
* Fixing bug in unsubscribing from morsal callback. Also fixing bug that added lots of morsals to environment.
* removing uneeded running variable
* Adding ability to run multiple times
* pushing latest changes
* fixed some minor issues
* Adding forgotten import
* removing old state machine scripts
* Working end-to-end
* Cleaning up and fixing some bugs
* Initial (untested) stab at breaking out demo into actions
* Contributors: Jennifer King, Stefanos Nikolaidis

0.1.0 (2015-04-08)
------------------
* added scripts for generating trajectories offline and loading cached trajectoreis
* Changed ExecutePath to ExecuteTrajectory, as this is even faster
* changed time-tests to use prpy.util.Timer and updated bite_serving_FSM to plan using execute=False and then executePath
* added file for time_testing
* removed hardcoded planner.Sequence
* deleted unused imports
* Set the arm to be active using setActiveManipulator
* Custom class now extends object
* Removed SetDOFValues in Sim=False
* Merge branch 'master' of https://github.com/personalrobotics/ada_meal_scenario
* deleted named configurations in the beginning of file
* Delete bite_serving.py
* removed some comments
* added trajectories
* now some trajectories are cached. still planning is excruciatingly slow
* working version with nlopt planner and greedy_ik
* changed so that PlanToEndEffectorOffset is Done with vectorfield/greedyik, I have it hardcoded for now
* fixed spaces
* latest version on bite_serving
* changed water_serving to ignore april tags for demo purposes
  added bite_serving with Finite State Machine
* new change
* Meal Serving Project script
* added tf publisher to water_serving scenario
* task logging formatter
* populated repository
* Initial commit
* Contributors: PyryM, Stefanos Nikolaidis
