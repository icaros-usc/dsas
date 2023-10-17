# Surrogate Assisted Generation of Human-Robot Interaction Scenarios

Code for the paper "Surrogate Assisted Generation of Human-Robot Interaction Scenarios".

## Contents

* [Manifest](#manifest)
* [Installation](#installation)
  * [Notes](#notes)
* [Running Examples](#running-examples)
  * [Running Shared Control Teleoperation Example](#running-shared-control-teleoperation-example)
  * [Running Shared Workspace Collaboration Example](#running-shared-workspace-collaboration-example)
* [Running QD Search Experiments](#running-qd-search-experiments)
  * [Running the Simulation Server](#running-the-simulation-server)
  * [Running QD Search](#running-qd-search)
* [License](#license)

## Manifest

- `qd/`: Code for QD search.
- `src/`: Code for simulating the scenarios.
- `scripts/`: Bash scripts for running experiments.

## Installation

1. Install
   [singularity](https://docs.sylabs.io/guides/3.6/user-guide/quick_start.html#quick-installation-steps).
1. Create the openrave container: `sudo singularity build openrave_container.sif
   openrave_container.def`. See
   [openrave_container_README.md](openrave_container_README.md) for more details about the
   container and potential issues.

### Notes
1. When opening singularity shell or running commands in the container, set
   `SINGULARITYENV_DISPLAY=$DISPLAY` to allow GUI apps to work. Since X11 and local
   monitor use different environment variables, this needs to be set during execution
   time.
1. Use option `--bind <path_to_this_repo>/src:/usr/project/catkin/src` when opening
   singularity shell or running commands in the container. Otherwise, singularity's python
   will continue to import the original python files included during the build.

## Running Examples

Note, you might require multiple singularity shells for this step. In the first shell,
run `roscore`.

Run the python script in the second shell as described in the subsections below.

In the third shell, run `rosrun rviz rviz`.

In the rviz window, add `MarkerArray` with Marker Topic
`visualization_marker_array` and `InteractiveMarkers` with Update Topic
`/openrave/update`.

### Running Shared Control Teleoperation Example

Commands to run the python script:

```bash
cd src/simple_environment/src/simple_environment
python scenario_generators/scenario_generator.py
```

In the rviz window, the robot should move towards a goal.

### Running Shared Workspace Collaboration Example

Commands to run the python script:

```bash
cd src/simple_environment/src/simple_environment
python scenario_generators/collab_scenario_generator.py
```

In the rviz window, the human hand (visualized using a sphere) should move towards a goal
and the robot arm should move towards a different goal.

## Running QD Search Experiments

In this setup, we connect the QD code in this directory with the simulation code running
on [OpenRAVE](https://github.com/rdiankov/openrave). The simulation code is turned into a
Python 2 Flask server which handles requests for OpenRave evaluations while this code uses
pyribs on Python 3 and queries the server for evaluations.

### Running the Simulation Server

Note that we assume Python 2.7.12, as that is what the machine uses.

Install the additional requirements:

```bash
pip install -r requirements_networking.txt
```

In one terminal, run:

```bash
roscore
```

In another, start the server with:

```bash
cd src/simple_environment/src/simple_environment
python search/server.py -c SERVER_CONFIG
```

Replace `SERVER_CONFIG` with `search/config/experiment/experiment.tml` for shared control
teleoperation experiments or with `search/config/experiment/experiment_collab.tml`
for shared workspace collaboration.

### Running QD Search

See [qd/README.md](qd/README.md)

## License

This code is released under the [MIT License](LICENSE). Some packages in `src/` have a
BSD-3 license.
