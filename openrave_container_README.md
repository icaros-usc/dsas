# OpenRAVE Container Notes

WARNING: Any of these can change since the required versions of Ubuntu, python, ROS, and
OpenRAVE are not supported anymore.

## Base Container

Since the original installation used Ubuntu 14.04, the container extends an Ubuntu 14.04
container.

## Files

- `src/` is copied before building the container since we don't change any C++ files. Many
  of the python packages in `requirements.txt` are provided by ROS/catkin so we need to
  run `catkin build` and source the setup files to properly set the environment variables
  before `pip install`. Ideally, `requirements.txt` should not have those packages listed
  since they will be installed, but I am too lazy to clean it up.
- `openrave_pybind_cmakelists.txt` adds some extra library links to OpenRAVE's CMakeLists.
  Without these links, including boost python headers fails, causing the build to install
  OpenRAVE without its python libraries. Since we are installing an older version of
  OpenRAVE and there was no error with the original CMakeLists when installing it directly
  on an Ubuntu 14.04 machine, this hack seemed to be the best way to fix it instead of
  creating a fork of OpenRAVE or some other cleaner solution.

## Installation steps

- Tzdata needs to be installed explicitly since it can be installed in some other `apt-get
  install` step and cause an error since it expects a UI without the specified environment
  variables.
- Basic requirements are combined from multiple places. Not sure if all of them are needed
  but doesn't seem to hurt to keep them.
- Python needs to be installed from source since the default version of python and pip
  will cause an SNI error when installing packages.
- `--enable-shared` is required since OpenRAVE needs the python shared library to build
  its python libraries.
- `--enable-unicode=ucs4` is required since Ubuntu's python uses it and I think
  ROS/OpenRAVE is always installed using Ubuntu's python. Since we overwrite Ubuntu's
  python, not having this option creates a mismatch, leading to some symbols not being found
  when running the experiments.
- `LD_LIBRARY_PATH` needs to be set manually since for some reason; it doesn't get set by
  default when `--enable-shared` option is set.
- `get-pip.py` is required to install pip since we are installing python from source.
- ROS install is standard, but it doesn't like custom python versions. So python 2.7.6
  gets installed from apt. Since our custom python build overwrites the links, I think this
  python gets hidden during regular use. However, it still seems to be installed in the
  container and used by some packages during installation.
- OpenRAVE installation is pretty close to how it was done in [this
  repo](https://github.com/crigroup/openrave-installation). I had to copy and modify a few
  commands from the GitHub to make it run in singularity's post section. I also needed to
  modify one of the CMakeLists file in the OpenRAVE repo due to the reason mentioned
  above. So I just re-write the file with my version before building.
- Catkin workspace + packages from `src/` are installed similar to a regular catkin
  workspace. `rosdep` messes up some package versions, so they are reinstalled after
  `rosdep install`.  I use a small trick in Line 182 of
  [openrave_container.def](openrave_container.def) to automatically source the setup files
  when creating a singularity shell or running a command in the container since the
  commands in singularity's environment section run in sh instead of bash, which creates
  problems with sourcing `devel/setup.bash`. Installing catkin packages this way also
  affects python files since importing will import the python files that were copied over
  while building the container, leading to any subsequent changes being ignored. A hack to
  fix this is to bind `src/` to `/usr/project/catkin/src` when creating a singularity
  shell or running a command in the container so that python uses the updated files. Note
  that this hack only works if there are no new catkin packages. If there are new
  packages, simply doing `cd /usr/project/catkin; catkin build` in a shell created with
  the hack and restarting the singularity shells would probably work, but I haven't tried
  it.
- Some packages in `requirements.txt` were removed since they weren't found by the
  container's pip and don't seem to be needed for this project.

