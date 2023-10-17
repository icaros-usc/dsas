cmake_minimum_required(VERSION 2.8.3)
project(simple_environment)
 
find_package(catkin REQUIRED COMPONENTS)
  #openrave_catkin
  #message_generation)


catkin_python_setup()

## Generate added messages and services with any dependencies listed here
#generate_messages()

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  CATKIN_DEPENDS message_runtime
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

#add_custom_target(${PROJECT_NAME}_data ALL
#    COMMAND ${CMAKE_COMMAND} -E copy_directory
#            "${PROJECT_SOURCE_DIR}/data"
#            "${OpenRAVE_DEVEL_DIR}/${OpenRAVE_DATA_DIR}"
#)

#install(DIRECTORY "${PROJECT_SOURCE_DIR}/data/"
#    DESTINATION "${OpenRAVE_INSTALL_DIR}/${OpenRAVE_DATA_DIR}"
#)
