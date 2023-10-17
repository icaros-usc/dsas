cmake_minimum_required(VERSION 2.8.3)
project(or_trac_ik)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(DetectCXX11Flags)

find_package(catkin REQUIRED COMPONENTS
  openrave_catkin
  eigen_conversions
  trac_ik_lib
)

find_package(OpenRAVE REQUIRED)

#message(STATUS "Found NLopt at: ${NLOPT_INCLUDE_DIR}")

catkin_package()

include_directories(
  include
  ${OpenRAVE_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${eigen_INCLUDE_DIRS}
)

openrave_plugin(${PROJECT_NAME}
  src/or_trac_ik.cpp
  src/plugins.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)
