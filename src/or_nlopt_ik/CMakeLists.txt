cmake_minimum_required(VERSION 2.8.3)
project(or_nlopt_ik)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(DetectCXX11Flags)

find_package(catkin REQUIRED COMPONENTS
  openrave_catkin
)
find_package(NLOPT REQUIRED)
find_package(OpenRAVE REQUIRED)

message(STATUS "Found NLopt at: ${NLOPT_INCLUDE_DIR}")

catkin_package()
include_directories(
  include
  ${OpenRAVE_INCLUDE_DIRS}
  ${NLOPT_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

openrave_plugin(${PROJECT_NAME}
  src/nloptik.cpp
  src/plugins.cpp
)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${NLOPT_LIBRARIES}
)
