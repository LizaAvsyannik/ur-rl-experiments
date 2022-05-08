cmake_minimum_required(VERSION 3.0.2)
project(ur5_rl)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_python_setup()

catkin_package()
 
include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS src/ur5_rl/a2c_main.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
