cmake_minimum_required(VERSION 2.8.3)
project(rs_ekf)

find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  sensor_msgs 
  nav_msgs 
  tf
  tf_conversions
  std_msgs
)

find_package(Eigen3 3.1.0 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES
#  CATKIN_DEPENDS 	roscpp std_msgs
#  DEPENDS
)

set(CMAKE_BUILD_TYPE Release)

include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR} ${catkin_INCLUDE_DIRS})

add_executable(rs_ekf src/rs_ekf.cpp)
target_link_libraries(rs_ekf ${catkin_LIBRARIES} ${EIGEN3_LIBS})
add_dependencies(rs_ekf ${catkin_EXPORTED_TARGETS})
add_dependencies(rs_ekf ${PROJECT_NAME}_gencpp)
