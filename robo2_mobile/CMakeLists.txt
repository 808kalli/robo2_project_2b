cmake_minimum_required(VERSION 2.8.3)
project(robo2_mobile)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs geometry_msgs sensor_msgs message_generation genmsg)

## Declare ROS messages and services
#add_message_files(FILES)
#add_service_files(FILES)

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs geometry_msgs sensor_msgs)

## Declare a catkin package
catkin_package(CATKIN_DEPENDS message_runtime)

## Build follower
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(random_walker ./src/random_walker.cpp)
target_link_libraries(random_walker ${catkin_LIBRARIES})
#add_dependencies(random_walker robo2_mobile_generate_messages_cpp)
add_dependencies(random_walker ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
