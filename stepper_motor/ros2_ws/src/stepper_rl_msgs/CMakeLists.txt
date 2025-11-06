cmake_minimum_required(VERSION 3.8)
project(stepper_rl_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RefTarget.msg"
  "msg/Observation.msg"
  "msg/Action.msg"
  "msg/PolicyUpdate.msg"
  "msg/Health.msg"
  "msg/EncoderData.msg"
  DEPENDENCIES std_msgs
)

ament_package()
