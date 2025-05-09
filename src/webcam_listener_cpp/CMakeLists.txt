cmake_minimum_required(VERSION 3.8)
project(webcam_listener_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(webcam_listener src/webcam_listener.cpp)
ament_target_dependencies(webcam_listener
  rclcpp
  sensor_msgs
  image_transport
  cv_bridge
  OpenCV
)
add_executable(image_evaluator src/image_qual_evaluation.cpp)
ament_target_dependencies(image_evaluator
  rclcpp
  sensor_msgs
  image_transport
  cv_bridge
  OpenCV
)

install(TARGETS
  webcam_listener
  image_evaluator
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
