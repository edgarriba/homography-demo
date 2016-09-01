cmake_minimum_required(VERSION 2.8)

project(homography-demo)

find_package(OpenCV REQUIRED COMPONENTS
             core imgproc features2d highgui calib3d xfeatures2d)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})
