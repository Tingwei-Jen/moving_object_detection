cmake_minimum_required( VERSION 2.8 )
project( optical_flow )

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3" )

#opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

#source files
set(SOURCE_FILES 
	main.cpp
)

add_executable(${PROJECT_NAME}  ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})
