CMAKE_MINIMUM_REQUIRED(VERSION "3.12")


#this is set manually for the lab computers this should not be required for other systems with proper installation
set(CMAKE_TOOLCHAIN_FILE "K:\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake")

set(CMAKE_CXX_STANDARD 14)

project( ObjectDetection )
find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( ObjectDetection objectDetection.cpp Timer.h Timer.cpp)

target_link_libraries( ObjectDetection ${OpenCV_LIBS} )

set_target_properties (
	ObjectDetection PROPERTIES 
	VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)