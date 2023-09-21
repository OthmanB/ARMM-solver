cmake_minimum_required(VERSION 3.5)
project(LinearFitTests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og")

# Find Eigen package depending on OS
if(APPLE)
    find_package(Eigen3 REQUIRED)
endif()
include_directories( "$ENV{EIGEN3_INCLUDE_DIR}" )

# Find Boost unit_test_framework package
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
set(BOOST_TEST_LOGGER "HRF_LOG")
set(BOOST_TEST_MESSAGES "YES")

# Add executable
add_executable(LinearFitTests linfit_unittest.cpp ../../unchanged_code/linfit.cpp ../../changed_code/linfit.cpp)

# Link Eigen library
target_link_libraries(LinearFitTests Eigen3::Eigen)
