cmake_minimum_required(VERSION 3.5)
project(ZetaTests)

set(CMAKE_CXX_STANDARD 17)

# Set compiler optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og -fopenmp")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(Boost_USE_MULTITHREADED TRUE)

# Find Boost packages
find_package(Boost REQUIRED COMPONENTS unit_test_framework)
include_directories(${Boost_INCLUDE_DIRS})

# Find Eigen and OpenMP package depending on OS
if(APPLE)
    find_package(OpenMP REQUIRED COMPONENTS C)
    find_package(Eigen3 REQUIRED)
else()
    find_package(OpenMP REQUIRED)
endif()
include_directories( "$ENV{EIGEN3_INCLUDE_DIR}" )

# Bring the core program sources into the project
set(SOURCES_TESTS
    #ksi2_fct_old.cpp
    ../../unchanged_code/bump_DP.cpp
    ../../unchanged_code/solver_mm.cpp
    ../../unchanged_code/interpol.cpp
    ../../unchanged_code/string_handler.cpp
    ../../unchanged_code/derivatives_handler.cpp
    ../../unchanged_code/noise_models.cpp
    ../../unchanged_code/linfit.cpp
    ../../unchanged_code/linspace.cpp
    ../../changed_code/ksi2_fct_new.cpp
)

# Add executable
add_executable(ZetaTests ${SOURCES_TESTS} ksi2_fct_unittest.cpp)

# Link Eigen and Boost libraries
target_link_libraries(ZetaTests Eigen3::Eigen ${Boost_LIBRARIES})

