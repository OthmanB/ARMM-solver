cmake_minimum_required(VERSION 3.5)
project(MakeStarTests)

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
    ../../changed_code/solver_mm_new.cpp
    ../../changed_code/ksi2_fct_new.cpp
    ../../changed_code/make_asymptotic_star_new.cpp
    ../../changed_code/linfit.cpp
    ../../unchanged_code/configure_make_star.cpp
    ../../unchanged_code/bump_DP.cpp
    ../../unchanged_code/solver_mm.cpp
    ../../unchanged_code/interpol.cpp
    ../../unchanged_code/string_handler.cpp
    ../../unchanged_code/derivatives_handler.cpp
    ../../unchanged_code/noise_models.cpp
    ../../unchanged_code/linfit.cpp
    ../../unchanged_code/linspace.cpp
    ../../unchanged_code/readparams_job.cpp
    ../../unchanged_code/writeparams_job.cpp
)

# Add executable
add_executable(MakeStarTests ${SOURCES_TESTS} make_star_unittest.cpp)

# Link Eigen and Boost libraries
target_link_libraries(MakeStarTests Eigen3::Eigen ${Boost_LIBRARIES})

