cmake_minimum_required(VERSION 3.4.1)

project(ARMMSolver VERSION 0.71)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(Boost_USE_MULTITHREADED TRUE)
set(CMAKE_CXX_FLAGS "-I/usr/include/  -O2")
set(CMAKE_C_FLAGS "-I/usr/include/  -O2")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

#Bring the headers into the project
#set(HEADERS  #_COMMON
#	
#   )

#Bring the core program sources into the project
set(SOURCES_MAIN  #_COMMON
    do_solve.cpp
    bump_DP.cpp
    solver_mm.cpp
    interpol.cpp
    string_handler.cpp
    derivatives_handler.cpp
    noise_models.cpp
    linfit.cpp
    linspace.cpp
    )

set(SOURCES_TEST  #_COMMON
    test.cpp
    bump_DP.cpp
    solver_mm.cpp
    interpol.cpp
    string_handler.cpp
    derivatives_handler.cpp
    noise_models.cpp
    linfit.cpp
    linspace.cpp
    )

set(SOURCES_TEST2  #_COMMON
    main.cpp
    bump_DP.cpp
    solver_mm.cpp
    interpol.cpp
    string_handler.cpp
    derivatives_handler.cpp
    noise_models.cpp
    linfit.cpp
    linspace.cpp
    )

set(HEADERS
    bump_DP.h
    solver_mm.h
    data.h
    data_solver.h
    interpol.h
    string_handler.h
    derivatives_handler.h
    noise_models.h
    linfit.h
    linspace.h
    version_solver.h
)

#Look for eigen and explicitly specify to use it. EIGEN3_INCLUDE_DIR Must be specified in the bashrc
include_directories( "$ENV{EIGEN3_INCLUDE_DIR}" )

#Dependencies that are not portable (BOOST)
#find_package(Boost COMPONENTS system filesystem iostreams math REQUIRED)
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

add_executable(${CMAKE_PROJECT_NAME}  ${HEADERS} ${SOURCES_MAIN} ${DIAG_SRC})
#target_link_libraries(${CMAKE_PROJECT_NAME}  ${Boost_SYSTEM_LIBRARY_RELEASE} ${Boost_MATH_LIBRARY_RELEASE} ${Boost_FILESYSTEM_LIBRARY_RELEASE} ${Boost_IOSTREAMS_LIBRARY_RELEASE} Boost::boost)
target_link_libraries(${CMAKE_PROJECT_NAME} Boost::boost)

add_executable(solver_test ${HEADERS} ${SOURCES_TEST} ${DIAG_SRC})
#target_link_libraries(${CMAKE_PROJECT_NAME}  ${Boost_SYSTEM_LIBRARY_RELEASE} ${Boost_MATH_LIBRARY_RELEASE} ${Boost_FILESYSTEM_LIBRARY_RELEASE} ${Boost_IOSTREAMS_LIBRARY_RELEASE} Boost::boost)
target_link_libraries(${CMAKE_PROJECT_NAME} Boost::boost)

add_executable(solver_test2 ${HEADERS} ${SOURCES_TEST2} ${DIAG_SRC})
#target_link_libraries(${CMAKE_PROJECT_NAME}  ${Boost_SYSTEM_LIBRARY_RELEASE} ${Boost_MATH_LIBRARY_RELEASE} ${Boost_FILESYSTEM_LIBRARY_RELEASE} ${Boost_IOSTREAMS_LIBRARY_RELEASE} Boost::boost)
target_link_libraries(${CMAKE_PROJECT_NAME} Boost::boost)

