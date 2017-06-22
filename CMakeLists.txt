cmake_minimum_required (VERSION 3.2)
# Requirement for the findMatlab script shipped with CMake.
project (fertilized C CXX)

set(PROJECT_SOURCE_DIR "C:/Developer/fertilized-forests/")

list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/build_support/cmake/modules)

# Versioning.
set (fertilized_VERSION_MAJOR 1)
set (fertilized_VERSION_MINOR 0)
set (fertilized_VERSION_PATCH 2)
set (fertilized_VERSION
  "${fertilized_VERSION_MAJOR}.${fertilized_VERSION_MINOR}.${fertilized_VERSION_PATCH}")
set (fertilized_VERSION_NUMBER
  "${fertilized_VERSION_MAJOR}${fertilized_VERSION_MINOR}${fertilized_VERSION_PATCH}")

# Set a default build type if none was specified.
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message (STATUS "Setting build type to 'Release' since none was specified.")
  set (CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui.
  set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Setup the library.
include_directories (include)
include_directories (${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories (${PROJECT_SOURCE_DIR}/include)

set (BOOST_COMPONENTS date_time
                      serialization
                      filesystem
                      system
                      unit_test_framework
                      thread)
# Eigen.
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
# OpenCV.
find_package (OpenCV COMPONENTS core imgproc highgui)
if (OpenCV_FOUND)
  include_directories (${OpenCV_INCLUDE_DIRS})
  if (OpenCV_LIBRARIES)
    set (OpenCV_COMPATIBLE_LIBRARIES ${OpenCV_LIBRARIES})
  else()
    set (OpenCV_COMPATIBLE_LIBRARIES ${OpenCV_LIBS})
  endif()
endif()

add_definitions (-DSERIALIZATION_ENABLED)


# Use C++11 features.
set (REQ_CPP11_FEATURES
     cxx_strong_enums
     cxx_auto_type)

# Check for OpenMP support.
find_package (OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# Fix clang warnings.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-tautological-compare -Wno-logical-op-parentheses -Wno-deprecated-register")
endif()

# Configure the dependencies.
find_package (Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS})
# Deactivate boost auto-linking in favor of direct CMake linking.
add_definitions (-DBOOST_ALL_NO_LIB)
include_directories (${Boost_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
add_definitions (-DEIGEN_MPL2_ONLY)
include_directories (${EIGEN3_INCLUDE_DIR})

include(build_support/cmake/modules/ConfigureRealSense.cmake)

# Targets.
add_subdirectory (include)
add_subdirectory (src)
# Create the examples.
add_subdirectory (examples/c++)


