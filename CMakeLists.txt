# Set the name of the project and target:
SET(TARGET "matrix-free-dealii-precice")
SET(SOLVER1 "solid")
SET(SOLVER2 "heat")

CMAKE_MINIMUM_REQUIRED(VERSION 3.10.2)

# Disable in-source compilation
IF (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  MESSAGE(FATAL_ERROR "In-source builds are disabled. Please change to a seperate
      build directory and run cmake again.
     ")
ENDIF()

INCLUDE_DIRECTORIES(
 ${CMAKE_SOURCE_DIR}/include
)

SET(CLEAN_UP_FILES
  # a custom list of globs, e.g. *.log *.vtk
  *.vtu
)

# Assume a debug build if not specified
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "DEBUG")
  MESSAGE(STATUS "No build type specified. Building in debug mode.")
ENDIF()

# Print the current build type
MESSAGE(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

FIND_PACKAGE(deal.II 9.3
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
   MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

#
# Are all dependencies fulfilled?
#
IF(NOT DEAL_II_WITH_P4EST) # keep in one line
  MESSAGE(FATAL_ERROR "
Error! This tutorial requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_P4EST = ON
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options
    DEAL_II_WITH_P4EST = ${DEAL_II_WITH_P4EST}
which conflict with the requirements."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Run macro to get Git info:
DEAL_II_QUERY_GIT_INFORMATION()
# If we could not get the tag from Git, get it from VERSION file
IF(NOT GIT_TAG)
  FILE(STRINGS "${CMAKE_SOURCE_DIR}/VERSION" GIT_TAG LIMIT_COUNT 1)
ENDIF()
MESSAGE(STATUS "Code version ${GIT_TAG}")
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/src/version.cc.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.cc)

PROJECT(${TARGET} LANGUAGES CXX)

# This project requires c++17
IF(NOT ${DEAL_II_WITH_CXX17} AND NOT ${DEAL_II_WITH_CUDA})
    SET(CMAKE_CXX_STANDARD 17)
    SET(CMAKE_CXX_EXTENSIONS OFF)
    SET(CMAKE_CXX_STANDARD_REQUIRED ON)
ENDIF()

# Build the version specific object
ADD_LIBRARY(version OBJECT ${CMAKE_CURRENT_BINARY_DIR}/version.cc)

# Build the executables
ADD_EXECUTABLE(${SOLVER1}
    src/${SOLVER1}.cc
  )

# Build the cuda version in case we have cuda support
IF(${DEAL_II_WITH_CUDA})
  ADD_EXECUTABLE(${SOLVER2}
      src/${SOLVER2}.cu
    )
  IF(NOT CMAKE_CUDA_HOST_COMPILER)
       SET(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  ENDIF()
  ENABLE_LANGUAGE(CUDA)
ELSE()
  ADD_EXECUTABLE(${SOLVER2}
      src/${SOLVER2}.cc
    )
ENDIF()

# Setup targets
DEAL_II_SETUP_TARGET(${SOLVER1})
DEAL_II_SETUP_TARGET(${SOLVER2})

# the environment variable precice_DIR is searched by default
FIND_PACKAGE(precice 2.0
        HINTS ${precice_DIR} ${PRECICE_DIR} $ENV{PRECICE_DIR}
  )
IF(NOT ${precice_FOUND})
   MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of preCICE. ***\n\n"
    "You may want to either pass a flag -Dprecice_DIR=/path/to/precice to cmake\n"
    "(where the path points to the installation prefix or the build directory)\n"
    " or set an environment variable \"precice_DIR\" that contains this path."
    )
ENDIF()
MESSAGE(STATUS "Found preCICE version ${precice_VERSION} at ${precice_DIR}")

# Add a define for experimental preCICE features
IF(${precice_VERSION} GREATER_EQUAL "2.3.0")
  SET(SUPPORT_EXPERIMENTAL ON)
ELSE()
  SET(SUPPORT_EXPERIMENTAL OFF)
ENDIF()
OPTION(PRECICE_EXPERIMENTAL "Enable experimental preCICE (>=v2.3.0) features, which require the \"experimental\" tag within the preCICE configuration file." ${SUPPORT_EXPERIMENTAL})

if(PRECICE_EXPERIMENTAL)
    ADD_DEFINITIONS(-DPRECICE_EXPERIMENTAL)
ENDIF(PRECICE_EXPERIMENTAL)
MESSAGE(STATUS "Support for experimental preCICE features: ${PRECICE_EXPERIMENTAL}")

# Link targets to preCICE
TARGET_LINK_LIBRARIES(${SOLVER1} version precice::precice)
TARGET_LINK_LIBRARIES(${SOLVER2} version precice::precice)

INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/include)

# Some take over from deal.II's autopilot
IF(CMAKE_GENERATOR MATCHES "Ninja")
  SET(_make_command "$ ninja")
ELSE()
  SET(_make_command " $ make")
ENDIF()

INSTALL(TARGETS ${SOLVER1} ${SOLVER2} DESTINATION bin)
#
# Custom "debug" and "release" make targets:
#
IF(${DEAL_II_BUILD_TYPE} MATCHES "Debug")
  ADD_CUSTOM_TARGET(debug
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} -E echo "***"
    COMMAND ${CMAKE_COMMAND} -E echo "*** Switched to Debug mode. Now recompile with: ${_make_command}"
    COMMAND ${CMAKE_COMMAND} -E echo "***"
    COMMENT "Switching CMAKE_BUILD_TYPE to Debug"
    VERBATIM
    )
ENDIF()

IF(${DEAL_II_BUILD_TYPE} MATCHES "Release")
  ADD_CUSTOM_TARGET(release
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} -E echo "***"
    COMMAND ${CMAKE_COMMAND} -E echo "*** Switched to Release mode. Now recompile with: ${_make_command}"
    COMMAND ${CMAKE_COMMAND} -E echo "***"
    COMMENT "Switching CMAKE_BUILD_TYPE to Release"
    VERBATIM
    )
ENDIF()
