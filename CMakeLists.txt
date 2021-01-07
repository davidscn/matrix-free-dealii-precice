##
#  CMake script for the step-<XYZ> tutorial program:
##

# Set the name of the project and target:
SET(TARGET "main")
SET(TARGETLIB "mf_elasticity")

INCLUDE_DIRECTORIES(
 ${CMAKE_SOURCE_DIR}/include
)

# Declare all source files the target consists of:
SET(TARGETLIB_SRC
  src/mf_elasticity.cc
  ${CMAKE_CURRENT_BINARY_DIR}/version.cc
)

SET(CLEAN_UP_FILES
  # a custom list of globs, e.g. *.log *.vtk
  *.vtu
)

MESSAGE(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 9.2 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
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
MESSAGE("-- Code version ${GIT_TAG}")
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/src/version.cc.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.cc)

PROJECT(${TARGET})

#
# RPATHS: https://cmake.org/Wiki/CMake_RPATH_handling#Mac_OS_X_and_the_RPATH
#
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  ## cmake  --help-policy CMP0042
  set(CMAKE_MACOSX_RPATH ON)
ENDIF()

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

# make sure RPATHs are used on macOS
SET(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")

ADD_LIBRARY(${TARGETLIB} SHARED
    ${TARGETLIB_SRC}
  )

ADD_EXECUTABLE(${TARGET}
    src/main.cc
  )

DEAL_II_SETUP_TARGET(${TARGETLIB})
DEAL_II_SETUP_TARGET(${TARGET})

TARGET_LINK_LIBRARIES(${TARGET}
    ${TARGETLIB}
  )

INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/include)

#
# Custom "debug" and "release" make targets:
#
ADD_CUSTOM_TARGET(debug
COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
)

ADD_CUSTOM_TARGET(release
COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
COMMENT "Switch CMAKE_BUILD_TYPE to Release"
)

# Benchmarks
#ADD_SUBDIRECTORY(benchmark)

# Enable testing and descent into tests/ subdirectory:
ENABLE_TESTING()
INCLUDE(CTest)
ADD_SUBDIRECTORY(tests)
