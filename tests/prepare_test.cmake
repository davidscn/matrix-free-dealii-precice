foreach(_required_variable
    TEST_WORK_ROOT TEST_WORK_DIR CASE_NAME PARAMETER_FILE_ONE)
  if(NOT DEFINED "${_required_variable}" OR
     "${${_required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${_required_variable} must be defined")
  endif()
endforeach()

get_filename_component(_work_root "${TEST_WORK_ROOT}" ABSOLUTE)
get_filename_component(_work_directory "${TEST_WORK_DIR}" ABSOLUTE)
string(FIND "${_work_directory}/" "${_work_root}/" _work_root_position)
if(NOT _work_root_position EQUAL 0 OR _work_directory STREQUAL _work_root)
  message(FATAL_ERROR
    "Refusing to clean test directory outside ${_work_root}: "
    "${_work_directory}")
endif()

# Every run starts from a newly staged directory. This removes all solver
# output, logs, comparison files, and the preCICE communication directory.
file(REMOVE_RECURSE "${_work_directory}")
file(MAKE_DIRECTORY "${_work_directory}/${CASE_NAME}")

foreach(_parameter_file PARAMETER_FILE_ONE PARAMETER_FILE_TWO)
  if(DEFINED "${_parameter_file}" AND
     NOT "${${_parameter_file}}" STREQUAL "")
    if(NOT EXISTS "${${_parameter_file}}")
      message(FATAL_ERROR
        "Test parameter file does not exist: ${${_parameter_file}}")
    endif()
    get_filename_component(
      _parameter_file_name "${${_parameter_file}}" NAME)
    configure_file(
      "${${_parameter_file}}"
      "${_work_directory}/${CASE_NAME}/${_parameter_file_name}"
      COPYONLY)
  endif()
endforeach()

if(DEFINED CONFIG_FILE AND NOT "${CONFIG_FILE}" STREQUAL "")
  if(NOT DEFINED CONFIG_DESTINATION OR
     "${CONFIG_DESTINATION}" STREQUAL "")
    message(FATAL_ERROR
      "CONFIG_DESTINATION is required when CONFIG_FILE is provided")
  endif()
  if(NOT EXISTS "${CONFIG_FILE}")
    message(FATAL_ERROR
      "preCICE configuration file does not exist: ${CONFIG_FILE}")
  endif()
  configure_file(
    "${CONFIG_FILE}"
    "${_work_directory}/${CONFIG_DESTINATION}"
    COPYONLY)
endif()
