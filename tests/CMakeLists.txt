find_program(BASH_EXECUTABLE bash)
if(NOT BASH_EXECUTABLE)
  message(FATAL_ERROR "bash is required to run the coupled regression tests")
endif()

find_program(NUMDIFF_EXECUTABLE numdiff)
if(NOT NUMDIFF_EXECUTABLE)
  message(FATAL_ERROR "numdiff is required to compare regression-test output")
endif()

set(COUPLED_TEST_TIMEOUT 120 CACHE STRING
  "Wall-clock timeout in seconds for each coupled regression test")
set(COUPLED_TEST_MPI_RANKS 4 CACHE STRING
  "Number of MPI ranks used by each parallel test participant")
set(SOLID_TEST_CG_RELATIVE_TOLERANCE 0.01 CACHE STRING
  "Allowed relative increase in average CG iterations")
set(SOLID_TEST_CG_ABSOLUTE_TOLERANCE 1 CACHE STRING
  "Minimum allowed absolute increase in average CG iterations")
set(HEAT_TEST_ABSOLUTE_TOLERANCE 2e-13 CACHE STRING
  "Absolute tolerance used for heat error regression values")

add_executable(dummy_tester EXCLUDE_FROM_ALL solid/dummy_tester.cpp)
target_link_libraries(dummy_tester PRIVATE precice::precice)

# CTest deliberately builds only the test participant. The application
# executables have to be built before running their corresponding tests.
add_test(
  NAME test_dependency.dummy_tester
  COMMAND
    "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}"
    --target dummy_tester --config "$<CONFIG>")
set_tests_properties(test_dependency.dummy_tester PROPERTIES
  FIXTURES_SETUP dummy_tester_executable
  LABELS "solid;test_dependency"
  TIMEOUT "${COUPLED_TEST_TIMEOUT}")

set(_test_work_root "${CMAKE_CURRENT_BINARY_DIR}/work")
set(_prepare_test_script "${CMAKE_CURRENT_SOURCE_DIR}/prepare_test.cmake")
set(_run_test_script "${CMAKE_CURRENT_SOURCE_DIR}/run_coupled_test.sh")

function(coupled_test_command output_variable target parameter_file mpi_ranks)
  if("${mpi_ranks}" STREQUAL "1")
    set(_command "$<TARGET_FILE:${target}>" "${parameter_file}")
  else()
    set(_command
      "${DEAL_II_MPIEXEC}"
      ${DEAL_II_MPIEXEC_NUMPROC_FLAG} "${mpi_ranks}"
      ${DEAL_II_MPIEXEC_PREFLAGS}
      "$<TARGET_FILE:${target}>" "${parameter_file}"
      ${DEAL_II_MPIEXEC_POSTFLAGS})
  endif()
  set("${output_variable}" "${_command}" PARENT_SCOPE)
endfunction()

function(add_test_clean_fixture
    test_name label fixture work_directory case_name
    parameter_file_one parameter_file_two config_file config_destination)
  set(_clean_test_name "clean.${test_name}")
  add_test(
    NAME "${_clean_test_name}"
    COMMAND
      "${CMAKE_COMMAND}"
      "-DTEST_WORK_ROOT=${_test_work_root}"
      "-DTEST_WORK_DIR=${work_directory}"
      "-DCASE_NAME=${case_name}"
      "-DPARAMETER_FILE_ONE=${parameter_file_one}"
      "-DPARAMETER_FILE_TWO=${parameter_file_two}"
      "-DCONFIG_FILE=${config_file}"
      "-DCONFIG_DESTINATION=${config_destination}"
      -P "${_prepare_test_script}")
  set_tests_properties("${_clean_test_name}" PROPERTIES
    FIXTURES_SETUP "${fixture}"
    LABELS "${label};clean"
    TIMEOUT 30)
endfunction()

function(add_solid_regression_test case_name mode mpi_ranks)
  set(_test_name "solid.${case_name}.${mode}")
  set(_fixture "fixture_${_test_name}")
  set(_work_directory "${_test_work_root}/${_test_name}")
  set(_parameter_file
    "${CMAKE_CURRENT_SOURCE_DIR}/solid/${case_name}/${case_name}.prm")
  set(_reference_file
    "${CMAKE_CURRENT_SOURCE_DIR}/solid/${case_name}/${case_name}_${mode}.output")

  add_test_clean_fixture(
    "${_test_name}" solid "${_fixture}" "${_work_directory}" "${case_name}"
    "${_parameter_file}" ""
    "${CMAKE_CURRENT_SOURCE_DIR}/solid/precice-config_unified.xml"
    "precice-config.xml")

  coupled_test_command(
    _solid_command solid "${case_name}/${case_name}.prm" "${mpi_ranks}")

  add_test(
    NAME "${_test_name}"
    COMMAND
      "${BASH_EXECUTABLE}" "${_run_test_script}"
      solid "${_work_directory}" "${case_name}/output"
      "${_reference_file}" "${NUMDIFF_EXECUTABLE}"
      "${SOLID_TEST_CG_RELATIVE_TOLERANCE},${SOLID_TEST_CG_ABSOLUTE_TOLERANCE}"
      -- ${_solid_command}
      --and "$<TARGET_FILE:dummy_tester>")
  math(EXPR _processor_count "${mpi_ranks} + 1")
  set_tests_properties("${_test_name}" PROPERTIES
    FIXTURES_REQUIRED "${_fixture};dummy_tester_executable"
    LABELS "solid;${mode}"
    PROCESSORS "${_processor_count}"
    TIMEOUT "${COUPLED_TEST_TIMEOUT}"
    WORKING_DIRECTORY "${_work_directory}")
endfunction()

function(add_heat_regression_test case_name mode mpi_ranks)
  set(_test_name "heat.${case_name}.${mode}")
  set(_fixture "fixture_${_test_name}")
  set(_work_directory "${_test_work_root}/${_test_name}")
  set(_dirichlet_parameter
    "${CMAKE_CURRENT_SOURCE_DIR}/heat/${case_name}/dirichlet.prm")
  set(_neumann_parameter
    "${CMAKE_CURRENT_SOURCE_DIR}/heat/${case_name}/neumann.prm")
  set(_reference_file
    "${CMAKE_CURRENT_SOURCE_DIR}/heat/${case_name}/${case_name}_${mode}.output")

  add_test_clean_fixture(
    "${_test_name}" heat "${_fixture}" "${_work_directory}" "${case_name}"
    "${_dirichlet_parameter}" "${_neumann_parameter}"
    "${CMAKE_CURRENT_SOURCE_DIR}/heat/${case_name}/precice-config.xml"
    "${case_name}/precice-config.xml")

  coupled_test_command(
    _dirichlet_command heat "${case_name}/dirichlet.prm" "${mpi_ranks}")
  coupled_test_command(
    _neumann_command heat "${case_name}/neumann.prm" "${mpi_ranks}")

  add_test(
    NAME "${_test_name}"
    COMMAND
      "${BASH_EXECUTABLE}" "${_run_test_script}"
      extract-errors "${_work_directory}" output
      "${_reference_file}" "${NUMDIFF_EXECUTABLE}"
      "${HEAT_TEST_ABSOLUTE_TOLERANCE}"
      -- ${_dirichlet_command}
      --and ${_neumann_command})
  math(EXPR _processor_count "2 * ${mpi_ranks}")
  set_tests_properties("${_test_name}" PROPERTIES
    FIXTURES_REQUIRED "${_fixture}"
    LABELS "heat;${mode}"
    PROCESSORS "${_processor_count}"
    TIMEOUT "${COUPLED_TEST_TIMEOUT}"
    WORKING_DIRECTORY "${_work_directory}")
endfunction()

set(_solid_cases
  turek_gmg_scalar_referential
  turek_gmg_tensor2
  turek_gmg_tensor4
  turek_gmg_tensor4_ns
  turek_gmg_tensor2_form0
  turek_jacobi_tensor2)

foreach(_case IN LISTS _solid_cases)
  add_solid_regression_test("${_case}" serial 1)
  add_solid_regression_test(
    "${_case}" parallel "${COUPLED_TEST_MPI_RANKS}")
endforeach()

set(_heat_cases
  partitioned-heat
  partitioned-heat-direct-access)

foreach(_case IN LISTS _heat_cases)
  add_heat_regression_test("${_case}" serial 1)
  add_heat_regression_test(
    "${_case}" parallel "${COUPLED_TEST_MPI_RANKS}")
endforeach()
