

list( APPEND c_sw_test_data
  data/inputs/c_sw_12x24.nc
  data/inputs/c_sw_24x24.nc
  data/inputs/c_sw_48x24.nc
  data/inputs/c_sw_48x48.nc
)

list( APPEND c_sw_test_input
  test_input/c_sw_12x24_0.nl
  test_input/c_sw_24x24_0.nl
  test_input/c_sw_48x24_0.nl
  test_input/c_sw_48x48_0.nl
  test_input/c_sw_12x24_3.nl
  test_input/c_sw_24x24_3.nl
  test_input/c_sw_48x24_3.nl
  test_input/c_sw_48x48_3.nl
)

list( APPEND c_sw_test_output
  test_output/c_sw_12x24_0.test
  test_output/c_sw_24x24_0.test
  test_output/c_sw_48x24_0.test
  test_output/c_sw_48x48_0.test
  test_output/c_sw_12x24_3.test
  test_output/c_sw_24x24_3.test
  test_output/c_sw_48x24_3.test
  test_output/c_sw_48x48_3.test
)
 
add_subdirectory( tools )


# Create data directory for kernel input and symlink all files
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data/inputs)
foreach(FILENAME ${c_sw_test_data})
execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink
${CMAKE_CURRENT_SOURCE_DIR}/../../${FILENAME}
${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} )
endforeach(FILENAME)

# Create empty data directory for model test output
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data/outputs)

# Create test_input directory for test input and symlink all files
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_input)
foreach(FILENAME ${c_sw_test_input})
execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink
${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} )
endforeach(FILENAME)

# Create data directory for reference output and symlink all files
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_output)
foreach(FILENAME ${c_sw_test_output})
execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink
${CMAKE_CURRENT_SOURCE_DIR}/../../${FILENAME}
${CMAKE_BINARY_DIR}/${FILENAME} )
endforeach(FILENAME)

# Get parent directory variable for comparing baselines tests
get_filename_component(PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)

set(COMPARE_SH ${CMAKE_BINARY_DIR}/bin/c_sw_compare.sh)

# Add unit tests here.
list (APPEND test_src_files
  ${PROJECT_SOURCE_DIR}/src/interpolate.f90
  ${PROJECT_SOURCE_DIR}/test/src/interpolateTest.f90
)
add_executable (interpolateTest ${test_src_files} )

target_include_directories( interpolateTest SYSTEM PUBLIC ${NetCDF_INCLUDE_DIRS} )

target_link_libraries( interpolateTest PUBLIC NetCDF::NetCDF_Fortran)
if(OpenMP_FOUND)
    target_link_libraries( interpolateTest PUBLIC OpenMP::OpenMP_Fortran)
    target_link_libraries( interpolateTest PUBLIC OpenMP::OpenMP_C)
endif()

if ( GPTL_FOUND )
   target_link_libraries( interpolateTest PUBLIC GPTL::GPTL )
endif()

if ( MPI_FOUND )
  target_link_libraries( interpolateTest PUBLIC MPI::MPI_Fortran )
  target_link_libraries( interpolateTest PUBLIC MPI::MPI_C )
endif()


# Tests when MPI is enabled
if ( ENABLE_MPI )

  # 12 x 24 workload
  add_test(NAME regression_12x24_0
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_12x24_0.nl")
  add_test(NAME compare_12x24_0
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_0.log.0000 ${PARENT_DIR}/baselines/c_sw_12x24_0.test)

  # 24 x 24 workload
  add_test(NAME regression_24x24_0
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_24x24_0.nl")
  add_test(NAME compare_24x24_0
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_0.log.0000 ${PARENT_DIR}/baselines/c_sw_24x24_0.test)

  # 48 x 24 workload
  add_test(NAME regression_48x24_0
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_48x24_0.nl")
  add_test(NAME compare_48x24_0
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_0.log.0000 ${PARENT_DIR}/baselines/c_sw_48x24_0.test)

  # 48 x 48 workload
  add_test(NAME regression_48x48_0
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_48x48_0.nl")
  add_test(NAME compare_48x48_0
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_0.log.0000 ${PARENT_DIR}/baselines/c_sw_48x48_0.test)

  # 12 x 24 workload with 4 MPI ranks
  add_test(NAME regression_12x24_0_procs4
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_12x24_0.nl")
  add_test(NAME compare_12x24_0_rank1
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_0.log.0000 ${PARENT_DIR}/baselines/c_sw_12x24_0.test)
  add_test(NAME compare_12x24_0_rank2
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_0.log.0001 ${PARENT_DIR}/baselines/c_sw_12x24_0.test)
  add_test(NAME compare_12x24_0_rank3
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_0.log.0002 ${PARENT_DIR}/baselines/c_sw_12x24_0.test)
  add_test(NAME compare_12x24_0_rank4
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_0.log.0003 ${PARENT_DIR}/baselines/c_sw_12x24_0.test)

  # 24 x 24 workload with 4 MPI ranks
  add_test(NAME regression_24x24_0_procs4
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_24x24_0.nl")
  add_test(NAME compare_24x24_0_rank1
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_0.log.0000 ${PARENT_DIR}/baselines/c_sw_24x24_0.test)
  add_test(NAME compare_24x24_0_rank2
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_0.log.0001 ${PARENT_DIR}/baselines/c_sw_24x24_0.test)
  add_test(NAME compare_24x24_0_rank3
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_0.log.0002 ${PARENT_DIR}/baselines/c_sw_24x24_0.test)
  add_test(NAME compare_24x24_0_rank4
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_0.log.0003 ${PARENT_DIR}/baselines/c_sw_24x24_0.test)

  # 48 x 24 workload with 4 MPI ranks
  add_test(NAME regression_48x24_0_procs4
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_48x24_0.nl")
  add_test(NAME compare_48x24_0_rank1
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_0.log.0000 ${PARENT_DIR}/baselines/c_sw_48x24_0.test)
  add_test(NAME compare_48x24_0_rank2
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_0.log.0001 ${PARENT_DIR}/baselines/c_sw_48x24_0.test)
  add_test(NAME compare_48x24_0_rank3
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_0.log.0002 ${PARENT_DIR}/baselines/c_sw_48x24_0.test)
  add_test(NAME compare_48x24_0_rank4
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_0.log.0003 ${PARENT_DIR}/baselines/c_sw_48x24_0.test)

  # 48 x 48 workload with 4 MPI ranks
  add_test(NAME regression_48x48_0_procs4
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_48x48_0.nl")
  add_test(NAME compare_48x48_0_rank1
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_0.log.0000 ${PARENT_DIR}/baselines/c_sw_48x48_0.test)
  add_test(NAME compare_48x48_0_rank2
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_0.log.0001 ${PARENT_DIR}/baselines/c_sw_48x48_0.test)
  add_test(NAME compare_48x48_0_rank3
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_0.log.0002 ${PARENT_DIR}/baselines/c_sw_48x48_0.test)
  add_test(NAME compare_48x48_0_rank4
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_0.log.0003 ${PARENT_DIR}/baselines/c_sw_48x48_0.test)

# Tests when MPI is not enabled
else()

# 12 x 24 workload
  add_test(NAME regression_12x24_0
           COMMAND bash -c "../src/c_sw test_input/c_sw_12x24_0.nl")
  add_test(NAME compare_12x24_0
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_0.log ${PARENT_DIR}/baselines/c_sw_12x24_0.test)

  # 24 x 24 workload
  add_test(NAME regression_24x24_0
           COMMAND bash -c "../src/c_sw test_input/c_sw_24x24_0.nl")
  add_test(NAME compare_24x24_0
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_0.log ${PARENT_DIR}/baselines/c_sw_24x24_0.test)

  # 48 x 24 workload
  add_test(NAME regression_48x24_0
           COMMAND bash -c "../src/c_sw test_input/c_sw_48x24_0.nl")
  add_test(NAME compare_48x24_0
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_0.log ${PARENT_DIR}/baselines/c_sw_48x24_0.test)

  # 48 x 48 workload
  add_test(NAME regression_48x48_0
           COMMAND bash -c "../src/c_sw test_input/c_sw_48x48_0.nl")
  add_test(NAME compare_48x48_0
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_0.log ${PARENT_DIR}/baselines/c_sw_48x48_0.test)

  # 12 x 24 workload
  add_test(NAME regression_12x24_3
           COMMAND bash -c "../src/c_sw test_input/c_sw_12x24_3.nl")
  add_test(NAME compare_12x24_3
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_3.log ${PARENT_DIR}/baselines/c_sw_12x24_3.test)

  # 24 x 24 workload
  add_test(NAME regression_24x24_3
           COMMAND bash -c "../src/c_sw test_input/c_sw_24x24_3.nl")
  add_test(NAME compare_24x24_3
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_3.log ${PARENT_DIR}/baselines/c_sw_24x24_3.test)

  # 48 x 24 workload
  add_test(NAME regression_48x24_3
           COMMAND bash -c "../src/c_sw test_input/c_sw_48x24_3.nl")
  add_test(NAME compare_48x24_3
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_3.log ${PARENT_DIR}/baselines/c_sw_48x24_3.test)

  # 48 x 48 workload
  add_test(NAME regression_48x48_3
           COMMAND bash -c "../src/c_sw test_input/c_sw_48x48_3.nl")
  add_test(NAME compare_48x48_3
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_3.log ${PARENT_DIR}/baselines/c_sw_48x48_3.test)
  
  add_test(NAME InterpolateTest 
           COMMAND bash -c "set -o pipefail; ../test/interpolateTest 2>&1 | tee test_output/interpolateTest.out")

endif()
