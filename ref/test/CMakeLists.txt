

list( APPEND c_sw_test_data
  test/data/inputs/c_sw_12x24.nc
  test/data/inputs/c_sw_24x24.nc
  test/data/inputs/c_sw_48x24.nc
  test/data/inputs/c_sw_48x48.nc
)

list( APPEND c_sw_test_input
  test_input/c_sw_12x24_0_1x4.nl
  test_input/c_sw_12x24_0_2x2.nl
  test_input/c_sw_12x24_0.nl
  test_input/c_sw_24x24_0_1x4.nl
  test_input/c_sw_24x24_0_2x2.nl
  test_input/c_sw_24x24_0.nl
  test_input/c_sw_48x24_0_1x4.nl
  test_input/c_sw_48x24_0_2x2.nl
  test_input/c_sw_48x24_0.nl
  test_input/c_sw_48x48_0_1x4.nl
  test_input/c_sw_48x48_0_2x2.nl
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
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test/data)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test/data/inputs)
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

# Create test_output directory for test output and symlink all files
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_output)
foreach(FILENAME ${c_sw_test_output})
    execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink
           ${CMAKE_CURRENT_SOURCE_DIR}/../../test/${FILENAME}
           ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} )
endforeach()

set(COMPARE_SH ${CMAKE_BINARY_DIR}/bin/c_sw_compare.sh)

# Add unit tests here.
list (APPEND interpolateTest_src_files
  ${PROJECT_SOURCE_DIR}/src/interpolate.f90
  ${PROJECT_SOURCE_DIR}/test/src/interpolateTest.f90
)
add_executable (interpolateTest ${interpolateTest_src_files} )
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

if ( MPI_FOUND )
  list (APPEND exchangeTest_src_files
    ${PROJECT_SOURCE_DIR}/src/sw_core.F90
    ${PROJECT_SOURCE_DIR}/src/netCDFModule.f90
    ${PROJECT_SOURCE_DIR}/src/interpolate.f90
    ${PROJECT_SOURCE_DIR}/test/src/exchangeTest.F90
  )
  add_executable (exchangeTest ${exchangeTest_src_files} )
  target_include_directories( exchangeTest SYSTEM PUBLIC ${NetCDF_INCLUDE_DIRS} )
  target_link_libraries( exchangeTest PUBLIC NetCDF::NetCDF_Fortran)

  if(OpenMP_FOUND)
      target_link_libraries( exchangeTest PUBLIC OpenMP::OpenMP_Fortran)
      target_link_libraries( exchangeTest PUBLIC OpenMP::OpenMP_C)
  endif()

  if ( GPTL_FOUND )
     target_link_libraries( exchangeTest PUBLIC GPTL::GPTL )
  endif()

  if ( MPI_FOUND )
    target_link_libraries( exchangeTest PUBLIC MPI::MPI_Fortran )
    target_link_libraries( exchangeTest PUBLIC MPI::MPI_C )
  endif()

endif()

# Tests when MPI is enabled
if ( ENABLE_MPI )

  # 12 x 24 workload
  add_test(NAME regression_12x24_0
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_12x24_0.nl")
  add_test(NAME compare_12x24_0
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_0.log.0000 test_output/c_sw_12x24_0.test)

  # 24 x 24 workload
  add_test(NAME regression_24x24_0
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_24x24_0.nl")
  add_test(NAME compare_24x24_0
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_0.log.0000 test_output/c_sw_24x24_0.test)

  # 48 x 24 workload
  add_test(NAME regression_48x24_0
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_48x24_0.nl")
  add_test(NAME compare_48x24_0
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_0.log.0000 test_output/c_sw_48x24_0.test)

  # 48 x 48 workload
  add_test(NAME regression_48x48_0
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_48x48_0.nl")
  add_test(NAME compare_48x48_0
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_0.log.0000 test_output/c_sw_48x48_0.test)

  # 12 x 24 workload with 4 MPI ranks (1x4)
  add_test(NAME regression_12x24_0_procs4_1x4
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_12x24_0_1x4.nl")
  add_test(NAME compare_12x24_0_rank1_1x4
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_0.log.0000 test_output/c_sw_12x24_0.test)
  add_test(NAME compare_12x24_0_rank2_1x4
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_0.log.0001 test_output/c_sw_12x24_0.test)
  add_test(NAME compare_12x24_0_rank3_1x4
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_0.log.0002 test_output/c_sw_12x24_0.test)
  add_test(NAME compare_12x24_0_rank4_1x4
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_0.log.0003 test_output/c_sw_12x24_0.test)

  # 12 x 24 workload with 4 MPI ranks (2x2)
  add_test(NAME regression_12x24_0_procs4_2x2
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_12x24_0_2x2.nl")
  add_test(NAME compare_12x24_0_rank1_2x2
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_0.log.0000 test_output/c_sw_12x24_0.test)
  add_test(NAME compare_12x24_0_rank2_2x2
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_0.log.0001 test_output/c_sw_12x24_0.test)
  add_test(NAME compare_12x24_0_rank3_2x2
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_0.log.0002 test_output/c_sw_12x24_0.test)
  add_test(NAME compare_12x24_0_rank4_2x2
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_0.log.0003 test_output/c_sw_12x24_0.test)

  # 24 x 24 workload with 4 MPI ranks (1x4)
  add_test(NAME regression_24x24_0_procs4_1x4
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_24x24_0_1x4.nl")
  add_test(NAME compare_24x24_0_rank1_1x4
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_0.log.0000 test_output/c_sw_24x24_0.test)
  add_test(NAME compare_24x24_0_rank2_1x4
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_0.log.0001 test_output/c_sw_24x24_0.test)
  add_test(NAME compare_24x24_0_rank3_1x4
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_0.log.0002 test_output/c_sw_24x24_0.test)
  add_test(NAME compare_24x24_0_rank4_1x4
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_0.log.0003 test_output/c_sw_24x24_0.test)

  # 24 x 24 workload with 4 MPI ranks (2x2)
  add_test(NAME regression_24x24_0_procs4_2x2
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_24x24_0_2x2.nl")
  add_test(NAME compare_24x24_0_rank1_2x2
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_0.log.0000 test_output/c_sw_24x24_0.test)
  add_test(NAME compare_24x24_0_rank2_2x2
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_0.log.0001 test_output/c_sw_24x24_0.test)
  add_test(NAME compare_24x24_0_rank3_2x2
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_0.log.0002 test_output/c_sw_24x24_0.test)
  add_test(NAME compare_24x24_0_rank4_2x2
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_0.log.0003 test_output/c_sw_24x24_0.test)

  # 48 x 24 workload with 4 MPI ranks (1x4)
  add_test(NAME regression_48x24_0_procs4_1x4
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_48x24_0_1x4.nl")
  add_test(NAME compare_48x24_0_rank1_1x4
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_0.log.0000 test_output/c_sw_48x24_0.test)
  add_test(NAME compare_48x24_0_rank2_1x4
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_0.log.0001 test_output/c_sw_48x24_0.test)
  add_test(NAME compare_48x24_0_rank3_1x4
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_0.log.0002 test_output/c_sw_48x24_0.test)
  add_test(NAME compare_48x24_0_rank4_1x4
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_0.log.0003 test_output/c_sw_48x24_0.test)

  # 48 x 24 workload with 4 MPI ranks (2x2)
  add_test(NAME regression_48x24_0_procs4_2x2
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_48x24_0_2x2.nl")
  add_test(NAME compare_48x24_0_rank1_2x2
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_0.log.0000 test_output/c_sw_48x24_0.test)
  add_test(NAME compare_48x24_0_rank2_2x2
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_0.log.0001 test_output/c_sw_48x24_0.test)
  add_test(NAME compare_48x24_0_rank3_2x2
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_0.log.0002 test_output/c_sw_48x24_0.test)
  add_test(NAME compare_48x24_0_rank4_2x2
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_0.log.0003 test_output/c_sw_48x24_0.test)

  # 48 x 48 workload with 4 MPI ranks (1x4)
  add_test(NAME regression_48x48_0_procs4_1x4
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_48x48_0_1x4.nl")
  add_test(NAME compare_48x48_0_rank1_1x4
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_0.log.0000 test_output/c_sw_48x48_0.test)
  add_test(NAME compare_48x48_0_rank2_1x4
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_0.log.0001 test_output/c_sw_48x48_0.test)
  add_test(NAME compare_48x48_0_rank3_1x4
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_0.log.0002 test_output/c_sw_48x48_0.test)
  add_test(NAME compare_48x48_0_rank4_1x4
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_0.log.0003 test_output/c_sw_48x48_0.test)

  # 48 x 48 workload with 4 MPI ranks (2x2)
  add_test(NAME regression_48x48_0_procs4_2x2
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ../src/c_sw test_input/c_sw_48x48_0_2x2.nl")
  add_test(NAME compare_48x48_0_rank1_2x2
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_0.log.0000 test_output/c_sw_48x48_0.test)
  add_test(NAME compare_48x48_0_rank2_2x2
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_0.log.0001 test_output/c_sw_48x48_0.test)
  add_test(NAME compare_48x48_0_rank3_2x2
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_0.log.0002 test_output/c_sw_48x48_0.test)
  add_test(NAME compare_48x48_0_rank4_2x2
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_0.log.0003 test_output/c_sw_48x48_0.test)

  add_test(NAME exchange
           COMMAND bash -c "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ../test/exchangeTest")

# Tests when MPI is not enabled
else()

# 12 x 24 workload
  add_test(NAME regression_12x24_0
           COMMAND bash -c "../src/c_sw test_input/c_sw_12x24_0.nl")
  add_test(NAME compare_12x24_0
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_0.log test_output/c_sw_12x24_0.test)

  # 24 x 24 workload
  add_test(NAME regression_24x24_0
           COMMAND bash -c "../src/c_sw test_input/c_sw_24x24_0.nl")
  add_test(NAME compare_24x24_0
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_0.log test_output/c_sw_24x24_0.test)

  # 48 x 24 workload
  add_test(NAME regression_48x24_0
           COMMAND bash -c "../src/c_sw test_input/c_sw_48x24_0.nl")
  add_test(NAME compare_48x24_0
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_0.log test_output/c_sw_48x24_0.test)

  # 48 x 48 workload
  add_test(NAME regression_48x48_0
           COMMAND bash -c "../src/c_sw test_input/c_sw_48x48_0.nl")
  add_test(NAME compare_48x48_0
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_0.log test_output/c_sw_48x48_0.test)

  # 12 x 24 workload
  add_test(NAME regression_12x24_3
           COMMAND bash -c "../src/c_sw test_input/c_sw_12x24_3.nl")
  add_test(NAME compare_12x24_3
           COMMAND ${COMPARE_SH} test_output/c_sw_12x24_3.log test_output/c_sw_12x24_3.test)

  # 24 x 24 workload
  add_test(NAME regression_24x24_3
           COMMAND bash -c "../src/c_sw test_input/c_sw_24x24_3.nl")
  add_test(NAME compare_24x24_3
           COMMAND ${COMPARE_SH} test_output/c_sw_24x24_3.log test_output/c_sw_24x24_3.test)

  # 48 x 24 workload
  add_test(NAME regression_48x24_3
           COMMAND bash -c "../src/c_sw test_input/c_sw_48x24_3.nl")
  add_test(NAME compare_48x24_3
           COMMAND ${COMPARE_SH} test_output/c_sw_48x24_3.log test_output/c_sw_48x24_3.test)

  # 48 x 48 workload
  add_test(NAME regression_48x48_3
           COMMAND bash -c "../src/c_sw test_input/c_sw_48x48_3.nl")
  add_test(NAME compare_48x48_3
           COMMAND ${COMPARE_SH} test_output/c_sw_48x48_3.log test_output/c_sw_48x48_3.test)
  
  add_test(NAME InterpolateTest 
           COMMAND bash -c "set -o pipefail; ../test/interpolateTest 2>&1 | tee test_output/interpolateTest.out")

endif()
