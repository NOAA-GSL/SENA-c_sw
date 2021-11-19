################################################################################
# c_sw
################################################################################

cmake_minimum_required( VERSION 3.10 )

project( c_sw LANGUAGES Fortran C CXX)

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )
set( CMAKE_DIRECTORY_LABELS ${PROJECT_NAME} )
#set( CMAKE_EXE_LINKER_FLAGS "-g -lifcore" )
set( CMAKE_EXE_LINKER_FLAGS "-g -lifcore -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice \
   -Xs \"-march=sse4.2\"" )
#set( CMAKE_EXE_LINKER_FLAGS "-g -lifcore -fsycl-targets=spir64_gen-unknown-unknown-sycldevice \
#    -Xs \"-device gen9\"" )

option( ENABLE_MPI "Simulate MPI execution" ON )


enable_testing()

include(${PROJECT_NAME}_compiler_flags)

find_package( OpenMP COMPONENTS C Fortran )
find_package( NetCDF REQUIRED COMPONENTS C Fortran )

if ( ENABLE_MPI )
  find_package( MPI COMPONENTS C CXX Fortran)
  add_compile_definitions(ENABLE_MPI)
endif()

add_subdirectory(src)
add_subdirectory(test)

set_target_properties( ${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX )
