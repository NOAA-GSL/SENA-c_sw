list( APPEND c_sw_src_files
new_sw_core_interfaces.f90
new_sw_core.cpp
CSWOuter.cpp
netCDFModule.f90
print_affinity.c
runnable.c
sw_core.F90
sw_driver.F90
interpolate.f90
)

SET(OBJS
  /global/panfs01/admin/opt/intel/oneAPI/2021.1.2/compiler/2021.1.2/linux/compiler/lib/intel64_lin/for_main.o
)

add_executable( ${PROJECT_NAME} ${OBJS} ${c_sw_src_files} )

#target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-v 
#  -fsycl-targets=spir64_gen-unknown-unknown-sycldevice>)
#target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-v 
#  -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice>)

target_include_directories( ${PROJECT_NAME} SYSTEM PUBLIC ${NetCDF_INCLUDE_DIRS} )
target_link_libraries(${PROJECT_NAME} PUBLIC NetCDF::NetCDF_Fortran)

if(OpenMP_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_Fortran)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_C)
endif()

if ( GPTL_FOUND )
   target_link_libraries( ${PROJECT_NAME} PUBLIC GPTL::GPTL )
endif()

if ( MPI_FOUND )
  target_link_libraries( ${PROJECT_NAME} PUBLIC MPI::MPI_Fortran )
  target_link_libraries( ${PROJECT_NAME} PUBLIC MPI::MPI_C )
endif()

install(TARGETS c_sw
        DESTINATION ${PROJECT_SOURCE_DIR}/exe)
