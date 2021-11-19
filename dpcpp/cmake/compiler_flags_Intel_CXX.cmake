####################################################################
# COMMON FLAGS
####################################################################

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -traceback -fexceptions -g -debug full -qopenmp" )

####################################################################
# RELEASE FLAGS
####################################################################

set( CMAKE_CXX_FLAGS_RELEASE "-O3 -xHost -qopt-zmm-usage=high" )

####################################################################
# DEBUG FLAGS
####################################################################

set( CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -debug -nolib-inline -fno-inline-functions -prec-div -prec-sqrt -check=conversions,stack,uninit -fp-stack-check" )