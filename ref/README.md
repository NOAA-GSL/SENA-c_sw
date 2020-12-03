This is the reference implementation of the `c_sw` kernel extracted from the FV3 model.
It is an excellent kernel for OpenMP analysis. The number of threads is determined by
the OMP_NUM_THREADS environment variable, which defaults to the number of hyperthreads
on the machine if it is not set.  Currently the number of threads is hard-coded to 4 in
the build script when running "ctest", but users can customize it at runtime for other
runs.

# Building the kernel

This kernel uses an out-of-source cmake build, meaning that the build must be done in 
directory that is not in the source tree.

## Dependencies 
* C Compiler
* Fortran Compiler
* NetCDF 
* cmake 
* git-lfs 

## Machines that use modules to manage software

The build procedure is (from the directory containing this file):

```
rm -rf build ; mkdir build ; cd build
module load <compiler> netcdf cmake
export CC=<gcc | icc>
export FC=<gfortran | ifort>
cmake -DCMAKE_BUILD_TYPE=<debug | release> ..
make VERBOSE=1
```
You may need to load modules for your compiler, NetCDF, and/or cmake before following the steps above.

For example, "module load intel netcdf cmake".

## Machines that do not use modules to manage software

The build procedure is (from the directory containing this file):

```
rm -rf build ; mkdir build ; cd build
export CC=<name of C compiler>
export FC=<name of fortran compiler> 
cmake -DCMAKE_BUILD_TYPE=<debug | release> ..
make VERBOSE=1
```
If NetCDF is not installed in a standard location where cmake can find it, you may need to add the paths where your NetCDF and NetCDF-Fortran are installed to the CMAKE_PREFIX_PATH variable. For example:

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/path/to/netcdf:/path/to/netcdf-fortran


### Building on a Mac

By default, gcc points to the clang compiler on Mac. The clang compiler is not yet supported. To use the GNU compiler on Mac, depending on how the GNU compiler was installed, you may need to specify the c compiler name as gcc-$version. For example: gcc-10

## Testing the kernel

To run the test suite (from the build directory):

`ctest`

To run a specific test (for example):

`ctest -R regression_12x24`

To run a specific test with full output to get more information about a failure (for example):

`ctest -VV -R regression_12x24`

## Build and test script

For convenience, a build script is provided that builds the code and runs the test suite:

**(NOTE: This script may need to be modified, depending on how modules are set up on your system)**

`./build.sh <gcc | intel> <debug | release>`

## Installation and running

To (optionally) install the built executable into exe/

`make install`

To run the installed executable (for example):

`exe/c_sw ../test/test_input/c_sw_12x24.nl`

## NOTES:

1. The test suite does not measure performance, but reports how long each test takes to run.
2. Detailed performance timings are printed in the stdout when the kernel runs.
3. To view kernel output, either run ctest in verbose mode, or run the kernel manually.

## Here is a list of the files and what they contain.

- `cmake/` contains compiler flag settings and cmake helper scripts
- `src/` contains the kernel source code
- `test/` contains the tests, test input, and test output
- `test/data/outputs` is where test output data is written
- `test/test_input` contains the test namelist input files
- `test/test_output` contains the test baselines
- `exe/` contains the installed executable
