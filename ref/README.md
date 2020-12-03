This is the reference implementation of the `c_sw` kernel extracted from the FV3 model.
It is an excellent kernel for OpenMP analysis. The number of threads is determined by
the OMP_NUM_THREADS environment variable, which defaults to the number of hyperthreads
on the machine if it is not set.  Currently the number of threads is hard-coded to 4 in
the build script when running "ctest", but users can customize it at runtime for other
runs.

## Building the kernel (HPC systems with modules)

This kernel uses an out-of-source cmake build, meaning that the build must be done in 
directory that is not in the source tree.

The build procedure is (from the directory containing this file):

```
rm -rf build ; mkdir build ; cd build
module load <compiler> netcdf cmake
export CC=<gcc | icc>
export FC=<gfortran | ifort>
cmake -DCMAKE_BUILD_TYPE=<debug | release> ..
make VERBOSE=1
```

### Building the kernel (Systems without modules)

```
rm -rf build ; mkdir build ; cd build
export CC=<gcc | icc>
export FC=<gfortran | ifort> 
cmake -DCMAKE_BUILD_TYPE=<debug | release> ..
make VERBOSE=1
```

### Additional description for build configurations 

Some alterations could potentially need to be made in regards to the CC and FC compilers  
* export CC= < appropriate compiler version being used (ie. gcc-10) > 
* export FC= < appropriate compiler being used >


## Testing the kernel

Ensure system is quipped with LFS to run the .nl files in test/test_input 

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
