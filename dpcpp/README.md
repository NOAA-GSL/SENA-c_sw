This is a Data Parallel C++ implementation of the `c_sw` kernel extracted from the FV3 model.
There are currently two offload models implemented:
OUTER offloads the entire c_sw call for each k index.
MIXED collapses the k index into the c_sw calls and offloads only key loops
The offload model is currently set in source.

## Selecting the offload model

new_sw_core.cpp contains a global constant "tcase" which select which offload model is tested.
tcase=OUTER selects the OUTER model
tcase=MIXED selects the MIXED model

## Building the kernel

The build process requires a Data Parallel C++ compiler.  Please ensure that one is set in
path before proceeding.

This kernel uses an out-of-source cmake build, meaning that the build must be done in 
directory that is not in the source tree.

The build procedure is (from the directory containing this file):

```
rm -rf build ; mkdir build ; cd build
module load <compiler> netcdf cmake
export CXX=dpcpp
export CC=<gcc | icc>
export FC=<gfortran | ifort>
cmake -DCMAKE_BUILD_TYPE=<debug | release> ..
make VERBOSE=1
```

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
