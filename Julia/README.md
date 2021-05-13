# The Julia C_SW kernel

NOTE: If you are reading this with a plain text editor, please note that this document is
formatted with Markdown syntax elements.  See https://www.markdownguide.org/cheat-sheet/
for more information.

This is the [Julia](https://github.com/JuliaLang/julia) implementation of the `c_sw` kernel extracted from the FV3 model.  The number of threads is determined by
the `JULIA_NUM_THREADS` environment variable, which defaults to 1.  Currently the number of threads is hard-coded to 4 in
the build script, but users can customize it at runtime for other runs.

## Dependencies
The following packages are required for building and running this kernel:

* [Julia v1.6](https://julialang.org/downloads/) 
* git
* [git-lfs](https://git-lfs.github.com/)


## Prerequisites
This code requires git-lfs. Before cloning the repository, verify that git-lfs is installed, by issuing the following command. This only needs to be done once per user per machine.

```bash
$ git lfs install
```

If the above gives an error you (or your systems administrator) may need to install git-lfs.

Some systems that use modules to manage software provide git with git-lfs support via a
module (e.g. `module load git`).  If you are using a system that uses modules, use
`module avail` to look for alternative versions of git that may have git-lfs support.

Make sure the files in `../test/data/inputs` are NetCDF data files (not text) before proceeding to
the build step. A simple way to do that is with the file command as shown below:

```
$ file test/data/inputs/*
../test/data/inputs/c_sw_12x24.nc: NetCDF Data Format data
../test/data/inputs/c_sw_24x24.nc: NetCDF Data Format data
../test/data/inputs/c_sw_48x24.nc: NetCDF Data Format data
../test/data/inputs/c_sw_48x48.nc: NetCDF Data Format data
```

**NOTE**: If you cloned the repository with a version of git without git-lfs installed, or before you ran `git lfs install`, you
must run the following command (with a version of git that does support git-lfs) from the base
of the repository to fetch the input data before proceeding to the build steps. Or you can
reclone the repository with git-lfs installed, instead.

```bash
$ git lfs pull
```

Alternatively, you can reclone the repository with git-lfs installed.

## Building the kernel

* [Julia v1.6](https://julialang.org/downloads/) is required to build & test the kernel.
* Internet access is required to install the Julia Pkg dependencies. 


### Basic build procedure (from the directory containing this file)

First, open the Julia REPL and activate this project specific environment:

```bash
$ julia --project
```
Within the Julia REPL, run the following commands:

```julia 
using Pkg 
Pkg.instantiate()
```

If you'd prefer not to enter the Julia REPL: 

```bash 
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Machines that do not have Julia installed

1. Install the binary release by downloading and expanding the tarball (e.g. `julia-1.6.1-linux-x86_64.tar.gz`) where you want to put it.
2. Create a module to load it.  Make sure to add both the `lib/` and `lib/julia` directories to `LD_LIBRARY_PATH`.  Suggested module (in LUA) would look like:
```LUA
local helpMsg = [[
Sets up the environment for julia version 1.6.1
]]
help(helpMsg)
prepend_path("PATH","/home/USER/opt/julia/1.6.1/bin")
prepend_path("LD_LIBRARY_PATH","/home/USER/opt/julia/1.6.1/lib")
prepend_path("LD_LIBRARY_PATH","/home/USER/opt/julia/1.6.1/lib/julia")
setenv("SSL_CERT_FILE", "/etc/ssl/certs/ca-bundle.crt")
```

### Machines that do not have internet access 

If you'd like to run the Julia implementation on an HPC that does not have internet access, first you must build it on a system with the same OS/Architecture (i.e. Linux x86 64 system) that does have an internet connection. Then, copy the resulting `~/.julia` over to the system you'd like to run on. This will preload the dependencies, so they will not need to be downloaded.

## Testing the kernel

Optionally set the number of threads you want to use for the tests. For example:

```bash
$ export JULIA_NUM_THREADS=4
```

Two additional steps are required to create the test output directories before testing the kernel:

```bash
$ julia --project=. -e 'mkdir("test/test_output")'
$ julia --project=. -e 'mkpath("../test/data/outputs")'
```

Once the test output directories are created, you can run the test file from the Julia REPL: 

```julia 
include("test/test_c_sw.jl")
```

If you'd prefer not to enter the Julia REPL, run: 
```bash
$ julia test/test_c_sw.jl
```

To run a specific test, call julia with the `test/single_regression.jl` file and provide the argument of the dataset you'd like to test. The available datasets are contained in the [inputs.TOML](test/data/inputs/inputs.toml) file.

For example: 

```bash 
$ julia test/single_regression.jl "c_sw_12x24_0"
```

## Build and test script

For convenience, a build script is provided that builds the code and runs the test suite.

```bash
sh build.sh
```

### Increasing Data Size

With the coming exascale supercomputers, faster computations and larger memory per node, 
this kernel has the functionality to increase the database under control of the user.  The 
variable `interpFactor` in the [inputs.TOML](test/data/inputs/inputs.toml) file allows the user to increase the size of the 
database.  The variable `interpFactor` is a positive integer or zero which controls the 
addition of interpolated points to the database.  If `interpFactor` is zero, no additional 
points are added to the database.

For example:

An interpolation factor of 1 means 1 new interpolated element is added between 
the original data.  So, a 3x3 matrix with `interpFactor=1` is interpolated
and becomes a 5x5 matrix

The x's are points from the original data.  The o's are points to be interpolated 
between the original data. 

```
[ x x x ]     ==> [x o x o x]
[ x x x ]         [o o o o o]
[ x x x ]         [x o x o x]
                  [o o o o o]
                  [x o x o x]
```

And an `interpFactor=2`, transforms a 3x3 matrix into a 7x7 matrix.

```
[ x x x ]     ==> [x o o x o o x]
[ x x x ]         [o o o o o o o]
[ x x x ]         [o o o o o o o]
                  [x o o x o o x]
                  [o o o o o o o]
                  [o o o o o o o]
                  [x o o x o o x]
```
The `interpFactor` can be modified in the `inputs.TOML` file without rebuilding the kernel; 
the interpolation is done "on the fly" before calling the computational kernel within 
SENA-c_sw.  The reference text logs are correct for interpFactor 0 and 3 currently.  
Alas, when you use another `interpFactor`, the numbers in the text logs change such that 
a comparison to the reference text logs will not be exact.  That said, the text logs 
should be numerically "close" to the reference texts, at least the first two or three 
significant digits of each floating-point number should match.

## NOTES:

### Here is a list of the files and what they contain:

- `src/` contains the kernel source code
- `test/` contains the test files
- `test/test_input` contains the test input TOML configuration file
- `../test/data/output` is where test output data is written

### Troubleshooting

1. All tests fail on my machine.

    Check to make sure git-lfs is installed and that all files in `../test/data/inputs` are NetCDF 
    data files and are not text. Run `git lfs pull` to download NetCDF files if necessary.

2. I get `Skipping object checkout, Git LFS is not installed.` when running `git lfs pull`

    Run `git lfs install` to perform the one-time installation that git-lfs requires per user per machine.

3. I get `git: 'lfs' is not a git command.` when running `git lfs pull`

    Your version of git does not support git-lfs. Install git-lfs or load a version of git that supports it.

4. I get `git-lfs smudge -- 'test/data/inputs/c_sw_12x24.nc': git-lfs: command not found` when cloning.

    Your version of git does not support git-lfs. Install git-lfs or load a version of git that supports it.

