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
* Internet access is required to install the Julia package dependencies. 


### Basic build procedure (from the directory containing this file)

From this directory, open the Julia REPL.

```bash
$ julia --project
```
From within the Julia REPL, run the following commands. 

```julia 
using Pkg 
Pkg.instantiate()
```

If you'd prefer not to enter the Julia REPL: 

```bash 
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Machines that do not have Julia installed

#### TODO:
* Create a 'standalone' Julia implementation.

* Use: [BinaryBuilder.jl](https://github.com/JuliaPackaging/BinaryBuilder.jl)
       & [PackageCompiler.jl](https://github.com/JuliaLang/PackageCompiler.jl)

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

From the Julia REPL: 

```julia 
include("test/test_c_sw.jl")
```

If you'd prefer not to enter the Julia REPL, run: 
```bash
$ julia test/test_c_sw.jl
```

To run a specific test call julia with the `test/single_regression.jl` file providing the argument of the dataset you'd like to test. The available datasets are contained in the [inputs.TOML](test/data/inputsinputs/inputs.toml) file.

For example: 

```bash 
$ julia test/single_regression.jl "c_sw_12x24_0"
```

## Build and test script

For convenience, a build script is provided that builds the code and runs the test suite.

```bash
sh build.sh
```

## Installation and running


## NOTES:

## Here is a list of the files and what they contain.


- `src/` contains the kernel source code
- `test/` contains the test files
- `test/test_input` contains the test input TOML configuration file
- `../test/data/output` is where test output data is written

## Troubleshooting

1. All tests fail on my machine.

    Check to make sure git-lfs is installed and that all files in `test/data/inputs` are NetCDF 
    data files and are not text. Run `git lfs pull` to download NetCDF files if necessary.

2. I get `Skipping object checkout, Git LFS is not installed.` when running `git lfs pull`

    Run `git lfs install` to perform the one-time installation that git-lfs requires per user per machine.

3. I get `git: 'lfs' is not a git command.` when running `git lfs pull`

    Your version of git does not support git-lfs. Install git-lfs or load a version of git that supports it.

4. I get `git-lfs smudge -- 'test/data/inputs/c_sw_12x24.nc': git-lfs: command not found` when cloning.

    Your version of git does not support git-lfs. Install git-lfs or load a version of git that supports it.

