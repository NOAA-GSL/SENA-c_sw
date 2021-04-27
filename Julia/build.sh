#!/bin/bash

export JULIA_NUM_THREADS=4

julia --project=. -e 'using Pkg; Pkg.instantiate()'

julia --project=. -e 'mkdir("test/test_output")'

julia --project=. -e 'mkdir("../data/outputs")'

julia --project=. test/test_c_sw.jl
