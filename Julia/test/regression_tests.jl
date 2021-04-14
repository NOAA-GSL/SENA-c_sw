#=
    Regression tests verify that the sw_driver program successfully ran for each datasize
=#

# include source file of the driver program
include("../src/c_sw.jl")

using Test, TOML, .program

input_file_name = joinpath(@__DIR__, "data/inputs/inputs.toml")

configfile = TOML.parsefile(input_file_name)

# Iterate over the data sets in the config file
for (dataset, dataIODict) in configfile
    @testset "$dataset Regression " begin
        # This conditional is a crude way to avoid the interpolation in the ref kernel 
        if dataset[end] == '0'
            main(dataIODict)
            @test true
        end
    end 
end

return true
