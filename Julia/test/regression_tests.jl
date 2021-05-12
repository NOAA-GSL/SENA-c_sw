#=
    Regression tests verify that the sw_driver program successfully ran for each datasize
=#

# include source file of the driver program
include("../src/Csw.jl")

using Test, TOML, .program

input_file_name = joinpath(@__DIR__, "data/inputs/inputs.toml")

configfile = TOML.parsefile(input_file_name)

# Iterate over the data sets in the config file
for (dataset, dataIODict) in configfile
    @testset "$dataset Regression " begin
        main(dataIODict)
        @test true
    end 
end

return true
