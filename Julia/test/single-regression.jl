# Run a single regression test for a user speicfied dataset

dataset = string(ARGS[1])


include("../src/c_sw.jl")

using Test, TOML, .program

configfile = TOML.parsefile("./test/data/inputs/inputs.toml")

@testset "$dataset Regression " begin
        
    main(configfile["$dataset"])
    @test true

end

