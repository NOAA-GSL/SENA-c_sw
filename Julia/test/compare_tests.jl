#= 
    Comparison tests of the kernel output (for each data size)
        against the baseline tests
=#

using Test

inputfiles = readdir("../test/test_output", join=true)

outputfiles = readdir("test/test_output", join=true)

for (input, output) in zip(inputfiles, outputfiles)
    datasize = input[21:32]
    @testset "$datasize Comparison " begin

        @test success(`cmp --quiet $input $output`)
    end
end

return true

