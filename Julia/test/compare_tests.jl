#= 
    Comparison tests of the kernel output (for each data size)
        against the baseline tests
=#

using Test

# baselines directory same as in SENA/C-s_w
inputfiles = readdir("../baselines", join=true)

outputfiles = readdir("test/test_output", join=true)

# Filter the files in baselines/ to only include non_interpolated_files
non_interpolated_input_files = filter(p->p[25] == '0', inputfiles)

for (input, output) in zip(non_interpolated_input_files, outputfiles)
    datasize = input[14:23]
    @testset "$datasize Comparison " begin

        @test success(`cmp --quiet $input $output`)
    end
end

return true

