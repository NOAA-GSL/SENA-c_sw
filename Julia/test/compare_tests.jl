#= 
    Comparison tests of the kernel output (for each data size)
        against the baseline tests
=#

using Test

inputfiles = readdir("../test/test_output", join=true)

outputfiles = readdir("test/test_output", join=true)

# Filter the files to only include non_interpolated_files
non_interpolated_input_files = filter(p->p[32] == '0', inputfiles)
non_interpolated_output_files = filter(p->p[29] == '0', outputfiles)

println(inputfiles)

for (input, output) in zip(non_interpolated_input_files, non_interpolated_output_files)

# Once interpolation is working, remove three lines above and use below 'for' loop
# for (input, output) in zip(inputfiles, outputfiles)
    datasize = input[21:32]
    @testset "$datasize Comparison " begin

        @test success(`cmp --quiet $input $output`)
    end
end

return true

