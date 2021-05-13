
# Unit Tests of functions within src/SwCore.jl

using NCDatasets, Test

include("../src/SwCore.jl")
include("../src/Csw.jl")

using .sw_core_mod, .program

current_directory = @__DIR__

input_file_name = relpath("test/data/inputs/c_sw_12x24.nc", ".")

example_dataset = NCDataset(input_file_name)

state = State(example_dataset)

function subroutine_tests()
    @testset "c_sw subroutine tests" begin
    
        # @test c_sw!(state, 1)

        # @test divergence_corner!(state, 1)

        # @test d2a2c_vect!(state, 1)

        # @test edge_interpolate4!(state.ua, state.dxa)

        # @test fill2_4corners!(state.delp[:,:,1], state.pt[:,:,1], 1, true, true, true, true, state.npx, state.npy)

        # @test fill_4corners!(state.w[:,:,1], 1, true, true, true, true, state.npx, state.npy)


    end
end

function load_state_tests()

   # Test various variables, attributes and dimensions from .nc file are properly loaded into Julia Struct
    @test state.isd == -2

end

function write_and_print_tests() 
    @testset "write_state test"

        # create a new .nc file, write some data to it, test that the data was writen correctly, delete the file 

    @testset "print_state test"

        # create a new .test file, print some data to it, test that the data was writen correctly, delete the file 

end


@testset "All tests" begin
    subroutine_tests()
    # load_state_tests()
    # write_print_tests()
end
