#=
    Top level test suite, which runs unit tests, Pkg dependencies, regression & comparison tests.
=#

using Test

@testset "All tests" begin 
    
    # Integration test
    include("test_c_sw.jl")
    
    # Unit tests (in progress)
    # include("unit_tests.jl")

    # Test Pkg dependencies (not set up)
    # include("PkgTests.jl")

end
