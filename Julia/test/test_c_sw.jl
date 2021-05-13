#=
    Main test suite, which runs regression & comparison tests.
=#

using Test 

@testset verbose=true "All c_sw tests"  begin
    @testset verbose=true "Regression" begin
        @test include("regression_tests.jl")
    end
    @testset verbose=true "Comparison " begin
        @test include("compare_tests.jl")
    end
end

