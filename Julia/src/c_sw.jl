#=
      Program module containing the main() function which runs the c_sw kernel.
=#

module program

include("./sw_core.jl")
include("./states.jl")

using TOML, Printf
using NCDatasets, OffsetArrays
using .sw_core_mod


export main


    function main(input::Dict)

    # define vars
    datasize = input["name"]
    nc_input_file = input["input_file"]
    print_output_file = input["test_out_file"]
    nc_output_file = input["output_file"]

    # Open the export file with write/create/truncate privileges ("w")
    io = open(print_output_file,"w")

    # Define the dataset using NCDatasets pkg from each datafile 
    ds = NCDataset(nc_input_file, "a")

    # Assign Variables from the NetCDF file to a ::State Julia Struct
    current_state = State(ds)

    # Print input state 
    print_state("Input State - Original", current_state, io)

    # Call the Julia kernel
    # println("Num Threads : ", Threads.nthreads())
    println("Run the kernel :  $datasize  , Num Threads : ", Threads.nthreads())
    @time Threads.@threads for k = 1 : current_state.npz
        c_sw!(current_state, k)
    end

    # Print output state 
    print_state("Output State", current_state, io)

    # Write a new NetCDF file
    println("Write NetCDF file $datasize : ") 
    @time write_state(nc_output_file, current_state)

    # Close the IO file
    close(io)

    return true

    end # function main

end # module program
