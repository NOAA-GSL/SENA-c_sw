#=
    Top level program containing the function main which runs the c_sw kernel.
=#

module program

include("./SwCore.jl")
include("./states.jl")

using TOML, Printf
using NCDatasets, OffsetArrays
using .SwCoreModule


export main

    function main(input::Dict)

    # Input configuration variables
    datasize = input["name"]
    nc_input_file = input["input_file"]
    print_output_file = input["test_out_file"]
    nc_output_file = input["output_file"]
    interpFactor = input["interpFactor"]

    # Open the statistics log file
    io = open(print_output_file,"w")

    # Define the dataset using NCDatasets pkg for each datafile 
    ds = NCDataset(nc_input_file, "a")

    # Assign Variables from the NetCDF file to a ::State Julia Struct
    current_state = State(ds)

    # Print the input state statistics to the log file
    print_state("Input State - Original", current_state, io)

    # Interpolate the data according to the interpolation factor.
    if (interpFactor > 0) 
      interpolate_state!(current_state, interpFactor)
      # Write the input state statistics to the log
      print_state("Input State - Interpolated", current_state, io)
    elseif (interpFactor == 0) 
      # Do nothing.
    else
      println("Error, InterpFactor less than zero.")
    end

    # Run the kernel
    println("Run the kernel :  $datasize  , Num Threads : ", Threads.nthreads())
    @time Threads.@threads for k = 1 : current_state.npz
        c_sw!(current_state, k)
    end

    # Print the output state statistics to the log file
    print_state("Output State", current_state, io)

    # Write a new NetCDF file
    println("Write NetCDF file $datasize : ") 
    @time write_state(nc_output_file, current_state)

    # Close the statistics log file
    close(io)

    return true

    end # function main

end # module program
