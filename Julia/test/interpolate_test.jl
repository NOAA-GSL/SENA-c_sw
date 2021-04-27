# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
# interpolateTest
#
# Unit test to test interpolation software.  The origArray is
# expanded by interpFactor to interpolateArray.  The origArray is
# filled with 1.0(s), interpolation should create a larger
# array, also filled with 1.0(s) if successful.
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
module interpolateTest

    include("../src/interpolate.jl")

    using .interpolate

    #  lower/upper bounds for 2D and 3D  arrays
    idims = Array{Int64}(undef, 6)
    odims = Array{Int64}(undef, 6)
    ϵ  = 1e-5

    # Helper function to declare the size of a particular dimension as opposed to subscripts
    indexdim(a, b) = b - a + 1

    #
    # Test the 2D interpolation.
    #

    for ii = 3: 7 # create these 2D arrays
      for interpfactor = 1: 3 # create these interpFactor(s)
        
        odims[1] = 1
        odims[2] = ii
        odims[3] = 1
        odims[4] = ii + 1 # non - sqare matrix
        
        # Allocate and populate the originalArray with 1.0
        origarray2d = ones(Float64, ii, ii +1)


        println(size(origarray2d))

        # Calculate dimensions for interpolated array
        interpolateCalculateSpace2D!(odims, interpfactor, idims)

        # Create the interpolateArray
        interparray2d = Array{Float64, 2}(undef, indexdim(idims[1],idims[2]), indexdim(idims[3],idims[4]))
        # populate the interparray with 2.0 (2.0 # = 1.0)
        interparray2d .= 2.0
        
        # Interpolate the interpolateArray.
        interpolateArray2D!(origarray2d, odims, interparray2d, idims, interpfactor)
        
        # The inperpolateArray should be all 1.0, + / - ϵ.
        # Print success or failure.  Exit with non - zero error code if falure.
        for j = idims[3]: idims[4]
          for k = idims[1]: idims[2]
            if abs(interparray2d[k, j] - 1.0) > ϵ
              println("Failure 2D")
              println( k, " , ", j , " , ", interparray2d[k, j], " , ",  interparray2d[k, j] - 1.0)
              break
            end
          end
        end

      end
    end
    println("success")

    #
    # Test the 3D interpolation. 
    #

    for kk = 2: 3
      for ii = 3: 7 # create these 2D arrays
        for interpfactor = 1: 5 # create these interpFactor(s)

          # Allocate and populate the origArray3D with ones
          origarray3d = ones(Float64, ii, ii +1, kk)

          odims[1] = 1
          odims[2] = ii
          odims[3] = 1
          odims[4] = ii + 1 # non - sqare matrix
          odims[5] = 1
          odims[6] = kk

          # Calculate dimensions for interpolated array
          interpolateCalculateSpace3D!(odims, interpfactor, idims)
          # Create the interpolateArray
          interparray3d = Array{Float64, 3}(undef, indexdim(idims[1],idims[2]), indexdim(idims[3],idims[4]), indexdim(idims[5],idims[6]))
          # populate the interparray with 2.0 (2.0 # = 1.0)
          interparray3d .= 2.0 
          
          # Interpolate the interpolateArray.
          interpolateArray3D!(origarray3d, odims, interparray3d, idims, interpfactor)

          # The inperpolateArray should be all 1.0, + / - ϵ.
          # Print success or failure.  Exit with non - zero error code if falure.
          for i = idims[5]: idims[6]
            for j = idims[3]: idims[4]
              for k = idims[1]: idims[2]
                if abs(interparray3d[k, j, i] - 1.0) > ϵ
                  println("failure 3d")
                  println( "k, j, i = ", k, " , ",j, " , ", i, " , ", interparray3d[k, j, i], " , ", interparray3d[k, j, i] - 1.0)
                  break
                end
              end
            end
          end

        end
      end
    end

    println("success")

    return true
 
end #module interpolateTest