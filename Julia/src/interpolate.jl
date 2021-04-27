module interpolate
    
export interpolateCalculateSpace2D!, interpolateCalculateSpace3D!, interpolateArray2D!, interpolateArray3D!
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ *
    #
    # interpolateCalculateSpace2D
    #
    #
    # Subroutine for calculating the lower and upper bounds of an
    # interpolated 2D array with regard to interpFactor.  The r is
    # expected to this routine, allocate the interpolated array,
    #  function: interpolateArrray2D to fill the
    # interpolated array.
    #
    # Subroutine parameters:
    # - originalArray, 2D array to be interpolated
    # - odims, 4 elements
    # - interpFactor, interpolation factor
    # - odims, 4 elements
    #
    # For example:
    # dimension interpArray(:, :)
    # interpolateCalculateSpace2D(odims, interpFactor, fdims)
    # l1 = fdims(1)
    # u1 = fdims(2)
    # l2 = fdims(3)
    # u2 = fdims(4)
    # allocate (interpArray(l1:u1, l2:u2))
    # interpolateArray2D(originalArray, odims, interpolatedArray, fdims,
    # interpFactor)
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ *

    function interpolateCalculateSpace2D!(odims, interpfactor, fdims)
     
      fdims[1] = odims[1]
      fdims[3] = odims[3]
      fdims[2] = ((odims[2] - odims[1] + 1) + (odims[2] - odims[1]) * interpfactor) + (odims[1] - 1)
      fdims[4] = ((odims[4] - odims[3] + 1) + (odims[4] - odims[3]) * interpfactor) + (odims[3] - 1)

    end # function interpolatecalculatespace2d

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ *
    #
    # interpolateCalculateSpace3D
    #
    #
    # Subroutine for calculating the lower and upper bounds of an
    # interpolated 3D array with regard to interpFactor.  The r is
    # expected to this routine, allocate the interpolated array,
    #  function: interpolateArray3D to fill the
    # interpolated array.
    #
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ *

    function interpolateCalculateSpace3D!(odims, interpfactor, fdims)
  
      fdims[1] = odims[1]
      fdims[3] = odims[3]
      fdims[5] = odims[5]
      fdims[2] = ((odims[2] - odims[1] + 1) + (odims[2] - odims[1]) * interpfactor) + (odims[1] - 1)
      fdims[4] = ((odims[4] - odims[3] + 1) + (odims[4] - odims[3]) * interpfactor) + (odims[3] - 1)
      fdims[6] = odims[6] # The 3rd dimension is unchanged.

    end # function interpolatecalculatespace3d

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ *
    #
    # interpolateArray2D
    #
    # Bilinear interpolation of f with respect to oa.
    #
    # f"s size has been calculated in function interpolateCalculateSpace2D
    # f has been allocated in the tree above this function
    #
    # Subroutine Parameters:
    # oa - original array, 2D
    # odims - low / high subscripts for oa
    # f - interpolated array, 2D
    # fdims - low / high subscripts for f
    # interpFactor - interpolation factor
    # Here are some illustrations of interpFactor.
    # We are assuming that the grid points are equally spaced.
    #
    # An interpolation factor of 1 means 1 new interpolated element.
    # So for example, a 3x3 matrix with interpFactor = 1 becomes a 5x5 matrix
    #
    # x"s are data from oa, the o"s are points to be interpolated
    #
    # [ x x x ]     == > [x o x o x]
    # [ x x x ]          [o o o o o]
    # [ x x x ]          [x o x o x]
    #                    [o o o o o]
    #                    [x o x o x]
    #
    # And an interpFactor = 2, means a 3x3 matrix becomes a 7x7 matrix.
    # [ x x x ]     == > [x o o x o o x]
    # [ x x x ]          [o o o o o o o]
    # [ x x x ]          [o o o o o o o]
    #                    [x o o x o o x]
    #                    [o o o o o o o]
    #                    [o o o o o o o]
    #                    [x o o x o o x]
    #
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ *

    function interpolateArray2D!(oa, odims, f, fdims, interpfactor::Int64)
 
      # Unlock the subscript bounds of the originalArray.
      ol1 = odims[1]
      ou1 = odims[2]
      ol2 = odims[3]
      ou2 = odims[4]

      # Unlock the subscript bounds of the interpolatedArray.
      fl1 = fdims[1]
      fu1 = fdims[2]
      fl2 = fdims[3]
      fu2 = fdims[4]

      # Intersperse the oa points into f.
      for j = ol2: ou2
        for i = ol1: ou1
          f[ol1 + (i - ol1) * (interpfactor + 1), ol2 + (j - ol2) * (interpfactor + 1)] = oa[i, j]
        end
      end

      # Loop over all the squares.
      for j = fl2: interpfactor + 1 : fu2 - 1
        for i = fl1: interpfactor + 1 : fu1 - 1

          # Find the indices of the corner points
          x1 = i
          x2 = i + interpfactor + 1
          y1 = j
          y2 = j + interpfactor + 1

          # Find the value of the corner points
          fq11 = f[x1, y1]
          fq12 = f[x1, y2]
          fq21 = f[x2, y1]
          fq22 = f[x2, y2]

          # Loop over all the interpolated points
          for iy = j: j + interpfactor + 1
            for ix = i: i + interpfactor + 1
              # Skip the corner points
              if ((ix == x1) && (iy == y1)) ||
                 ((ix == x1) && (iy == y2)) ||
                 ((ix == x2) && (iy == y1)) ||
                 ((ix == x2) && (iy == y2))
                continue
              end

              # Get the weights for the x direction
              wx1 = real(x2 - ix) / real(x2 - x1)
              wx2 = real(ix - x1) / real(x2 - x1)

              # Get the weights for the y direction
              wy1 = real(y2 - iy) / real(y2 - y1)
              wy2 = real(iy - y1) / real(y2 - y1)

              # interpolate
              f[ix, iy] = wy1 * (wx1 * fq11 + wx2 * fq21) + wy2 * (wx1 * fq12 + wx2 * fq22)

            end # x loop
          end   # y loop
        end     # i loop
      end       # j loop

    end # function interpolatearray2d

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ *
    #
    # interpolateArray3D
    #
    # Bilinear interpolation of f with respect to oa.  The interpolation
    # ignores the 3rd dimension.
    #
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ *

    function interpolateArray3D!(oa, odims, f, fdims, interpfactor::Int64)

      for i = odims[5]: odims[6]
        interpolateArray2D!(view(oa, :,:,i), odims, view(f, :,:,i), fdims, interpfactor)
      end

    end # function interpolatearray3d

  end # module interpolate
