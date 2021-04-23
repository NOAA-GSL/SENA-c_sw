#=

*                   GNU Lesser General Public License
*
* This file is part of the FV3 dynamical core.
*
* The FV3 dynamical core is free software: you can redistribute it
* and/or modify it under the terms of the
* GNU Lesser General Public License as published by the
* Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* The FV3 dynamical core is distributed in the hope that it will be
* useful, but WITHOUT ANYWARRANTY; without even the implied warranty
* of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
* See the GNU General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with the FV3 dynamical core.
* If not, see <http://www.gnu.org/licenses/>.

=#

module sw_core_mod

using Printf, NCDatasets, Revise, OffsetArrays, Dates, Parameters, DataStructures

export  print_state, write_state, c_sw!

#------------------------------------------------------------------
# print_state
#
# Prints statistics for the kernel state variables
#------------------------------------------------------------------
function print_state(message::String, data, io::IOStream)

    println(io, "TEST")
    println(io, "TEST ", "="^115)
    @printf(io, "%-5s %29s\n", "TEST ", message)
    println(io, "TEST ", "="^115)
    @printf(io, "%-5s %14s %19s %19s %19s %19s %19s\n","TEST", "Variable", "Min", "Max", "First", "Last", "RMS")
    println(io, "TEST ", "-"^115)

    # print the variables
    print_2d_variable(io, "rarea", data.rarea)
    print_2d_variable(io, "rarea_c", data.rarea_c)
    print_3d_variable(io, "sin_sg", data.sin_sg)
    print_3d_variable(io, "cos_sg", data.cos_sg)
    print_2d_variable(io, "sina_v", data.sina_v)
    print_2d_variable(io, "cosa_v", data.cosa_v)
    print_2d_variable(io, "sina_u", data.sina_u)
    print_2d_variable(io, "cosa_u", data.cosa_u)
    print_2d_variable(io, "fC", data.fC)
    print_2d_variable(io, "rdxc", data.rdxc)
    print_2d_variable(io, "rdyc", data.rdyc)
    print_2d_variable(io, "dx", data.dx)
    print_2d_variable(io, "dy", data.dy)
    print_2d_variable(io, "dxc", data.dxc)
    print_2d_variable(io, "dyc", data.dyc)
    print_2d_variable(io, "cosa_s", data.cosa_s)
    print_2d_variable(io, "rsin_u", data.rsin_u)
    print_2d_variable(io, "rsin_v", data.rsin_v)
    print_2d_variable(io, "rsin2", data.rsin2)
    print_2d_variable(io, "dxa", data.dxa)
    print_2d_variable(io, "dya", data.dya)
  
    print_3d_variable(io,"delp", data.delp)
    print_3d_variable(io,"delpc", data.delpc)
    print_3d_variable(io,"pt", data.pt)
    print_3d_variable(io,"ptc", data.ptc)
    print_3d_variable(io,"u", data.u)
    print_3d_variable(io,"v", data.v)
    print_3d_variable(io,"w", data.w)
    print_3d_variable(io,"uc", data.uc)
    print_3d_variable(io,"vc", data.vc)
    print_3d_variable(io,"ua", data.ua)
    print_3d_variable(io,"va", data.va)
    print_3d_variable(io,"wc", data.wc)
    print_3d_variable(io,"ut", data.ut)
    print_3d_variable(io,"vt", data.vt)
    print_3d_variable(io,"divg_d", data.divg_d)

    println(io, "TEST ", "-"^115)
    println(io, "TEST")

    return nothing

end # function print_state

#------------------------------------------------------------------
# print_2d_variable
#
# Prints statistics for a 2d state variable
#------------------------------------------------------------------
function print_2d_variable(io::IOStream,name::String, data::OffsetArray)

    min =     minimum(data)
    max =     maximum(data)
    first =   data[begin, begin]
    last =    data[end, end]
    rms =     sqrt(sum(data.^2.) / length(data))

    @printf(io, "%-5s %14s %19.10E %19.10E %19.10E %19.10E %19.10E\n","TEST", name, min, max , first, last, rms)

    return nothing

end # function print_2d_variable

#------------------------------------------------------------------
# print_3d_variable
#
# Prints statistics for a 3d state variable
#------------------------------------------------------------------
function print_3d_variable(io::IOStream, name::String, data::OffsetArray)

    min =    minimum(data)
    max =    maximum(data)
    first =  data[begin, begin, begin]
    last =   data[end, end, end]
    rms =    sqrt(sum(data.^2.) / length(data))
  
    @printf(io, "%-5s %14s %19.10E %19.10E %19.10E %19.10E %19.10E\n","TEST", name, min, max , first, last, rms)

    return nothing

end # function print_3d_variable

#------------------------------------------------------------------
# write_state
#
# Write state to NetCDF file
#------------------------------------------------------------------
# create a NetCDF file from the data in netCDF-4 format
function write_state(outputfilename::String, data)

    date = Dates.now()
    dateString = Dates.format(date, "yyyy-mm-dd HH:MM:SS")

    ds = NCDataset(outputfilename, "c" ; format=:netcdf4, attrib=OrderedDict(
    "creation_date"             => dateString,
    "kernel_name"               => "c_sw",
    "isd"                       => data.isd,
    "ied"                       => data.ied,
    "jsd"                       => data.jsd,
    "jed"                       => data.jed,
    "is"                        => data.is,
    "ie"                        => data.ie,
    "js"                        => data.js,
    "je"                        => data.je,
    "nord"                      => data.nord,
    "npx"                       => data.npx,
    "npy"                       => data.npy,
    "npz"                       => data.npz,
    "dt2"                       => data.dt2,
    "sw_corner"                 => data.sw_corner,
    "se_corner"                 => data.se_corner,
    "nw_corner"                 => data.nw_corner,
    "ne_corner"                 => data.ne_corner,
  ))
  
  # define Dimensions
    ds.dim["ni"] =    data.ni
    ds.dim["nj"] =    data.nj
    ds.dim["nip1"] =  data.nip1
    ds.dim["njp1"] =  data.njp1
    ds.dim["nk"] =    data.nk
    ds.dim["nk9"] =   data.nk9
  
  # Declare variables
    ncrarea =           defVar(ds, "rarea", Float64, ("ni", "nj"))
    ncrarea_c =         defVar(ds, "rarea_c", Float64, ("nip1", "njp1"))
    ncsin_sg =          defVar(ds, "sin_sg", Float64, ("ni", "nj", "nk9"))
    nccos_sg =          defVar(ds, "cos_sg", Float64, ("ni", "nj", "nk9"))
    ncsina_v =          defVar(ds, "sina_v", Float64, ("ni", "njp1"))
    nccosa_v =          defVar(ds, "cosa_v", Float64, ("ni", "njp1"))
    ncsina_u =          defVar(ds, "sina_u", Float64, ("nip1", "nj"))
    nccosa_u =          defVar(ds, "cosa_u", Float64, ("nip1", "nj"))
    ncfC =              defVar(ds, "fC", Float64, ("nip1", "njp1"))
    ncrdxc =            defVar(ds, "rdxc", Float64, ("nip1", "nj"))
    ncrdyc =            defVar(ds, "rdyc", Float64, ("ni", "njp1"))
    ncdx =              defVar(ds, "dx", Float64, ("ni", "njp1"))
    ncdy =              defVar(ds, "dy", Float64, ("nip1", "nj"))
    ncdxc =             defVar(ds, "dxc", Float64, ("nip1", "nj"))
    ncdyc =             defVar(ds, "dyc", Float64, ("ni", "njp1"))
    nccosa_s =          defVar(ds, "cosa_s", Float64, ("ni", "nj"))
    ncrsin_u =          defVar(ds, "rsin_u", Float64, ("nip1", "nj"))
    ncrsin_v =          defVar(ds, "rsin_v", Float64, ("ni", "njp1"))
    ncrsin2 =           defVar(ds, "rsin2", Float64, ("ni", "nj"))
    ncdxa =             defVar(ds, "dxa", Float64, ("ni", "nj"))
    ncdya =             defVar(ds, "dya", Float64, ("ni", "nj"))
    ncdelp =            defVar(ds, "delp", Float64, ("ni", "nj", "nk"))
    ncdelpc =           defVar(ds, "delpc", Float64, ("ni", "nj", "nk"))
    ncpt =              defVar(ds, "pt", Float64, ("ni", "nj", "nk"))
    ncptc =             defVar(ds, "ptc", Float64, ("ni", "nj", "nk"))
    ncu =               defVar(ds, "u", Float64, ("ni", "njp1", "nk"))
    ncv =               defVar(ds, "v", Float64, ("nip1", "nj", "nk"))
    ncw =               defVar(ds, "w", Float64, ("ni", "nj", "nk"))
    ncuc =              defVar(ds, "uc", Float64, ("nip1", "nj", "nk"))
    ncvc =              defVar(ds, "vc", Float64, ("ni", "njp1", "nk"))
    ncua =              defVar(ds, "ua", Float64, ("ni", "nj", "nk"))
    ncva =              defVar(ds, "va", Float64, ("ni", "nj", "nk"))
    ncwc =              defVar(ds, "wc", Float64, ("ni", "nj", "nk"))
    ncut =              defVar(ds, "ut", Float64, ("ni", "nj", "nk"))
    ncvt =              defVar(ds, "vt", Float64, ("ni", "nj", "nk"))
    ncdivg_d =          defVar(ds, "divg_d", Float64, ("nip1", "njp1", "nk"))
  
  # Define variables
    ncrarea[:] =   data.rarea
    ncrarea_c[:] = data.rarea_c
    ncsin_sg[:] =  data.sin_sg
    nccos_sg[:] =  data.cos_sg
    ncsina_v[:] =  data.sina_v
    nccosa_v[:] =  data.cosa_v
    ncsina_u[:] =  data.sina_u
    nccosa_u[:] =  data.cosa_u
    ncfC[:] =      data.fC
    ncrdxc[:] =    data.rdxc
    ncrdyc[:] =    data.rdyc
    ncdx[:] =      data.dx
    ncdy[:] =      data.dy
    ncdxc[:] =     data.dxc
    ncdyc[:] =     data.dyc
    nccosa_s[:] =  data.cosa_s
    ncrsin_u[:] =  data.rsin_u
    ncrsin_v[:] =  data.rsin_v
    ncrsin2[:] =   data.rsin2
    ncdxa[:] =     data.dxa
    ncdya[:] =     data.dya
    ncdelp[:] =    data.delp
    ncdelpc[:] =   data.delpc
    ncpt[:] =      data.pt
    ncptc[:] =     data.ptc
    ncu[:] =       data.u
    ncv[:] =       data.v
    ncw[:] =       data.w
    ncuc[:] =      data.uc
    ncvc[:] =      data.vc
    ncua[:] =      data.ua
    ncva[:] =      data.va
    ncwc[:] =      data.wc
    ncut[:] =      data.ut
    ncvt[:] =      data.vt
    ncdivg_d[:] =  data.divg_d
  
    close(ds)
  
end # function write_state

#------------------------------------------------------------------
# c_sw
#
# Top-level routine for c_sw kernel
#------------------------------------------------------------------
function c_sw!(data, k::Int64)
  
  # Unpack arguments 
  @unpack sw_corner, se_corner, nw_corner, ne_corner,
  rarea, rarea_c, sin_sg, cos_sg, sina_v, cosa_v,
  sina_u, cosa_u, fC, rdxc, rdyc, dx, dy, dxc, dyc,
  cosa_s, rsin_u, rsin_v, rsin2, dxa, dya,
  delpc, delp, ptc, pt, u, v, w, uc, vc, ua, va, wc, ut,
  vt, divg_d, dt2  = data

  # Unpack local variables
  @unpack isd, ied, jsd, jed, is, ie, js, je, 
          nord, npx, npy, npz, dt2, ni, nj, nip1, njp1, nk, nk9  = data

  # re-dimension variables to the first two dimensions 
  delpc =        @view delpc[:,:,k]
  delp =         @view delp[:,:,k]
  ptc =          @view ptc[:,:,k]
  pt =           @view pt[:,:,k]
  u =            @view u[:,:,k]
  v =            @view v[:,:,k]
  w =            @view w[:,:,k]
  uc =           @view uc[:,:,k]
  vc =           @view vc[:,:,k]
  ua =           @view ua[:,:,k]
  va =           @view va[:,:,k]
  wc =           @view wc[:,:,k]
  ut =           @view ut[:,:,k]
  vt =           @view vt[:,:,k]
  divg_d =       @view divg_d[:,:,k]

  #helper function to index dimensions
  indexdim(a, b) = b - a + 1

  # define local variables arrays and index them 
  vort = OffsetArray(Array{Float64, 2}(undef, indexdim(is-1,ie+1) , indexdim(js-1, je+1)), is-1:ie+1, js-1:je+1)
  ke   = OffsetArray(Array{Float64, 2}(undef, indexdim(is-1,ie+1) , indexdim(js-1, je+1)), is-1:ie+1, js-1:je+1)
  fx   = OffsetArray(Array{Float64, 2}(undef, indexdim(is-1,ie+2) , indexdim(js-1, je+1)), is-1:ie+2, js-1:je+1)
  fx1  = OffsetArray(Array{Float64, 2}(undef, indexdim(is-1,ie+2) , indexdim(js-1, je+1)), is-1:ie+2, js-1:je+1)
  fx2  = OffsetArray(Array{Float64, 2}(undef, indexdim(is-1,ie+2) , indexdim(js-1, je+1)), is-1:ie+2, js-1:je+1)
  fy   = OffsetArray(Array{Float64, 2}(undef, indexdim(is-1,ie+1) , indexdim(js-1, je+2)), is-1:ie+1, js-1:je+2)
  fy1  = OffsetArray(Array{Float64, 2}(undef, indexdim(is-1,ie+1) , indexdim(js-1, je+2)), is-1:ie+1, js-1:je+2)
  fy2  = OffsetArray(Array{Float64, 2}(undef, indexdim(is-1,ie+1) , indexdim(js-1, je+2)), is-1:ie+1, js-1:je+2)

  # Convert corners from Int32 to Bool for Julia conditional statements
  sw_corner = !iszero(sw_corner)
  se_corner = !iszero(se_corner)
  ne_corner = !iszero(ne_corner)
  nw_corner = !iszero(nw_corner)

  iep1 = ie + 1
  jep1 = je + 1

  d2a2c_vect!(data, k)

  if nord > 0 
      divergence_corner!(data, k)
  end

  for j = js-1:jep1
      for i = is-1:iep1+1
          if ut[i, j] > 0.0
              ut[i, j] = dt2 * ut[i, j] * dy[i, j] * sin_sg[i-1, j, 3]
          else
              ut[i, j] = dt2 * ut[i, j] * dy[i, j] * sin_sg[i, j, 1]
          end
      end
  end
  for j = js-1:je+2
      for i = is-1:iep1
          if vt[i, j] > 0.0
              vt[i, j] = dt2 * vt[i, j] * dx[i, j] * sin_sg[i, j-1, 4]
          else
              vt[i, j] = dt2 * vt[i, j] * dx[i, j] * sin_sg[i, j, 2]
          end
      end
  end

  # -----------------
  #  Transport delp:
  # -----------------
  # Xdir:
  fill2_4corners!(delp, pt, 1, sw_corner, se_corner, ne_corner, nw_corner, npx, npy)
  fill_4corners!(w, 1, sw_corner, se_corner, ne_corner, nw_corner, npx, npy)

  for j = js-1:je+1
      for i = is-1:ie+2
          if ut[i, j] > 0.
              fx1[i, j] = delp[i-1, j]
              fx[i, j]  = pt[i-1, j]
              fx2[i, j] = w[i-1, j]
          else
              fx1[i, j] = delp[i, j]
              fx[i, j]  = pt[i, j]
              fx2[i, j] = w[i, j]
          end
          fx1[i, j] = ut[i, j]  * fx1[i, j]
          fx[i, j]  = fx1[i, j] * fx[i, j]
          fx2[i, j] = fx1[i, j] * fx2[i, j]
      end
  end

  # # Ydir:
  fill2_4corners!(delp, pt, 2, sw_corner, se_corner, ne_corner, nw_corner, npx, npy)
  fill_4corners!(w, 2, sw_corner, se_corner, ne_corner, nw_corner, npx, npy)

  for j = js-1:je+2
      for i = is-1:ie+1
          if vt[i, j] > 0.
              fy1[i, j] = delp[i, j-1]
              fy[i, j]  = pt[i, j-1]
              fy2[i, j] = w[i, j-1]
          else
              fy1[i, j] = delp[i, j]
              fy[i, j]  = pt[i, j]
              fy2[i, j] = w[i, j]
          end
          fy1[i, j] = vt[i, j]  * fy1[i, j]
          fy[i, j]  = fy1[i, j] * fy[i, j]
          fy2[i, j] = fy1[i, j] * fy2[i, j]
      end
  end
  for j = js-1:je+1
      for i = is-1:ie+1
          delpc[i, j] = delp[i, j] + (fx1[i, j] - fx1[i+1, j] + fy1[i, j] - fy1[i, j+1]) *
              rarea[i, j]
          ptc[i, j] = (pt[i, j] * delp[i, j] + (fx[i, j] - fx[i+1, j] + 
                       fy[i, j] - fy[i, j+1]) * rarea[i, j]) / delpc[i, j]
          wc[i, j] = (w[i, j] * delp[i, j] + (fx2[i, j] - fx2[i+1, j] +
                       fy2[i, j] - fy2[i, j+1]) * rarea[i, j]) / delpc[i, j]
      end
  end

  # ---------------
  #  Compute KE:
  # ---------------

  #  Since uc = u*, i.e. the covariant wind perpendicular to the face edge, if
  #  we want to compute kinetic energy we will need the true coordinate-parallel
  #  covariant wind, computed through u = uc*sina + v*cosa.
  # 
  #  Use the alpha for the cell KE is being computed in.
  for j = js-1:jep1
      for i = is-1:iep1
          if ua[i, j] > 0.
              if i == 1
                  ke[1, j] = uc[1, j] * sin_sg[1, j, 1] + v[1, j] * cos_sg[1, j, 1]
              elseif i == npx
                  ke[i, j] = uc[npx, j] * sin_sg[npx, j, 1] + v[npx, j] * cos_sg[npx, j, 1]
              else
                  ke[i, j] = uc[i, j]
              end
          else
              if i == 0
                  ke[0, j] = uc[1, j] * sin_sg[0, j, 3] + v[1, j] * cos_sg[0, j, 3]
              elseif i == npx - 1
                  ke[i, j] = uc[npx, j] * sin_sg[npx-1, j, 3] + v[npx, j] * cos_sg[npx-1, j, 3]
              else
                  ke[i, j] = uc[i+1, j]
              end
          end
      end
  end
  for j = js-1:jep1
      for i = is-1:iep1
          if va[i, j] > 0.
              if j == 1
                  vort[i, 1] = vc[i, 1] * sin_sg[i, 1, 2] + u[i, 1] * cos_sg[i, 1, 2]
              elseif j == npy
                  vort[i, j] = vc[i, npy] * sin_sg[i, npy, 2] + u[i, npy] * cos_sg[i, npy, 2]
              else
                  vort[i, j] = vc[i, j]
              end
          else
              if j == 0
                  vort[i, 0] = vc[i, 1] * sin_sg[i, 0, 4] + u[i, 1] * cos_sg[i, 0, 4]
              elseif j == npy - 1
                  vort[i, j] = vc[i, npy] * sin_sg[i, npy-1, 4] + u[i, npy] * cos_sg[i, npy-1, 4]
              else
                  vort[i, j] = vc[i, j+1]
              end
          end
      end
  end

  dt4 = 0.5 * dt2
  for j = js-1:jep1
      for i = is-1:iep1
          ke[i, j] = dt4 * (ua[i, j] * ke[i, j] + va[i, j] * vort[i, j])
      end
  end

  # -------------------------------
  # Compute circulation on C grid
  # -------------------------------
  # To consider using true co - variant winds at face edges?
  for j = js-1:je+1
      for i = is:ie+1
          fx[i, j] = uc[i, j] * dxc[i, j]
      end
  end

  for j = js:je+1
      for i = is-1:ie+1
          fy[i, j] = vc[i, j] * dyc[i, j]
      end
  end

  for j = js:je+1
      for i = is:ie+1
          vort[i, j] = fx[i, j-1] - fx[i, j] - fy[i-1, j] + fy[i, j]
      end
  end

  # Remove the extra term at the corners:
  if sw_corner
      vort[1, 1] = vort[1, 1] + fy[0, 1]
  end
  if se_corner
      vort[npx, 1] = vort[npx, 1] - fy[npx, 1]
  end
  if ne_corner
      vort[npx, npy] = vort[npx, npy] - fy[npx, npy]
  end
  if nw_corner
      vort[1, npy] = vort[1, npy] + fy[0, npy]
  end

  # ------------------------------------------------------------------
  # Compute absolute vorticity
  # ------------------------------------------------------------------
  for j = js:je+1
      for i = is:ie+1
          vort[i, j] = fC[i, j] + rarea_c[i, j] * vort[i, j]
      end
  end
  # ------------------------------
  # Transport absolute vorticity:
  # ------------------------------
  # To go from v to contravariant v at the edges, we divide by sin_sg;
  # but we  must multiply by sin_sg to get the proper flux.
  # These cancel, leaving us with fy1 = dt2 * v at the edges.
  # [For the same reason we only divide by sin instead of sin^2 in the interior]
  for j = js:je
      #DEC$ VECTOR ALWAYS
      for i = is:iep1
          if i == 1 || i == npx
              fy1[i, j] = dt2 * v[i, j]
          else
              fy1[i, j] = dt2 * (v[i, j] - uc[i, j] * cosa_u[i, j]) / sina_u[i, j]
          end
          if fy1[i, j] > 0.0
              fy[i, j] = vort[i, j]
          else
              fy[i, j] = vort[i, j+1]
          end
      end
  end
  for j = js:jep1
      if j == 1 || j == npy
          #DEC$ VECTOR ALWAYS
          for i = is:ie
              fx1[i, j] = dt2 * u[i, j]
              if fx1[i, j] > 0.0
                  fx[i, j] = vort[i, j]
              else
                  fx[i, j] = vort[i+1, j]
              end
          end
      else
          #DEC$ VECTOR ALWAYS
          for i = is:ie
              fx1[i, j] = dt2 * (u[i, j] - vc[i, j] * cosa_v[i, j]) / sina_v[i, j]
              if fx1[i, j] > 0.0
                  fx[i, j] = vort[i, j]
              else
                  fx[i, j] = vort[i+1, j]
              end
          end
      end
  end

  # Update time - centered winds on the C - Grid
  for j = js:je
      for i = is:iep1
          uc[i, j] = uc[i, j] + fy1[i, j] * fy[i, j] + rdxc[i, j] * (ke[i-1, j] - ke[i, j])
      end
  end
  for j = js:jep1
      for i = is:ie
          vc[i, j] = vc[i, j] - fx1[i, j] * fx[i, j] + rdyc[i, j] * (ke[i, j-1] - ke[i, j])
      end
  end

end # function c_sw

# ------------------------------------------------------------------
#  divergence_corner
# ------------------------------------------------------------------
function divergence_corner!(data, k::Int64)
  
  # unpack arguments
  @unpack sw_corner, se_corner, ne_corner, nw_corner,
  rarea_c, sin_sg, cos_sg, dxc, dyc, u, v, ua, va, divg_d,
  is, ie, js, je, npx, npy = data
  
  # re-dimension variables to the first two dimensions 
  u =         @view u[:,:,k]
  v =         @view v[:,:,k]
  ua =        @view ua[:,:,k]
  va =        @view va[:,:,k]
  divg_d =    @view divg_d[:,:,k]
  
  # Convert corners from Int32 to Bool for Julia conditional statements
  sw_corner = !iszero(sw_corner)
  se_corner = !iszero(se_corner)
  ne_corner = !iszero(ne_corner)
  nw_corner = !iszero(nw_corner)
  
  indexdim(a, b) = b - a + 1

  # local variables
  uf = OffsetArray(Array{Float64, 2}(undef, indexdim(is-2,ie+2) , indexdim(js-1, je+2)), is-2:ie+2, js-1:je+2)
  vf = OffsetArray(Array{Float64, 2}(undef, indexdim(is-1,ie+2) , indexdim(js-2, je+2)), is-1:ie+2, js-2:je+2)
  
  is2 = max(2, is)
  ie1 = min(npx - 1, ie + 1)
  
  #    9---4---8
  #    |       |
  #    1   5   3
  #    |       |
  #    6---2---7
  for j = js:je+1
    if j == 1 || j == npy
      for i = is-1:ie+1
        uf[i, j] = u[i, j] * dyc[i, j] * 0.5 * (sin_sg[i, j-1, 4] + sin_sg[i, j, 2])
      end
    else
      for i = is-1:ie+1
        uf[i, j] = (u[i, j] - 0.25 * (va[i, j-1] + va[i, j]) *
        (cos_sg[i, j-1, 4] + cos_sg[i, j, 2])) *
        dyc[i, j] * 0.5 * (sin_sg[i, j-1, 4] + sin_sg[i, j, 2])
      end
    end
  end
  
  for j = js-1:je+1
    for i = is2:ie1
      vf[i, j] = (v[i, j] - 0.25 * (ua[i-1, j] + ua[i, j]) *
      (cos_sg[i-1, j, 3] + cos_sg[i, j, 1])) *
      dxc[i, j] * 0.5 * (sin_sg[i-1, j, 3] + sin_sg[i, j, 1])
    end
    if is == 1
      vf[1, j] = v[1, j] * dxc[1, j] * 0.5 * (sin_sg[0, j, 3] + sin_sg[1, j, 1])
    end
    if ie + 1 == npx
      vf[npx, j] = v[npx, j] * dxc[npx, j] * 0.5 * (sin_sg[npx-1, j, 3] + sin_sg[npx, j, 1])
    end
  end
  
  for j = js:je+1
    for i = is:ie+1
      divg_d[i, j] = vf[i, j-1] - vf[i, j] + uf[i-1, j] - uf[i, j]
    end
  end
  
  # Remove the extra term at the corners:
  if sw_corner
    divg_d[1, 1] = divg_d[1, 1] - vf[1, 0]
  end
  if se_corner
    divg_d[npx, 1] = divg_d[npx, 1] - vf[npx, 0]
  end
  if ne_corner
    divg_d[npx, npy] = divg_d[npx, npy] + vf[npx, npy]
  end
  if nw_corner
    divg_d[1, npy] = divg_d[1, npy] + vf[1, npy]
  end
  
  for j = js:je+1
    for i = is:ie+1
      divg_d[i, j] = rarea_c[i, j] * divg_d[i, j]
    end
  end
  return nothing
end # function divergence_corner

# ------------------------------------------------------------------
# d2a2c_vect
#
#  There is a limit to how far this routine can fill uc and vc in the
#  halo, and so either mpp_update_domains or some sort of boundary
#  routine [extrapolation, outflow, interpolation from a nested grid]
#  is needed after c_sw is completed if these variables are needed
#  in the halo
# -------------------------------------------------------------------
function d2a2c_vect!(data, k::Int64)
  
  @unpack sw_corner, se_corner, ne_corner, nw_corner,
  sin_sg, cosa_u, cosa_v, cosa_s, rsin_u, rsin_v,
  rsin2, dxa, dya, u, v, ua, va, uc, vc, ut, vt,
  is, js, ie, je, isd, ied, jsd, jed, npx, npy = data
  
  # "slice the 3d arrays to 2d arrays"
  u =       @view u[:,:,k]
  v =       @view v[:,:,k]
  uc =      @view uc[:,:,k]
  vc =      @view vc[:,:,k]
  ua =      @view ua[:,:,k]
  va =      @view va[:,:,k]
  ut =      @view ut[:,:,k]
  vt =      @view vt[:,:,k]
  
  # Convert corners from Int32 to Bool for Julia conditional statements
  sw_corner = !iszero(sw_corner)
  se_corner = !iszero(se_corner)
  ne_corner = !iszero(ne_corner)
  nw_corner = !iszero(nw_corner)
  
  # 4-pt Lagrange interpolation
  a1 =  0.5625
  a2 = -0.0625
  # volume-conserving cubic with 2nd drv=0 at end point:
  c1 = -2 / 14
  c2 = 11 / 14
  c3 =  5 / 14

  indexdim(a, b) = b - a + 1
  
  # Local Variables
  utmp = OffsetArray(fill(1.e8, (indexdim(isd,ied),indexdim(jsd,jed))), isd:ied, jsd:jed)
  vtmp = OffsetArray(fill(1.e8, (indexdim(isd,ied),indexdim(jsd,jed))), isd:ied, jsd:jed)
  
  # -----------------------------
  # Interior:
  # -----------------------------
  for j = max(4, js - 1) : min(npy-4, je+1)
    for i = max(4, isd) : min(npx-4, ied)
      utmp[i, j] = a2 * (u[i, j-1] + u[i, j+2]) + a1 * (u[i, j] + u[i, j+1])
    end
  end
  for j = max(4, jsd) : min(npy-4, jed)
    for i = max(4, is - 1) : min(npx-4, ie+1)
      vtmp[i, j] = a2 * (v[i-1, j] + v[i+2, j]) + a1 * (v[i, j] + v[i+1, j])
    end
  end
  
  # -----------------------------
  # edges:
  # -----------------------------
  if js == 1 || jsd < 4
    for j = jsd:3
      for i = isd:ied
        utmp[i, j] = 0.5 * (u[i, j] + u[i, j+1])
        vtmp[i, j] = 0.5 * (v[i, j] + v[i+1, j])
      end
    end
  end
  
  if je + 1 == npy || jed >= npy - 4
    for j = npy-3:jed
      for i = isd:ied
        utmp[i, j] = 0.5 * (u[i, j] + u[i, j+1])
        vtmp[i, j] = 0.5 * (v[i, j] + v[i+1, j])
      end
    end
  end
  
  if is == 1 || isd < 4
    for j = max(4, jsd): min(npy-4, jed)
      for i = isd:3
        utmp[i, j] = 0.5 * (u[i, j] + u[i, j+1])
        vtmp[i, j] = 0.5 * (v[i, j] + v[i+1, j])
      end
    end
  end
  
  if ie + 1 == npx || ied >= npx - 4
    for j = max(4, jsd): min(npy-4, jed)
      for i = npx-3:ied
        utmp[i, j] = 0.5 * (u[i, j] + u[i, j+1])
        vtmp[i, j] = 0.5 * (v[i, j] + v[i+1, j])
      end
    end
  end
  
  for j = js-2:je+2
    for i = is-2:ie+2
      ua[i, j] = (utmp[i, j] - vtmp[i, j] * cosa_s[i, j]) * rsin2[i, j]
      va[i, j] = (vtmp[i, j] - utmp[i, j] * cosa_s[i, j]) * rsin2[i, j]
    end
  end
  
  # A - > C
  # -----------------------------
  # Fix the edges
  # -----------------------------
  # Xdir:
  if sw_corner
    for i = -2:0
      utmp[i, 0] = -vtmp[0, 1-i]
    end
  end
  if se_corner
    for i = 0:2
      utmp[npx+i, 0] = vtmp[npx, i+1]
    end
  end
  if ne_corner
    for i = 0:2
      utmp[npx+i, npy] = -vtmp[npx, je-i]
    end
  end
  if nw_corner
    for i = -2:0
      utmp[i, npy] = vtmp[0, je+i]
    end
  end
  
  ifirst = max(3,     is-1)
  ilast  = min(npx-2, ie+2)
  
  # ----------------------------------------------------------
  # 4th order interpolation for interior points:
  # -----------------------------------------------------------
  for j = js-1:je+1
    for i = ifirst:ilast
      uc[i, j] = a1 * (utmp[i-1, j] + utmp[i, j]) + a2 * (utmp[i-2, j] + utmp[i+1, j])
      ut[i, j] = (uc[i, j] - v[i, j] * cosa_u[i, j]) * rsin_u[i, j]
    end
  end
  
  # Xdir:
  if sw_corner
    ua[-1, 0] = -va[0, 2]
    ua[ 0, 0] = -va[0, 1]
  end
  if se_corner
    ua[npx,   0] = va[npx, 1]
    ua[npx+1, 0] = va[npx, 2]
  end
  if ne_corner
    ua[npx,   npy] = -va[npx, npy-1]
    ua[npx+1, npy] = -va[npx, npy-2]
  end
  if nw_corner
    ua[-1, npy] = va[0, npy-2]
    ua[ 0, npy] = va[0, npy-1]
  end
  
  if is == 1
    for j = js-1:je+1
      uc[0, j] = c1 * utmp[-2, j] + c2 * utmp[-1, j] + c3 * utmp[0, j]
      ut[1, j] = edge_interpolate4!(ua[-1:2, j], dxa[-1:2, j])
      #Want to  the UPSTREAM value
      if ut[1, j] < 0.0
        uc[1, j] = ut[1, j] * sin_sg[0, j, 3]
      else
        uc[1, j] = ut[1, j] * sin_sg[1, j, 1]
      end
      uc[2, j] = c1 * utmp[3, j] + c2 * utmp[2, j] + c3 * utmp[1, j]
      ut[0, j] = (uc[0, j] - v[0, j] * cosa_u[0, j]) * rsin_u[0, j]
      ut[2, j] = (uc[2, j] - v[2, j] * cosa_u[2, j]) * rsin_u[2, j]
    end
    
    if ie + 1 == npx
      for j = js-1:je+1
        uc[npx-1, j] =
        c1 * utmp[npx-3, j] + c2 * utmp[npx-2, j] + c3 * utmp[npx-1, j]
        i = npx
        ut[i, j] = 0.25 * (-ua[i-2, j] + 3.0 * (ua[i-1, j] + ua[i, j]) - ua[i+1, j])
        ut[i, j] = edge_interpolate4!(ua[i-2:i+1, j], dxa[i-2:i+1, j])
        if ut[i, j] < 0.0
          uc[i, j] = ut[i, j] * sin_sg[i-1, j, 3]
        else
          uc[i, j] = ut[i, j] * sin_sg[i, j, 1]
        end
        uc[npx+1, j] = c3 * utmp[npx, j] + c2 * utmp[npx+1, j] + c1 * utmp[npx+2, j]
        ut[npx-1, j] = (uc[npx-1, j] - v[npx-1, j] * cosa_u[npx-1, j]) * rsin_u[npx-1, j]
        ut[npx+1, j] = (uc[npx+1, j] - v[npx+1, j] * cosa_u[npx+1, j]) * rsin_u[npx+1, j]
      end
    end
    
  end
  
  # ---------
  # Ydir:
  # ---------
  if sw_corner
    for j = -2:0
      vtmp[0, j] = -utmp[1-j, 0]
    end
  end
  if nw_corner
    for j = 0:2
      vtmp[0, npy+j] = utmp[j+1, npy]
    end
  end
  if se_corner
    for j = -2:0
      vtmp[npx, j] = utmp[ie+j, 0]
    end
  end
  if ne_corner
    for j = 0:2
      vtmp[npx, npy+j] = -utmp[ie-j, npy]
    end
  end
  if sw_corner
    va[0, -1] = -ua[2, 0]
    va[0,  0] = -ua[1, 0]
  end
  if se_corner
    va[npx,  0] = ua[npx-1, 0]
    va[npx, -1] = ua[npx-2, 0]
  end
  if ne_corner
    va[npx,   npy] = -ua[npx-1, npy]
    va[npx, npy+1] = -ua[npx-2, npy]
  end
  if nw_corner
    va[0,   npy] = ua[1, npy]
    va[0, npy+1] = ua[2, npy]
  end
  
  for j = js-1:je+2
    if j == 1
      for i = is-1:ie+1
        vt[i, j] = edge_interpolate4!(va[i, -1:2], dya[i, -1:2])
        if vt[i, j] < 0.0
          vc[i, j] = vt[i, j] * sin_sg[i, j-1, 4]
        else
          vc[i, j] = vt[i, j] * sin_sg[i, j, 2]
        end
      end
    elseif j == 0 || j == npy - 1
      for i = is-1:ie+1
        vc[i, j] = c1 * vtmp[i, j-2] + c2 * vtmp[i, j-1] + c3 * vtmp[i, j]
        vt[i, j] = (vc[i, j] - u[i, j] * cosa_v[i, j]) * rsin_v[i, j]
      end
    elseif j == 2 || j == npy + 1
      for i = is-1:ie+1
        vc[i, j] = c1 * vtmp[i, j+1] + c2 * vtmp[i, j] + c3 * vtmp[i, j-1]
        vt[i, j] = (vc[i, j] - u[i, j] * cosa_v[i, j]) * rsin_v[i, j]
      end
    elseif j == npy
      for i = is-1:ie+1
        vt[i, j] = 0.25 * (-va[i, j-2] + 3.0 * (va[i, j-1] + va[i, j]) - va[i, j+1])
        vt[i, j] = edge_interpolate4!(va[i, j-2:j+1], dya[i, j-2:j+1])
        if vt[i, j] < 0.0
          vc[i, j] = vt[i, j] * sin_sg[i, j-1, 4]
        else
          vc[i, j] = vt[i, j] * sin_sg[i, j, 2]
        end
      end
    else
      # 4th order interpolation for interior points:
      for i = is-1:ie+1
        vc[i, j] = a2 * (vtmp[i, j-2] + vtmp[i, j+1]) + a1 * (vtmp[i, j-1] + vtmp[i, j])
        vt[i, j] = (vc[i, j] - u[i, j] * cosa_v[i, j]) * rsin_v[i, j]
      end
    end
  end
  return nothing
end # function d2a2c_vect

#------------------------------------------------------------------
# edge_interpolate4
#------------------------------------------------------------------
function edge_interpolate4!(ua, dxa)
  
  u0L = 0.5 * ((2.0 * dxa[2] + dxa[1]) * ua[2] - dxa[2] * ua[1]) / (dxa[1] + dxa[2])
  u0R = 0.5 * ((2.0 * dxa[3] + dxa[4]) * ua[3] - dxa[3] * ua[4]) / (dxa[3] + dxa[4])
  
  return u0L + u0R
  
  # This is the original edge - interpolation code, which makes
  # a relatively small increase in the error in unstretched case 2.
  #
  # edge_interpolate4 = 0.25 * [3 * [ua[2] + ua[3]] - [ua[1] + ua[4]]]
  
end #function edge_interpolate4

#----------------------------------------------------------------------------------
# fill2_4corners
#
#  This function fills the 4 corners of the scalar fields only as needed by c_core
#----------------------------------------------------------------------------------
function fill2_4corners!(q1, q2, dir, sw_corner, se_corner, ne_corner, nw_corner, npx, npy)

  if dir == 1
    if sw_corner
      q1[-1, 0] = q1[0, 2]
      q1[ 0, 0] = q1[0, 1]
      q2[-1, 0] = q2[0, 2]
      q2[ 0, 0] = q2[0, 1]
    end
    if se_corner
      q1[npx+1, 0] = q1[npx, 2]
      q1[npx,   0] = q1[npx, 1]
      q2[npx+1, 0] = q2[npx, 2]
      q2[npx,   0] = q2[npx, 1]
    end
    if nw_corner
      q1[ 0, npy] = q1[0, npy-1]
      q1[-1, npy] = q1[0, npy-2]
      q2[ 0, npy] = q2[0, npy-1]
      q2[-1, npy] = q2[0, npy-2]
    end
    if ne_corner
      q1[npx,   npy] = q1[npx, npy-1]
      q1[npx+1, npy] = q1[npx, npy-2]
      q2[npx,   npy] = q2[npx, npy-1]
      q2[npx+1, npy] = q2[npx, npy-2]
    end

  end
  
  if dir == 2
    if sw_corner 
      q1[0,  0] = q1[1, 0]
      q1[0, -1] = q1[2, 0]
      q2[0,  0] = q2[1, 0]
      q2[0, -1] = q2[2, 0]
    end
    if se_corner 
      q1[npx,  0] = q1[npx-1, 0]
      q1[npx, -1] = q1[npx-2, 0]
      q2[npx,  0] = q2[npx-1, 0]
      q2[npx, -1] = q2[npx-2, 0]
    end
    if nw_corner 
      q1[0, npy]   = q1[1, npy]
      q1[0, npy+1] = q1[2, npy]
      q2[0, npy]   = q2[1, npy]
      q2[0, npy+1] = q2[2, npy]
    end
    if ne_corner 
      q1[npx, npy]   = q1[npx-1, npy]
      q1[npx, npy+1] = q1[npx-2, npy]
      q2[npx, npy]   = q2[npx-1, npy]
      q2[npx, npy+1] = q2[npx-2, npy]
    end
    
  end

    return nothing
end # function fill2_4corners

#----------------------------------------------------------------------------------
# fill_4corners
#
#  This function fills the 4 corners of the scalar fields only as needed by c_core
#----------------------------------------------------------------------------------
function fill_4corners!(q, dir, sw_corner, se_corner, ne_corner, nw_corner, npx, npy)

  if dir == 1
    if sw_corner 
      q[-1, 0] = q[0, 2]
      q[ 0, 0] = q[0, 1]
    end
    if se_corner 
      q[npx+1, 0] = q[npx, 2]
      q[npx,   0] = q[npx, 1]
    end
    if nw_corner 
      q[ 0, npy] = q[0, npy-1]
      q[-1, npy] = q[0, npy-2]
    end
    if ne_corner 
      q[npx,   npy] = q[npx, npy-1]
      q[npx+1, npy] = q[npx, npy-2]
    end
  end
  if dir == 2
    if sw_corner
      q[0,  0] = q[1, 0]
      q[0, -1] = q[2, 0]
    end
    if se_corner 
      q[npx,  0] = q[npx-1, 0]
      q[npx, -1] = q[npx-2, 0]
    end
    if nw_corner 
      q[0, npy  ] = q[1, npy]
      q[0, npy+1] = q[2, npy]
    end
    if ne_corner 
      q[npx, npy  ] = q[npx-1, npy]
      q[npx, npy+1] = q[npx-2, npy]
    end
  end
  return nothing
end #function fill_4corners

end # module sw_core_mod
