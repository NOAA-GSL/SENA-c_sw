!***********************************************************************
!*                   GNU Lesser General Public License
!*
!* This file is part of the FV3 dynamical core.
!*
!* The FV3 dynamical core is free software: you can redistribute it
!* and/or modify it under the terms of the
!* GNU Lesser General Public License as published by the
!* Free Software Foundation, either version 3 of the License, or
!* (at your option) any later version.
!*
!* The FV3 dynamical core is distributed in the hope that it will be
!* useful, but WITHOUT ANYWARRANTY; without even the implied warranty
!* of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
!* See the GNU General Public License for more details.
!*
!* You should have received a copy of the GNU Lesser General Public
!* License along with the FV3 dynamical core.
!* If not, see <http://www.gnu.org/licenses/>.
!***********************************************************************

module sw_core_mod
 
  use netCDFModule
  use interpolate
#ifdef ENABLE_GPTL
  use gptl
#endif

  implicit none

  integer :: do_profile = 0  ! Flag for enabling profiling at runtime

  real, parameter:: big_number = 1.E8

  ! 4-pt Lagrange interpolation
  real, parameter:: a1 =  0.5625
  real, parameter:: a2 = -0.0625

  ! volume-conserving cubic with 2nd drv=0 at end point:
  real, parameter:: c1 = -2./14.
  real, parameter:: c2 = 11./14.
  real, parameter:: c3 =  5./14.

  ! Array domain indices
  integer :: isd
  integer :: ied
  integer :: jsd
  integer :: jed
  integer :: is
  integer :: ie
  integer :: js
  integer :: je
  integer :: nord
  integer :: npx
  integer :: npy
  integer :: npz

  ! State variables
  logical           :: sw_corner, se_corner, ne_corner, nw_corner
  real              :: dt2
  real, allocatable :: rarea(:,:), rarea_c(:,:) 
  real, allocatable :: sin_sg(:,:,:), cos_sg(:,:,:)
  real, allocatable :: sina_v(:,:), cosa_v(:,:)
  real, allocatable :: sina_u(:,:), cosa_u(:,:)
  real, allocatable :: fC(:,:)
  real, allocatable :: rdxc(:,:), rdyc(:,:)
  real, allocatable :: dx(:,:), dy(:,:)
  real, allocatable :: dxc(:,:), dyc(:,:)
  real, allocatable :: cosa_s(:,:)
  real, allocatable :: rsin_u(:,:), rsin_v(:,:)
  real, allocatable :: rsin2(:,:)
  real, allocatable :: dxa(:,:), dya(:,:)
  real, allocatable :: delpc(:,:,:), delp(:,:,:)
  real, allocatable :: ptc(:,:,:), pt(:,:,:)
  real, allocatable :: u(:,:,:), v(:,:,:), w(:,:,:)
  real, allocatable :: uc(:,:,:), vc(:,:,:), wc(:,:,:)
  real, allocatable :: ua(:,:,:), va(:,:,:)
  real, allocatable :: ut(:,:,:), vt(:,:,:)
  real, allocatable :: divg_d(:,:,:)

contains

  !------------------------------------------------------------------
  ! allocate_state
  !
  ! Allocate and initialize the state variables.
  !------------------------------------------------------------------
  subroutine allocate_state

    ! First deallocate state if allocated
    call deallocate_state()

    ! Allocate state arrays
    allocate(rarea  (isd:ied,   jsd:jed       ))
    allocate(rarea_c(isd:ied+1, jsd:jed+1     ))
    allocate(sin_sg (isd:ied,   jsd:jed,     9))
    allocate(cos_sg (isd:ied,   jsd:jed,     9))
    allocate(sina_v (isd:ied,   jsd:jed+1     ))
    allocate(cosa_v (isd:ied,   jsd:jed+1     ))
    allocate(sina_u (isd:ied+1, jsd:jed       ))
    allocate(cosa_u (isd:ied+1, jsd:jed       ))
    allocate(fC     (isd:ied+1, jsd:jed+1     ))
    allocate(rdxc   (isd:ied+1, jsd:jed       ))
    allocate(rdyc   (isd:ied,   jsd:jed+1     ))
    allocate(dx     (isd:ied,   jsd:jed+1     ))
    allocate(dy     (isd:ied+1, jsd:jed       ))
    allocate(dxc    (isd:ied+1, jsd:jed       ))
    allocate(dyc    (isd:ied,   jsd:jed+1     ))
    allocate(cosa_s (isd:ied,   jsd:jed       ))
    allocate(rsin_u (isd:ied+1, jsd:jed       ))
    allocate(rsin_v (isd:ied,   jsd:jed+1     ))
    allocate(rsin2  (isd:ied,   jsd:jed       ))
    allocate(dxa    (isd:ied,   jsd:jed       ))
    allocate(dya    (isd:ied,   jsd:jed       ))
    allocate(delpc  (isd:ied,   jsd:jed,   npz))
    allocate(delp   (isd:ied,   jsd:jed,   npz))
    allocate(ptc    (isd:ied,   jsd:jed,   npz))
    allocate(pt     (isd:ied,   jsd:jed,   npz))
    allocate(u      (isd:ied,   jsd:jed+1, npz))
    allocate(v      (isd:ied+1, jsd:jed,   npz))
    allocate(w      (isd:ied,   jsd:jed,   npz))
    allocate(uc     (isd:ied+1, jsd:jed,   npz))
    allocate(vc     (isd:ied,   jsd:jed+1, npz))
    allocate(ua     (isd:ied,   jsd:jed,   npz))
    allocate(va     (isd:ied,   jsd:jed,   npz))
    allocate(wc     (isd:ied,   jsd:jed,   npz))
    allocate(ut     (isd:ied,   jsd:jed,   npz))
    allocate(vt     (isd:ied,   jsd:jed,   npz))
    allocate(divg_d (isd:ied+1, jsd:jed+1, npz))

    ! Initialize state arrays
    rarea(:,:) = 0.0
    rarea_c(:,:) = 0.0
    sin_sg(:,:,:) = 0.0
    cos_sg(:,:,:) = 0.0
    sina_v(:,:) = 0.0
    cosa_v(:,:) = 0.0
    sina_u(:,:) = 0.0
    cosa_u(:,:) = 0.0
    fC(:,:) = 0.0
    rdxc(:,:) = 0.0
    rdyc(:,:) = 0.0
    dx(:,:) = 0.0
    dy(:,:) = 0.0
    dxc(:,:) = 0.0
    dyc(:,:) = 0.0
    cosa_s(:,:) = 0.0
    rsin_u(:,:) = 0.0
    rsin_v(:,:) = 0.0
    rsin2(:,:) = 0.0
    dxa(:,:) = 0.0
    dya(:,:) = 0.0
    delpc(:,:,:) = 0.0
    delp(:,:,:) = 0.0
    ptc(:,:,:) = 0.0
    pt(:,:,:) = 0.0
    u(:,:,:) = 0.0
    v(:,:,:) = 0.0
    w(:,:,:) = 0.0
    uc(:,:,:) = 0.0
    vc(:,:,:) = 0.0
    ua(:,:,:) = 0.0
    va(:,:,:) = 0.0
    wc(:,:,:) = 0.0
    ut(:,:,:) = 0.0
    vt(:,:,:) = 0.0
    divg_d(:,:,:) = 0.0

  end subroutine allocate_state


  !------------------------------------------------------------------
  ! deallocate_state
  !
  ! Deallocates the state
  !------------------------------------------------------------------
  subroutine deallocate_state()

    ! Deallocate state arrays
    if (allocated(rarea)) then
      deallocate(rarea)
    end if
    if (allocated(rarea_c)) then
      deallocate(rarea_c)
    end if
    if (allocated(sin_sg)) then
      deallocate(sin_sg)
    end if
    if (allocated(cos_sg)) then
      deallocate(cos_sg)
    end if
    if (allocated(sina_v)) then
      deallocate(sina_v)
    end if
    if (allocated(cosa_v)) then
      deallocate(cosa_v)
    end if
    if (allocated(sina_u)) then
      deallocate(sina_u)
    end if
    if (allocated(cosa_u)) then
      deallocate(cosa_u)
    end if
    if (allocated(fC)) then
      deallocate(fC)
    end if
    if (allocated(rdxc)) then
      deallocate(rdxc)
    end if
    if (allocated(rdyc)) then
      deallocate(rdyc)
    end if
    if (allocated(dx)) then
      deallocate(dx)
    end if
    if (allocated(dy)) then
      deallocate(dy)
    end if
    if (allocated(dxc)) then
      deallocate(dxc)
    end if
    if (allocated(dyc)) then
      deallocate(dyc)
    end if
    if (allocated(cosa_s)) then
      deallocate(cosa_s)
    end if
    if (allocated(rsin_u)) then
      deallocate(rsin_u)
    end if
    if (allocated(rsin_v)) then
      deallocate(rsin_v)
    end if
    if (allocated(rsin2)) then
      deallocate(rsin2)
    end if
    if (allocated(dxa)) then
      deallocate(dxa)
    end if
    if (allocated(dya)) then
      deallocate(dya)
    end if
    if (allocated(delpc)) then
      deallocate(delpc)
    end if
    if (allocated(delp)) then
      deallocate(delp)
    end if
    if (allocated(ptc)) then
      deallocate(ptc)
    end if
    if (allocated(pt)) then
      deallocate(pt)
    end if
    if (allocated(u)) then
      deallocate(u)
    end if
    if (allocated(v)) then
      deallocate(v)
    end if
    if (allocated(w)) then
      deallocate(w)
    end if
    if (allocated(uc)) then
      deallocate(uc)
    end if
    if (allocated(vc)) then
      deallocate(vc)
    end if
    if (allocated(ua)) then
      deallocate(ua)
    end if
    if (allocated(va)) then
      deallocate(va)
    end if
    if (allocated(wc)) then
      deallocate(wc)
    end if
    if (allocated(ut)) then
      deallocate(ut)
    end if
    if (allocated(vt)) then
      deallocate(vt)
    end if
    if (allocated(divg_d)) then
      deallocate(divg_d)
    end if

  end subroutine deallocate_state


  !------------------------------------------------------------------
  ! print_state
  !
  ! Prints statistics for the kernel state variables
  !------------------------------------------------------------------
  subroutine print_state(msg)

    character(len=*) :: msg

    write(*,'(A4)') "TEST"
    write(*,'(A5,A115)') "TEST ", repeat("=",115)
    write(*,'(A5,A20)') "TEST ", msg
    write(*,'(A5,A115)') "TEST ", repeat("=",115)
    write(*,'(A5,A15,5A20)') "TEST ", "Variable", "Min", "Max", "First", "Last", "RMS"
    write(*,'(A5,A115)') "TEST ", repeat("-",115)

    call print_2d_variable("rarea", rarea)
    call print_2d_variable("rarea_c", rarea_c)
    call print_3d_variable("sin_sg", sin_sg)
    call print_3d_variable("cos_sg", cos_sg)
    call print_2d_variable("sina_v", sina_v)
    call print_2d_variable("cosa_v", cosa_v)
    call print_2d_variable("sina_u", sina_u)
    call print_2d_variable("cosa_u", cosa_u)
    call print_2d_variable("fC", fC)
    call print_2d_variable("rdxc", rdxc)
    call print_2d_variable("rdyc", rdyc)
    call print_2d_variable("dx", dx)
    call print_2d_variable("dy", dy)
    call print_2d_variable("dxc", dxc)
    call print_2d_variable("dyc", dyc)
    call print_2d_variable("cosa_s", cosa_s)
    call print_2d_variable("rsin_u", rsin_u)
    call print_2d_variable("rsin_v", rsin_v)
    call print_2d_variable("rsin2", rsin2)
    call print_2d_variable("dxa", dxa)
    call print_2d_variable("dya", dya)

    call print_3d_variable("delp", delp)
    call print_3d_variable("delpc", delpc)
    call print_3d_variable("pt", pt)
    call print_3d_variable("ptc", ptc)
    call print_3d_variable("u", u)
    call print_3d_variable("v", v)
    call print_3d_variable("w", w)
    call print_3d_variable("uc", uc)
    call print_3d_variable("vc", vc)
    call print_3d_variable("ua", ua)
    call print_3d_variable("va", va)
    call print_3d_variable("wc", wc)
    call print_3d_variable("ut", ut)
    call print_3d_variable("vt", vt)
    call print_3d_variable("divg_d", divg_d)

    write(*,'(A5,A115)') "TEST ", repeat("-",115)
    write(*,'(A4)') "TEST"

  end subroutine print_state


  !------------------------------------------------------------------
  ! write_state_stats
  !
  ! Writes statistics for the kernel state variables to the given file unit
  !------------------------------------------------------------------
  subroutine write_state_stats(msg, test_file_unit)

    character(len= *),intent(IN) :: msg
    integer          ,intent(IN) :: test_file_unit

    write(test_file_unit,'(A4)') "TEST"
    write(test_file_unit,'(A5,A115)') "TEST ", repeat("=",115)
    write(test_file_unit,'(A5,A20)') "TEST ", msg
    write(test_file_unit,'(A5,A115)') "TEST ", repeat("=",115)
    write(test_file_unit,'(A5,A15,5A20)') "TEST ", "Variable", "Min", "Max", "First", "Last", "RMS"
    write(test_file_unit,'(A5,A115)') "TEST ", repeat("-",115)

    call write_2d_variable(test_file_unit,"rarea", rarea)
    call write_2d_variable(test_file_unit,"rarea_c", rarea_c)
    call write_3d_variable(test_file_unit,"sin_sg", sin_sg)
    call write_3d_variable(test_file_unit,"cos_sg", cos_sg)
    call write_2d_variable(test_file_unit,"sina_v", sina_v)
    call write_2d_variable(test_file_unit,"cosa_v", cosa_v)
    call write_2d_variable(test_file_unit,"sina_u", sina_u)
    call write_2d_variable(test_file_unit,"cosa_u", cosa_u)
    call write_2d_variable(test_file_unit,"fC", fC)
    call write_2d_variable(test_file_unit,"rdxc", rdxc)
    call write_2d_variable(test_file_unit,"rdyc", rdyc)
    call write_2d_variable(test_file_unit,"dx", dx)
    call write_2d_variable(test_file_unit,"dy", dy)
    call write_2d_variable(test_file_unit,"dxc", dxc)
    call write_2d_variable(test_file_unit,"dyc", dyc)
    call write_2d_variable(test_file_unit,"cosa_s", cosa_s)
    call write_2d_variable(test_file_unit,"rsin_u", rsin_u)
    call write_2d_variable(test_file_unit,"rsin_v", rsin_v)
    call write_2d_variable(test_file_unit,"rsin2", rsin2)
    call write_2d_variable(test_file_unit,"dxa", dxa)
    call write_2d_variable(test_file_unit,"dya", dya)

    call write_3d_variable(test_file_unit,"delp", delp)
    call write_3d_variable(test_file_unit,"delpc", delpc)
    call write_3d_variable(test_file_unit,"pt", pt)
    call write_3d_variable(test_file_unit,"ptc", ptc)
    call write_3d_variable(test_file_unit,"u", u)
    call write_3d_variable(test_file_unit,"v", v)
    call write_3d_variable(test_file_unit,"w", w)
    call write_3d_variable(test_file_unit,"uc", uc)
    call write_3d_variable(test_file_unit,"vc", vc)
    call write_3d_variable(test_file_unit,"ua", ua)
    call write_3d_variable(test_file_unit,"va", va)
    call write_3d_variable(test_file_unit,"wc", wc)
    call write_3d_variable(test_file_unit,"ut", ut)
    call write_3d_variable(test_file_unit,"vt", vt)
    call write_3d_variable(test_file_unit,"divg_d", divg_d)

    write(test_file_unit,'(A5,A115)') "TEST ", repeat("-",115)
    write(test_file_unit,'(A4)') "TEST"

  end subroutine write_state_stats


  !------------------------------------------------------------------
  ! print_3d_variable
  !
  ! Prints statistics for a 3d state variable
  !------------------------------------------------------------------
  subroutine print_3d_variable(name, data)

    character(len=*) :: name
    real             :: data(:,:,:)

    ! Note: Assumed shape array sections always start with index=1 for all dimensions
    !       So we don't have to know start/end indices here
    write(*,'(A5,A15,5ES20.10)') "TEST ", name, minval(data), maxval(data), data(1,1,1),  &
                            data(size(data,1), size(data,2), size(data,3)), &
                            sqrt(sum(data**2) / size(data))

  end subroutine print_3d_variable

  !------------------------------------------------------------------
  ! write_3d_variable
  !
  ! Writes statistics for a 3d state variable to the given file unit
  !------------------------------------------------------------------
  subroutine write_3d_variable(unit, name, data)

    integer         ,intent(IN) :: unit
    character(len=*),intent(IN) :: name
    real            ,intent(IN) :: data(:,:,:)

    ! Note: Assumed shape array sections always start with index=1 for all dimensions
    !       So we don't have to know start/end indices here
    write(unit,'(A5, A15,5ES20.10)') "TEST ", name, minval(data), maxval(data), data(1,1,1),  &
                                             data(size(data,1), size(data,2), size(data,3)), &
                                             sqrt(sum(data**2) / size(data))

  end subroutine write_3d_variable

  !------------------------------------------------------------------
  ! print_2d_variable
  !
  ! Prints statistics for a 2d state variable
  !------------------------------------------------------------------
  subroutine print_2d_variable(name, data)

    character(len=*) :: name
    real             :: data(:,:)

    ! Note: Assumed shape array sections always start with index=1 for all dimensions
    !       So we don't have to know start/end indices here
    write(*,'(A5, A15,5ES20.10)') "TEST ", name, minval(data), maxval(data), data(1,1), &
                            data(size(data,1), size(data,2)),            &
                            sqrt(sum(data**2) / size(data))

  end subroutine print_2d_variable


  !------------------------------------------------------------------
  ! write_2d_variable
  !
  ! Writes statistics for a 2d state variable to the given file unit
  !------------------------------------------------------------------
  subroutine write_2d_variable(unit, name, data)

    integer         ,intent(IN) :: unit
    character(len=*),intent(IN) :: name
    real            ,intent(IN) :: data(:,:)

    ! Note: Assumed shape array sections always start with index=1 for all dimensions
    !       So we don't have to know start/end indices here
    write(unit,'(A5, A15,5ES20.10)') "TEST ", name, minval(data), maxval(data), data(1,1), &
                                             data(size(data,1), size(data,2)),            &
                                             sqrt(sum(data**2) / size(data))

  end subroutine write_2d_variable

  !------------------------------------------------------------------
  ! read_state
  !
  ! Read state from NetCDF file
  !------------------------------------------------------------------
  subroutine read_state(filename)

    character(len=*), intent(in) :: filename

    ! netCDF variables
    integer :: ncFileID

    ! Local variables
    integer :: ni, nj, nip1, njp1, nk, nk9

    ! Open file for read only
    call open_file(filename, "r", ncFileID)

    ! Read global attributes
    call read_global_int(ncFileID, "isd", isd)
    call read_global_int(ncFileID, "ied", ied)
    call read_global_int(ncFileID, "jsd", jsd)
    call read_global_int(ncFileID, "jed", jed)
    call read_global_int(ncFileID, "is", is)
    call read_global_int(ncFileID, "ie", ie)
    call read_global_int(ncFileID, "js", js)
    call read_global_int(ncFileID, "je", je)
    call read_global_int(ncFileID, "nord", nord)
    call read_global_int(ncFileID, "npx", npx)
    call read_global_int(ncFileID, "npy", npy)
    call read_global_int(ncFileID, "npz", npz)
    call read_global_real(ncFileID, "dt2", dt2)
    call read_global_logical(ncFileID, "sw_corner", sw_corner)
    call read_global_logical(ncFileID, "se_corner", se_corner)
    call read_global_logical(ncFileID, "nw_corner", nw_corner)
    call read_global_logical(ncFileID, "ne_corner", ne_corner)

    ! Read the model dimensions
    call read_dimension(ncFileID, "ni", ni)
    call read_dimension(ncFileID, "nip1", nip1)
    call read_dimension(ncFileID, "nj", nj)
    call read_dimension(ncFileID, "njp1", njp1)
    call read_dimension(ncFileID, "nk", nk)
    call read_dimension(ncFileID, "nk9", nk9)

    ! Check to make sure state dimensions matches state indices
    if ((ni   /= (ied-isd+1)) .OR. (nj   /= (jed-jsd+1)) .OR. &
        (nip1 /= (ied-isd+2)) .OR. (njp1 /= (jed-jsd+2)) .OR. &
        (nk   /= npz)         .OR. (nk9  /= 9)) then
      write(*,*) "Dimensions and indices of input data are inconsistent"
      stop 1
    end if

    ! Allocate and initialize the state
    call allocate_state()

    ! Read the state variables
    call read_2d_real(ncFileID, "rarea", rarea)
    call read_2d_real(ncFileID, "rarea_c", rarea_c)
    call read_3d_real(ncFileID, "sin_sg", sin_sg)
    call read_3d_real(ncFileID, "cos_sg", cos_sg)
    call read_2d_real(ncFileID, "sina_v", sina_v)
    call read_2d_real(ncFileID, "cosa_v", cosa_v)
    call read_2d_real(ncFileID, "sina_u", sina_u)
    call read_2d_real(ncFileID, "cosa_u", cosa_u)
    call read_2d_real(ncFileID, "fC", fC)
    call read_2d_real(ncFileID, "rdxc", rdxc)
    call read_2d_real(ncFileID, "rdyc", rdyc)
    call read_2d_real(ncFileID, "dx", dx)
    call read_2d_real(ncFileID, "dy", dy)
    call read_2d_real(ncFileID, "dxc", dxc)
    call read_2d_real(ncFileID, "dyc", dyc)
    call read_2d_real(ncFileID, "cosa_s", cosa_s)
    call read_2d_real(ncFileID, "rsin_u", rsin_u)
    call read_2d_real(ncFileID, "rsin_v", rsin_v)
    call read_2d_real(ncFileID, "rsin2", rsin2)
    call read_2d_real(ncFileID, "dxa", dxa)
    call read_2d_real(ncFileID, "dya", dya)
    call read_3d_real(ncFileID, "delp", delp)
    call read_3d_real(ncFileID, "delpc", delpc)
    call read_3d_real(ncFileID, "pt", pt)
    call read_3d_real(ncFileID, "ptc", ptc)
    call read_3d_real(ncFileID, "u", u)
    call read_3d_real(ncFileID, "v", v)
    call read_3d_real(ncFileID, "w", w)
    call read_3d_real(ncFileID, "uc", uc)
    call read_3d_real(ncFileID, "vc", vc)
    call read_3d_real(ncFileID, "ua", ua)
    call read_3d_real(ncFileID, "va", va)
    call read_3d_real(ncFileID, "wc", wc)
    call read_3d_real(ncFileID, "ut", ut)
    call read_3d_real(ncFileID, "vt", vt)
    call read_3d_real(ncFileID, "divg_d", divg_d)

    ! Close the NetCDF file
    call close_file(ncFileID)

  end subroutine read_state


  !------------------------------------------------------------------
  ! write_state
  !
  ! Write state to NetCDF file
  !------------------------------------------------------------------
  subroutine write_state(filename)

    character(len=*), intent(in) :: filename

    ! General netCDF variables
    integer :: ncFileID
    integer :: niDimID, njDimID, nip1DimID, njp1DimID, nkDimID, nk9DimID
    integer :: rareaVarID, rarea_cVarID, sin_sgVarID, cos_sgVarID, sina_vVarID
    integer :: cosa_vVarID, sina_uVarID, cosa_uVarID, fCVarID, rdxcVarID
    integer :: rdycVarID, dxVarID, dyVarID, dxcVarID, dycVarID, cosa_sVarID
    integer :: rsin_uVarID, rsin_vVarID, rsin2VarID, dxaVarID, dyaVarID
    integer :: delpVarID, delpcVarID, ptVarID, ptcVarID, uVarID, vVarID, wVarID
    integer :: ucVarID, vcVarID, uaVarID, vaVarID, wcVarID, utVarID, vtVarID
    integer :: divg_dVarID

    ! Local variables
    character(len=8)      :: crdate  ! Needed by F90 DATE_AND_TIME intrinsic
    character(len=10)     :: crtime  ! Needed by F90 DATE_AND_TIME intrinsic
    character(len=5)      :: crzone  ! Needed by F90 DATE_AND_TIME intrinsic
    integer, dimension(8) :: values  ! Needed by F90 DATE_AND_TIME intrinsic
    character(len=19)     :: timestr ! String representation of clock

    ! Open new file, overwriting previous contents
    call open_file(filename, "w", ncFileID)

    ! Write Global Attributes
    call DATE_AND_TIME(crdate,crtime,crzone,values)
    write(timestr,'(i4,2(a,i2.2),1x,i2.2,2(a,i2.2))') &
          values(1), '/', values(2), '/', values(3), values(5), ':', &
          values(6), ':', values(7)
    call write_global_character(ncFileID, "creation_date", timestr)
    call write_global_character(ncFileID, "kernel_name", "c_sw")
    call write_global_int(ncFileID, "isd", isd)
    call write_global_int(ncFileID, "ied", ied)
    call write_global_int(ncFileID, "jsd", jsd)
    call write_global_int(ncFileID, "jed", jed)
    call write_global_int(ncFileID, "is", is)
    call write_global_int(ncFileID, "ie", ie)
    call write_global_int(ncFileID, "js", js)
    call write_global_int(ncFileID, "je", je)
    call write_global_int(ncFileID, "nord", nord)
    call write_global_int(ncFileID, "npx", npx)
    call write_global_int(ncFileID, "npy", npy)
    call write_global_int(ncFileID, "npz", npz)
    call write_global_real(ncFileID, "dt2", dt2)
    call write_global_logical(ncFileID, "sw_corner", sw_corner)
    call write_global_logical(ncFileID, "se_corner", se_corner)
    call write_global_logical(ncFileID, "nw_corner", nw_corner)
    call write_global_logical(ncFileID, "ne_corner", ne_corner)

    ! Define the i,j dimensions
    call define_dim(ncFileID, "ni", ied-isd+1, niDimID)
    call define_dim(ncFileID, "nj", jed-jsd+1, njDimID)

    ! Define the i+1, j+1 dimensions
    call define_dim(ncFileID, "nip1", ied-isd+2, nip1DimID)
    call define_dim(ncFileID, "njp1", jed-jsd+2, njp1DimID)

    ! Define the k dimension
    call define_dim(ncFileID, "nk", npz, nkDimID)
    call define_dim(ncFileID, "nk9", 9, nk9DimID)

    ! Define the fields
    call define_var_2d_real(ncFileID, "rarea",   niDimID,   njDimID,             rareaVarID)
    call define_var_2d_real(ncFileID, "rarea_c", nip1DimID, njp1DimID,           rarea_cVarID)
    call define_var_3d_real(ncFileID, "sin_sg",  niDimID,   njDimID,   nk9DimID, sin_sgVarID)
    call define_var_3d_real(ncFileID, "cos_sg",  niDimID,   njDimID,   nk9DimID, cos_sgVarID)
    call define_var_2d_real(ncFileID, "sina_v",  niDimID,   njp1DimID,           sina_vVarID)
    call define_var_2d_real(ncFileID, "cosa_v",  niDimID,   njp1DimID,           cosa_vVarID)
    call define_var_2d_real(ncFileID, "sina_u",  nip1DimID, njDimID,             sina_uVarID)
    call define_var_2d_real(ncFileID, "cosa_u",  nip1DimID, njDimID,             cosa_uVarID)
    call define_var_2d_real(ncFileID, "fC",      nip1DimID, njp1DimID,           fCVarID)
    call define_var_2d_real(ncFileID, "rdxc",    nip1DimID, njDimID,             rdxcVarID)
    call define_var_2d_real(ncFileID, "rdyc",    niDimID,   njp1DimID,           rdycVarID)
    call define_var_2d_real(ncFileID, "dx",      niDimID,   njp1DimID,           dxVarID)
    call define_var_2d_real(ncFileID, "dy",      nip1DimID, njDimID,             dyVarID)
    call define_var_2d_real(ncFileID, "dxc",     nip1DimID, njDimID,             dxcVarID)
    call define_var_2d_real(ncFileID, "dyc",     niDimID,   njp1DimID,           dycVarID)
    call define_var_2d_real(ncFileID, "cosa_s",  niDimID,   njDimID,             cosa_sVarID)
    call define_var_2d_real(ncFileID, "rsin_u",  nip1DimID, njDimID,             rsin_uVarID)
    call define_var_2d_real(ncFileID, "rsin_v",  niDimID,   njp1DimID,           rsin_vVarID)
    call define_var_2d_real(ncFileID, "rsin2",   niDimID,   njDimID,             rsin2VarID)
    call define_var_2d_real(ncFileID, "dxa",     niDimID,   njDimID,             dxaVarID)
    call define_var_2d_real(ncFileID, "dya",     niDimID,   njDimID,             dyaVarID)
    call define_var_3d_real(ncFileID, "delp",    niDimID,   njDimID,   nkDimID,  delpVarID)
    call define_var_3d_real(ncFileID, "delpc",   niDimID,   njDimID,   nkDimID,  delpcVarID)
    call define_var_3d_real(ncFileID, "pt",      niDimID,   njDimID,   nkDimID,  ptVarID)
    call define_var_3d_real(ncFileID, "ptc",     niDimID,   njDimID,   nkDimID,  ptcVarID)
    call define_var_3d_real(ncFileID, "u",       niDimID,   njp1DimID, nkDimID,  uVarID)
    call define_var_3d_real(ncFileID, "v",       nip1DimID, njDimID,   nkDimID,  vVarID)
    call define_var_3d_real(ncFileID, "w",       niDimID,   njDimID,   nkDimID,  wVarID)
    call define_var_3d_real(ncFileID, "uc",      nip1DimID, njDimID,   nkDimID,  ucVarID)
    call define_var_3d_real(ncFileID, "vc",      niDimID,   njp1DimID, nkDimID,  vcVarID)
    call define_var_3d_real(ncFileID, "ua",      niDimID,   njDimID,   nkDimID,  uaVarID)
    call define_var_3d_real(ncFileID, "va",      niDimID,   njDimID,   nkDimID,  vaVarID)
    call define_var_3d_real(ncFileID, "wc",      niDimID,   njDimID,   nkDimID,  wcVarID)
    call define_var_3d_real(ncFileID, "ut",      niDimID,   njDimID,   nkDimID,  utVarID)
    call define_var_3d_real(ncFileID, "vt",      niDimID,   njDimID,   nkDimID,  vtVarID)
    call define_var_3d_real(ncFileID, "divg_d",  nip1DimID, njp1DimID, nkDimID,  divg_dVarID)

    ! Leave define mode so we can fill
    call define_off(ncFileID)

    ! Fill the variables
    call write_var_2d_real(ncFileID, rareaVarID, rarea)
    call write_var_2d_real(ncFileID, rarea_cVarID, rarea_c)
    call write_var_3d_real(ncFileID, sin_sgVarID, sin_sg)
    call write_var_3d_real(ncFileID, cos_sgVarID, cos_sg)
    call write_var_2d_real(ncFileID, sina_vVarID, sina_v)
    call write_var_2d_real(ncFileID, cosa_vVarID, cosa_v)
    call write_var_2d_real(ncFileID, sina_uVarID, sina_u)
    call write_var_2d_real(ncFileID, cosa_uVarID, cosa_u)
    call write_var_2d_real(ncFileID, fCVarID, fC)
    call write_var_2d_real(ncFileID, rdxcVarID, rdxc)
    call write_var_2d_real(ncFileID, rdycVarID, rdyc)
    call write_var_2d_real(ncFileID, dxVarID, dx)
    call write_var_2d_real(ncFileID, dyVarID, dy)
    call write_var_2d_real(ncFileID, dxcVarID, dxc)
    call write_var_2d_real(ncFileID, dycVarID, dyc)
    call write_var_2d_real(ncFileID, cosa_sVarID, cosa_s)
    call write_var_2d_real(ncFileID, rsin_uVarID, rsin_u)
    call write_var_2d_real(ncFileID, rsin_vVarID, rsin_v)
    call write_var_2d_real(ncFileID, rsin2VarID, rsin2)
    call write_var_2d_real(ncFileID, dxaVarID, dxa)
    call write_var_2d_real(ncFileID, dyaVarID, dya)
    call write_var_3d_real(ncFileID, delpVarID, delp)
    call write_var_3d_real(ncFileID, delpcVarID, delpc)
    call write_var_3d_real(ncFileID, ptVarID, pt)
    call write_var_3d_real(ncFileID, ptcVarID, ptc)
    call write_var_3d_real(ncFileID, uVarID, u)
    call write_var_3d_real(ncFileID, vVarID, v)
    call write_var_3d_real(ncFileID, wVarID, w)
    call write_var_3d_real(ncFileID, ucVarID, uc)
    call write_var_3d_real(ncFileID, vcVarID, vc)
    call write_var_3d_real(ncFileID, uaVarID, ua)
    call write_var_3d_real(ncFileID, vaVarID, va)
    call write_var_3d_real(ncFileID, wcVarID, wc)
    call write_var_3d_real(ncFileID, utVarID, ut)
    call write_var_3d_real(ncFileID, vtVarID, vt)
    call write_var_3d_real(ncFileID, divg_dVarID, divg_d)

    ! Close the NetCDF file
    call close_file(ncFileID)

  end subroutine write_state


  !------------------------------------------------------------------
  ! write_subdomain
  !
  ! Write a subdomain of the state to NetCDF file
  !------------------------------------------------------------------
  subroutine write_subdomain(filename, i1, j1, ni, nj)

    character(len=*), intent(in) :: filename
    integer,          intent(in) :: i1, j1, ni, nj

    ! General netCDF variables
    integer :: ncFileID
    integer :: niDimID, njDimID, nip1DimID, njp1DimID, nkDimID, nk9DimID
    integer :: rareaVarID, rarea_cVarID, sin_sgVarID, cos_sgVarID, sina_vVarID
    integer :: cosa_vVarID, sina_uVarID, cosa_uVarID, fCVarID, rdxcVarID
    integer :: rdycVarID, dxVarID, dyVarID, dxcVarID, dycVarID, cosa_sVarID
    integer :: rsin_uVarID, rsin_vVarID, rsin2VarID, dxaVarID, dyaVarID
    integer :: delpVarID, delpcVarID, ptVarID, ptcVarID, uVarID, vVarID, wVarID
    integer :: ucVarID, vcVarID, uaVarID, vaVarID, wcVarID, utVarID, vtVarID
    integer :: divg_dVarID

    ! Local variables
    character(len=8)      :: crdate  ! Needed by F90 DATE_AND_TIME intrinsic
    character(len=10)     :: crtime  ! Needed by F90 DATE_AND_TIME intrinsic
    character(len=5)      :: crzone  ! Needed by F90 DATE_AND_TIME intrinsic
    integer, dimension(8) :: values  ! Needed by F90 DATE_AND_TIME intrinsic
    character(len=19)     :: timestr ! String representation of clock

    integer :: sub_isd, sub_ied, sub_jsd, sub_jed
    integer :: sub_is, sub_js, sub_ie, sub_je
    integer :: sub_npx, sub_npy

    ! Validate the subdomain indices
    if ((i1 < is) .OR. (i1 > ie) .OR. &
        (j1 < js) .OR. (j1 > je) .OR. &
        (i1 + ni - 1 > ie)      .OR.  &
        (j1 + nj - 1 > je)) then
      write(*,*) "Cannot write subdomain because it is out of bounds"
      stop 1
    end if

    ! Calculate indices for subdomain (halo size is 3)
    sub_is = i1
    sub_ie = i1 + ni - 1
    sub_js = j1
    sub_je = j1 + nj - 1
    sub_isd = sub_is - 3
    sub_ied = sub_ie + 3
    sub_jsd = sub_js - 3
    sub_jed = sub_je + 3
    sub_npx = sub_ie - sub_is + 2
    sub_npy = sub_je - sub_js + 2

    ! Open new file, overwriting previous contents
    call open_file(filename, "w", ncFileID)

    ! Write Global Attributes
    call DATE_AND_TIME(crdate,crtime,crzone,values)
    write(timestr,'(i4,2(a,i2.2),1x,i2.2,2(a,i2.2))') &
          values(1), '/', values(2), '/', values(3), values(5), ':', &
          values(6), ':', values(7)
    call write_global_character(ncFileID, "creation_date", timestr)
    call write_global_character(ncFileID, "kernel_name", "c_sw")
    call write_global_int(ncFileID, "isd", sub_isd)
    call write_global_int(ncFileID, "ied", sub_ied)
    call write_global_int(ncFileID, "jsd", sub_jsd)
    call write_global_int(ncFileID, "jed", sub_jed)
    call write_global_int(ncFileID, "is",  sub_is)
    call write_global_int(ncFileID, "ie",  sub_ie)
    call write_global_int(ncFileID, "js",  sub_js)
    call write_global_int(ncFileID, "je",  sub_je)
    call write_global_int(ncFileID, "nord", nord)
    call write_global_int(ncFileID, "npx", sub_npx)
    call write_global_int(ncFileID, "npy", sub_npy)
    call write_global_int(ncFileID, "npz", npz)
    call write_global_real(ncFileID, "dt2", dt2)
    if ((sub_is == 1) .AND. (sub_js == 1)) then
      call write_global_logical(ncFileID, "sw_corner", sw_corner)
    else
      call write_global_logical(ncFileID, "sw_corner", .not. sw_corner)
    end if
    if ((sub_npx == npx) .AND. (sub_js == 1)) then
      call write_global_logical(ncFileID, "se_corner", se_corner)
    else
      call write_global_logical(ncFileID, "se_corner", .not. se_corner)
    end if
    if ((sub_is == 1) .AND. (sub_npy == npy)) then
      call write_global_logical(ncFileID, "nw_corner", nw_corner)
    else
      call write_global_logical(ncFileID, "nw_corner", .not. nw_corner)
    end if
    if ((sub_npx == npx) .AND. (sub_npy == npy)) then
      call write_global_logical(ncFileID, "ne_corner", ne_corner)
    else
      call write_global_logical(ncFileID, "ne_corner", .not. ne_corner)
    end if

    ! Define the i,j dimensions
    call define_dim(ncFileID, "ni", sub_ied-sub_isd+1, niDimID)
    call define_dim(ncFileID, "nj", sub_jed-sub_jsd+1, njDimID)

    ! Define the i+1, j+1 dimensions
    call define_dim(ncFileID, "nip1", sub_ied-sub_isd+2, nip1DimID)
    call define_dim(ncFileID, "njp1", sub_jed-sub_jsd+2, njp1DimID)

    ! Define the k dimension
    call define_dim(ncFileID, "nk", npz, nkDimID)
    call define_dim(ncFileID, "nk9", 9, nk9DimID)

    ! Define the fields
    call define_var_2d_real(ncFileID, "rarea",   niDimID,   njDimID,             rareaVarID)
    call define_var_2d_real(ncFileID, "rarea_c", nip1DimID, njp1DimID,           rarea_cVarID)
    call define_var_3d_real(ncFileID, "sin_sg",  niDimID,   njDimID,   nk9DimID, sin_sgVarID)
    call define_var_3d_real(ncFileID, "cos_sg",  niDimID,   njDimID,   nk9DimID, cos_sgVarID)
    call define_var_2d_real(ncFileID, "sina_v",  niDimID,   njp1DimID,           sina_vVarID)
    call define_var_2d_real(ncFileID, "cosa_v",  niDimID,   njp1DimID,           cosa_vVarID)
    call define_var_2d_real(ncFileID, "sina_u",  nip1DimID, njDimID,             sina_uVarID)
    call define_var_2d_real(ncFileID, "cosa_u",  nip1DimID, njDimID,             cosa_uVarID)
    call define_var_2d_real(ncFileID, "fC",      nip1DimID, njp1DimID,           fCVarID)
    call define_var_2d_real(ncFileID, "rdxc",    nip1DimID, njDimID,             rdxcVarID)
    call define_var_2d_real(ncFileID, "rdyc",    niDimID,   njp1DimID,           rdycVarID)
    call define_var_2d_real(ncFileID, "dx",      niDimID,   njp1DimID,           dxVarID)
    call define_var_2d_real(ncFileID, "dy",      nip1DimID, njDimID,             dyVarID)
    call define_var_2d_real(ncFileID, "dxc",     nip1DimID, njDimID,             dxcVarID)
    call define_var_2d_real(ncFileID, "dyc",     niDimID,   njp1DimID,           dycVarID)
    call define_var_2d_real(ncFileID, "cosa_s",  niDimID,   njDimID,             cosa_sVarID)
    call define_var_2d_real(ncFileID, "rsin_u",  nip1DimID, njDimID,             rsin_uVarID)
    call define_var_2d_real(ncFileID, "rsin_v",  niDimID,   njp1DimID,           rsin_vVarID)
    call define_var_2d_real(ncFileID, "rsin2",   niDimID,   njDimID,             rsin2VarID)
    call define_var_2d_real(ncFileID, "dxa",     niDimID,   njDimID,             dxaVarID)
    call define_var_2d_real(ncFileID, "dya",     niDimID,   njDimID,             dyaVarID)
    call define_var_3d_real(ncFileID, "delp",    niDimID,   njDimID,   nkDimID,  delpVarID)
    call define_var_3d_real(ncFileID, "delpc",   niDimID,   njDimID,   nkDimID,  delpcVarID)
    call define_var_3d_real(ncFileID, "pt",      niDimID,   njDimID,   nkDimID,  ptVarID)
    call define_var_3d_real(ncFileID, "ptc",     niDimID,   njDimID,   nkDimID,  ptcVarID)
    call define_var_3d_real(ncFileID, "u",       niDimID,   njp1DimID, nkDimID,  uVarID)
    call define_var_3d_real(ncFileID, "v",       nip1DimID, njDimID,   nkDimID,  vVarID)
    call define_var_3d_real(ncFileID, "w",       niDimID,   njDimID,   nkDimID,  wVarID)
    call define_var_3d_real(ncFileID, "uc",      nip1DimID, njDimID,   nkDimID,  ucVarID)
    call define_var_3d_real(ncFileID, "vc",      niDimID,   njp1DimID, nkDimID,  vcVarID)
    call define_var_3d_real(ncFileID, "ua",      niDimID,   njDimID,   nkDimID,  uaVarID)
    call define_var_3d_real(ncFileID, "va",      niDimID,   njDimID,   nkDimID,  vaVarID)
    call define_var_3d_real(ncFileID, "wc",      niDimID,   njDimID,   nkDimID,  wcVarID)
    call define_var_3d_real(ncFileID, "ut",      niDimID,   njDimID,   nkDimID,  utVarID)
    call define_var_3d_real(ncFileID, "vt",      niDimID,   njDimID,   nkDimID,  vtVarID)
    call define_var_3d_real(ncFileID, "divg_d",  nip1DimID, njp1DimID, nkDimID,  divg_dVarID)

    ! Leave define mode so we can fill
    call define_off(ncFileID)

    ! Fill the variables
    call write_var_2d_real(ncFileID, rareaVarID,     rarea(sub_isd:sub_ied,   sub_jsd:sub_jed     ))
    call write_var_2d_real(ncFileID, rarea_cVarID, rarea_c(sub_isd:sub_ied+1, sub_jsd:sub_jed+1   ))
    call write_var_3d_real(ncFileID, sin_sgVarID,   sin_sg(sub_isd:sub_ied,   sub_jsd:sub_jed,   :))
    call write_var_3d_real(ncFileID, cos_sgVarID,   cos_sg(sub_isd:sub_ied,   sub_jsd:sub_jed,   :))
    call write_var_2d_real(ncFileID, sina_vVarID,   sina_v(sub_isd:sub_ied,   sub_jsd:sub_jed+1   ))
    call write_var_2d_real(ncFileID, cosa_vVarID,   cosa_v(sub_isd:sub_ied,   sub_jsd:sub_jed+1   ))
    call write_var_2d_real(ncFileID, sina_uVarID,   sina_u(sub_isd:sub_ied+1, sub_jsd:sub_jed     ))
    call write_var_2d_real(ncFileID, cosa_uVarID,   cosa_u(sub_isd:sub_ied+1, sub_jsd:sub_jed     ))
    call write_var_2d_real(ncFileID, fCVarID,           fC(sub_isd:sub_ied+1, sub_jsd:sub_jed+1   ))
    call write_var_2d_real(ncFileID, rdxcVarID,       rdxc(sub_isd:sub_ied+1, sub_jsd:sub_jed     ))
    call write_var_2d_real(ncFileID, rdycVarID,       rdyc(sub_isd:sub_ied,   sub_jsd:sub_jed+1   ))
    call write_var_2d_real(ncFileID, dxVarID,           dx(sub_isd:sub_ied,   sub_jsd:sub_jed+1   ))
    call write_var_2d_real(ncFileID, dyVarID,           dy(sub_isd:sub_ied+1, sub_jsd:sub_jed     ))
    call write_var_2d_real(ncFileID, dxcVarID,         dxc(sub_isd:sub_ied+1, sub_jsd:sub_jed     ))
    call write_var_2d_real(ncFileID, dycVarID,         dyc(sub_isd:sub_ied,   sub_jsd:sub_jed+1   ))
    call write_var_2d_real(ncFileID, cosa_sVarID,   cosa_s(sub_isd:sub_ied,   sub_jsd:sub_jed     ))
    call write_var_2d_real(ncFileID, rsin_uVarID,   rsin_u(sub_isd:sub_ied+1, sub_jsd:sub_jed     ))
    call write_var_2d_real(ncFileID, rsin_vVarID,   rsin_v(sub_isd:sub_ied,   sub_jsd:sub_jed+1   ))
    call write_var_2d_real(ncFileID, rsin2VarID,     rsin2(sub_isd:sub_ied,   sub_jsd:sub_jed     ))
    call write_var_2d_real(ncFileID, dxaVarID,         dxa(sub_isd:sub_ied,   sub_jsd:sub_jed     ))
    call write_var_2d_real(ncFileID, dyaVarID,         dya(sub_isd:sub_ied,   sub_jsd:sub_jed     ))
    call write_var_3d_real(ncFileID, delpVarID,       delp(sub_isd:sub_ied,   sub_jsd:sub_jed,   :))
    call write_var_3d_real(ncFileID, delpcVarID,     delpc(sub_isd:sub_ied,   sub_jsd:sub_jed,   :))
    call write_var_3d_real(ncFileID, ptVarID,           pt(sub_isd:sub_ied,   sub_jsd:sub_jed,   :))
    call write_var_3d_real(ncFileID, ptcVarID,         ptc(sub_isd:sub_ied,   sub_jsd:sub_jed,   :))
    call write_var_3d_real(ncFileID, uVarID,             u(sub_isd:sub_ied,   sub_jsd:sub_jed+1, :))
    call write_var_3d_real(ncFileID, vVarID,             v(sub_isd:sub_ied+1, sub_jsd:sub_jed,   :))
    call write_var_3d_real(ncFileID, wVarID,             w(sub_isd:sub_ied,   sub_jsd:sub_jed,   :))
    call write_var_3d_real(ncFileID, ucVarID,           uc(sub_isd:sub_ied+1, sub_jsd:sub_jed,   :))
    call write_var_3d_real(ncFileID, vcVarID,           vc(sub_isd:sub_ied,   sub_jsd:sub_jed+1, :))
    call write_var_3d_real(ncFileID, uaVarID,           ua(sub_isd:sub_ied,   sub_jsd:sub_jed,   :))
    call write_var_3d_real(ncFileID, vaVarID,           va(sub_isd:sub_ied,   sub_jsd:sub_jed,   :))
    call write_var_3d_real(ncFileID, wcVarID,           wc(sub_isd:sub_ied,   sub_jsd:sub_jed,   :))
    call write_var_3d_real(ncFileID, utVarID,           ut(sub_isd:sub_ied,   sub_jsd:sub_jed,   :))
    call write_var_3d_real(ncFileID, vtVarID,           vt(sub_isd:sub_ied,   sub_jsd:sub_jed,   :))
    call write_var_3d_real(ncFileID, divg_dVarID,   divg_d(sub_isd:sub_ied+1, sub_jsd:sub_jed+1, :))

    ! Close the NetCDF file
    call close_file(ncFileID)

  end subroutine write_subdomain

  !------------------------------------------------------------------
  ! interpolate_state
  !
  ! Increase the database by interpFactor.
  !------------------------------------------------------------------

  subroutine interpolate_state(interpFactor)

    integer, intent(in) :: interpFactor

    ! Locals.
    ! Define new variables to replace the old variables.
    ! Once the new variables are defined, they will 
    ! replace the old variables.

    integer new_ied
    integer new_jed

    real, allocatable :: new_rsin2  (:,:)
    real, allocatable :: new_dxa    (:,:)
    real, allocatable :: new_dya    (:,:)
    real, allocatable :: new_cosa_s (:,:)
    real, allocatable :: new_rarea  (:,:)
    real, allocatable :: new_sina_v (:,:)
    real, allocatable :: new_cosa_v (:,:)
    real, allocatable :: new_rsin_v (:,:)
    real, allocatable :: new_rdyc   (:,:)
    real, allocatable :: new_dx     (:,:)
    real, allocatable :: new_dyc    (:,:)
    real, allocatable :: new_sina_u (:,:)
    real, allocatable :: new_cosa_u (:,:)
    real, allocatable :: new_rdxc   (:,:)
    real, allocatable :: new_dy     (:,:)
    real, allocatable :: new_dxc    (:,:)
    real, allocatable :: new_rsin_u (:,:)
    real, allocatable :: new_rarea_c(:,:)
    real, allocatable :: new_fC     (:,:)
    real, allocatable :: new_sin_sg (:,:,:)
    real, allocatable :: new_cos_sg (:,:,:)
    real, allocatable :: new_delpc  (:,:,:)
    real, allocatable :: new_delp   (:,:,:)
    real, allocatable :: new_ptc    (:,:,:)
    real, allocatable :: new_pt     (:,:,:)
    real, allocatable :: new_w      (:,:,:)
    real, allocatable :: new_ua     (:,:,:)
    real, allocatable :: new_va     (:,:,:)
    real, allocatable :: new_wc     (:,:,:)
    real, allocatable :: new_ut     (:,:,:)
    real, allocatable :: new_vt     (:,:,:)
    real, allocatable :: new_v      (:,:,:)
    real, allocatable :: new_uc     (:,:,:)
    real, allocatable :: new_u      (:,:,:)
    real, allocatable :: new_vc     (:,:,:)
    real, allocatable :: new_divg_d (:,:,:)

    integer :: odims(6), idims(6)

    odims(1) = isd
    odims(2) = ied
    odims(3) = jsd
    odims(4) = jed
    call interpolateCalculateSpace2D(odims, interpFactor, idims)
    allocate (new_rsin2(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(rsin2, odims, new_rsin2, idims, interpFactor)
    allocate (new_dxa(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(dxa, odims, new_dxa, idims, interpFactor)
    allocate (new_dya(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(dya, odims, new_dya, idims, interpFactor)
    allocate (new_cosa_s(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(cosa_s, odims, new_cosa_s, idims, interpFactor)
    allocate (new_rarea(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(rarea, odims, new_rarea, idims, interpFactor)

    new_ied = idims(2) ! Beginning subscripts are unchanged.
    new_jed = idims(4) ! No changes to the 3rd dimension.

    odims(1) = isd
    odims(2) = ied
    odims(3) = jsd
    odims(4) = jed + 1
    call interpolateCalculateSpace2D(odims, interpFactor, idims)
    allocate (new_sina_v(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(sina_v, odims, new_sina_v, idims, interpFactor)
    allocate (new_cosa_v(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(cosa_v, odims, new_cosa_v, idims, interpFactor)
    allocate (new_rsin_v(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(rsin_v, odims, new_rsin_v, idims, interpFactor)
    allocate (new_rdyc(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(rdyc, odims, new_rdyc, idims, interpFactor)
    allocate (new_dx(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(dx, odims, new_dx, idims, interpFactor)
    allocate (new_dyc(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(dyc, odims, new_dyc, idims, interpFactor)

    odims(1) = isd
    odims(2) = ied + 1
    odims(3) = jsd
    odims(4) = jed
    call interpolateCalculateSpace2D(odims, interpFactor, idims)
    allocate (new_sina_u(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(sina_u, odims, new_sina_u, idims, interpFactor)
    allocate (new_cosa_u(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(cosa_u, odims, new_cosa_u, idims, interpFactor)
    allocate (new_rdxc(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(rdxc, odims, new_rdxc, idims, interpFactor)
    allocate (new_dy(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(dy, odims, new_dy, idims, interpFactor)
    allocate (new_dxc(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(dxc, odims, new_dxc, idims, interpFactor)
    allocate (new_rsin_u(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(rsin_u, odims, new_rsin_u, idims, interpFactor)

    odims(1) = isd
    odims(2) = ied + 1
    odims(3) = jsd
    odims(4) = jed + 1
    call interpolateCalculateSpace2D(odims, interpFactor, idims)
    allocate (new_rarea_c(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(rarea_c, odims, new_rarea_c, idims, interpFactor)
    allocate (new_fC(idims(1):idims(2), idims(3):idims(4)))
    call interpolateArray2D(fC, odims, new_fC, idims, interpFactor)

    odims(1) = isd
    odims(2) = ied
    odims(3) = jsd
    odims(4) = jed
    odims(5) = 1
    odims(6) = 9
    call interpolateCalculateSpace3D(odims, interpFactor, idims)
    allocate (new_sin_sg(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(sin_sg, odims, new_sin_sg, idims, interpFactor)
    allocate (new_cos_sg(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(cos_sg, odims, new_cos_sg, idims, interpFactor)

    odims(1) = isd
    odims(2) = ied
    odims(3) = jsd
    odims(4) = jed
    odims(5) = 1
    odims(6) = npz
    call interpolateCalculateSpace3D(odims, interpFactor, idims)
    allocate (new_delpc(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(delpc, odims, new_delpc, idims, interpFactor)
    allocate (new_delp(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(delp, odims, new_delp, idims, interpFactor)
    allocate (new_ptc(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(ptc, odims, new_ptc, idims, interpFactor)
    allocate (new_pt(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(pt, odims, new_pt, idims, interpFactor)
    allocate (new_w(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(w, odims, new_w, idims, interpFactor)
    allocate (new_ua(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(ua, odims, new_ua, idims, interpFactor)
    allocate (new_va(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(va, odims, new_va, idims, interpFactor)
    allocate (new_wc(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(wc, odims, new_wc, idims, interpFactor)
    allocate (new_ut(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(ut, odims, new_ut, idims, interpFactor)
    allocate (new_vt(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(vt, odims, new_vt, idims, interpFactor)

    odims(1) = isd
    odims(2) = ied + 1
    odims(3) = jsd
    odims(4) = jed
    odims(5) = 1
    odims(6) = npz
    call interpolateCalculateSpace3D(odims, interpFactor, idims)
    allocate (new_v(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(v, odims, new_v, idims, interpFactor)
    allocate (new_uc(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(uc, odims, new_uc, idims, interpFactor)

    odims(1) = isd
    odims(2) = ied
    odims(3) = jsd
    odims(4) = jed + 1
    odims(5) = 1
    odims(6) = npz
    call interpolateCalculateSpace3D(odims, interpFactor, idims)
    allocate (new_u(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(u, odims, new_u, idims, interpFactor)
    allocate (new_vc(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(vc, odims, new_vc, idims, interpFactor)
    
    odims(1) = isd
    odims(2) = ied + 1
    odims(3) = jsd
    odims(4) = jed + 1
    odims(5) = 1
    odims(6) = npz
    call interpolateCalculateSpace3D(odims, interpFactor, idims)
    allocate (new_divg_d(idims(1):idims(2), idims(3):idims(4), idims(5):idims(6)))
    call interpolateArray3D(divg_d, odims, new_divg_d, idims, interpFactor)

    ! Exchange the old dimensions to the new dimensions.
    ied = new_ied
    jed = new_jed
    is = 1
    ie = new_ied - 3
    js = 1
    je = new_jed - 3
    nord = 1
    npx = ie + 1
    npy = je + 1
    npz = idims(6)
    !npz = 255

    ! State is allocated with the new dimensions.
    call allocate_state()

    ! Switch the old arrays to the new arrays.
    rarea  (isd:ied,   jsd:jed     ) = new_rarea  (isd:ied,   jsd:jed     )
    rarea_c(isd:ied+1, jsd:jed+1   ) = new_rarea_c(isd:ied+1, jsd:jed+1   )
    sin_sg (isd:ied,   jsd:jed,   :) = new_sin_sg (isd:ied,   jsd:jed,   :)
    cos_sg (isd:ied,   jsd:jed,   :) = new_cos_sg (isd:ied,   jsd:jed,   :)
    sina_v (isd:ied,   jsd:jed+1   ) = new_sina_v (isd:ied,   jsd:jed+1   )
    cosa_v (isd:ied,   jsd:jed+1   ) = new_cosa_v (isd:ied,   jsd:jed+1   )
    sina_u (isd:ied+1, jsd:jed     ) = new_sina_u (isd:ied+1, jsd:jed     )
    cosa_u (isd:ied+1, jsd:jed     ) = new_cosa_u (isd:ied+1, jsd:jed     )
    fC     (isd:ied+1, jsd:jed+1   ) = new_fC     (isd:ied+1, jsd:jed+1   )
    rdxc   (isd:ied+1, jsd:jed     ) = new_rdxc   (isd:ied+1, jsd:jed     )
    rdyc   (isd:ied,   jsd:jed+1   ) = new_rdyc   (isd:ied,   jsd:jed+1   )
    dx     (isd:ied,   jsd:jed+1   ) = new_dx     (isd:ied,   jsd:jed+1   )
    dy     (isd:ied+1, jsd:jed     ) = new_dy     (isd:ied+1, jsd:jed     )
    dxc    (isd:ied+1, jsd:jed     ) = new_dxc    (isd:ied+1, jsd:jed     )
    dyc    (isd:ied,   jsd:jed+1   ) = new_dyc    (isd:ied,   jsd:jed+1   )
    cosa_s (isd:ied,   jsd:jed     ) = new_cosa_s (isd:ied,   jsd:jed     )
    rsin_u (isd:ied+1, jsd:jed     ) = new_rsin_u (isd:ied+1, jsd:jed     )
    rsin_v (isd:ied,   jsd:jed+1   ) = new_rsin_v (isd:ied,   jsd:jed+1   )
    rsin2  (isd:ied,   jsd:jed     ) = new_rsin2  (isd:ied,   jsd:jed     )
    dxa    (isd:ied,   jsd:jed     ) = new_dxa    (isd:ied,   jsd:jed     )
    dya    (isd:ied,   jsd:jed     ) = new_dya    (isd:ied,   jsd:jed     )
    delpc  (isd:ied,   jsd:jed,   :) = new_delpc  (isd:ied,   jsd:jed,   :)
    delp   (isd:ied,   jsd:jed,   :) = new_delp   (isd:ied,   jsd:jed,   :)
    ptc    (isd:ied,   jsd:jed,   :) = new_ptc    (isd:ied,   jsd:jed,   :)
    pt     (isd:ied,   jsd:jed,   :) = new_pt     (isd:ied,   jsd:jed,   :)
    u      (isd:ied,   jsd:jed+1, :) = new_u      (isd:ied,   jsd:jed+1, :)
    v      (isd:ied+1, jsd:jed,   :) = new_v      (isd:ied+1, jsd:jed,   :)
    w      (isd:ied,   jsd:jed,   :) = new_w      (isd:ied,   jsd:jed,   :)
    uc     (isd:ied+1, jsd:jed,   :) = new_uc     (isd:ied+1, jsd:jed,   :)
    ua     (isd:ied,   jsd:jed,   :) = new_ua     (isd:ied,   jsd:jed,   :)
    vc     (isd:ied,   jsd:jed+1, :) = new_vc     (isd:ied,   jsd:jed+1, :)
    va     (isd:ied,   jsd:jed,   :) = new_va     (isd:ied,   jsd:jed,   :)
    wc     (isd:ied,   jsd:jed,   :) = new_wc     (isd:ied,   jsd:jed,   :)
    ut     (isd:ied,   jsd:jed,   :) = new_ut     (isd:ied,   jsd:jed,   :)
    vt     (isd:ied,   jsd:jed,   :) = new_vt     (isd:ied,   jsd:jed,   :)
    divg_d (isd:ied+1, jsd:jed+1, :) = new_divg_d (isd:ied+1, jsd:jed+1, :)

    ! Deallocate the new arrays.  They are not needed now.
    deallocate( new_rarea   )  
    deallocate( new_rarea_c )
    deallocate( new_sin_sg  )
    deallocate( new_cos_sg  )
    deallocate( new_sina_v  )
    deallocate( new_cosa_v  )
    deallocate( new_sina_u  )
    deallocate( new_cosa_u  )
    deallocate( new_fC      )
    deallocate( new_rdxc    )
    deallocate( new_rdyc    )
    deallocate( new_dx      )
    deallocate( new_dy      )
    deallocate( new_dxc     )
    deallocate( new_dyc     )
    deallocate( new_cosa_s  )
    deallocate( new_rsin_u  )
    deallocate( new_rsin_v  )
    deallocate( new_rsin2   )
    deallocate( new_dxa     )
    deallocate( new_dya     )
    deallocate( new_delpc   )
    deallocate( new_delp    )
    deallocate( new_ptc     )
    deallocate( new_pt      )
    deallocate( new_u       )
    deallocate( new_v       )
    deallocate( new_w       )
    deallocate( new_uc      )
    deallocate( new_ua      )
    deallocate( new_vc      )
    deallocate( new_va      )
    deallocate( new_wc      )
    deallocate( new_ut      )
    deallocate( new_vt      )
    deallocate( new_divg_d  )

  end subroutine interpolate_state

end module sw_core_mod

