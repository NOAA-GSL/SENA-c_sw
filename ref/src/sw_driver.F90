!------------------------------------------------------------------
! sw_driver
!
! Driver program to run the c_sw kernel
!------------------------------------------------------------------
program sw_driver

  use OMP_LIB
  use sw_core_mod
#ifdef ENABLE_GPTL
  use gptl
#endif
#ifdef ENABLE_MPI
  use mpi
#endif

  implicit none

  ! Driver variables
  integer :: k                                  ! Vertical level Loop index
  integer :: nthreads                           ! # of OpenMP threads
  integer :: narg                               ! # of command line arguments
  integer :: count_start, count_end, count_rate ! Timer start/stop
  integer :: ret                                ! Return status
  integer :: rank = 0                           ! The rank of MPI tasks
  integer :: ierr                               ! Error return
  integer :: test_file_unit                     ! unit number for writing out test files
  integer, external :: print_affinity           ! External subroutine

  ! Input configuration variables
  character(len=64) :: namelist_file = "test_input/c_sw_12x24.nl"
  character(len=64) :: input_file, output_file, test_file_name
  integer           :: nl_unit

  ! Input namelists
  namelist /io/       input_file, output_file
  namelist /debug/    do_profile                ! Defined in sw_core_mod

#ifdef ENABLE_MPI
  call MPI_INIT(ierr)
#endif

  ! Get the number of arguments
  narg = command_argument_count()
  if (narg /= 1) then
    write(*,*) "Usage: c_sw <namelist_file>"
    stop 1
  end if

  ! Get the MPI rank
#ifdef ENABLE_MPI
  call mpi_comm_rank(mpi_comm_world, rank, ierr)
#endif

  ! Get the namelist file name from the argument list
  call get_command_argument(1, namelist_file)

  ! Open the namelist file
  open(newunit=nl_unit, file=TRIM(namelist_file), form='formatted', status='old')

  ! Read the data IO settings from the namelist
  read(nl_unit, nml=io)

  ! Read the debug settings from the namelist
  read(nl_unit, nml=debug)

  ! Get OMP_NUM_THREADS value
  nthreads = omp_get_max_threads()

  ! Initialize GPTL if enabled
#ifdef ENABLE_GPTL
  if (do_profile == 1) then
    ret = GPTLinitialize()
  end if
#endif

  ! Read the input state from the NetCDF input file
  call read_state(TRIM(input_file))

  ! Print out configuration settings
  if(rank == 0) then
    write(*, '(A,I0,A,I0)') 'Problem size = ', ie - is + 1, "x", je - js + 1
    write(*, '(A,I0)') 'nthreads = ', nthreads
  endif

  ! Write configuration settings to test_file for each rank
#ifdef ENABLE_MPI
  write(test_file_name, '(a,i4.4,a)') 'test_file',rank,'.txt'
  open (newunit=test_file_unit, file=TRIM(test_file_name), form='formatted', status='new')
  write(test_file_unit, '(A,I0,A,I0)') 'Problem size = ', ie - is + 1, "x", je - js + 1
  write(test_file_unit, '(A,I0)') 'nthreads = ', nthreads
#endif

  ret = print_affinity()

  ! Print the input state
  if(rank == 0) then
    call print_state("Input State")
  endif

  ! Write the output state to test_file for each rank
#ifdef ENABLE_MPI
  call write_state_test("Input State",test_file_unit)
#endif

  ! Get the start time
  call system_clock(count_start, count_rate)

#ifdef ENABLE_GPTL
  if (do_profile == 1) then
     ret = gptlstart('kernel')
  end if
#endif

  ! Run the kernel
  !$OMP parallel do schedule(runtime)
  do k=1,npz
     call c_sw(sw_corner, se_corner, nw_corner, ne_corner,           &
               rarea, rarea_c, sin_sg, cos_sg, sina_v, cosa_v,       &
               sina_u, cosa_u, fC, rdxc, rdyc, dx, dy, dxc, dyc,     &
               cosa_s, rsin_u, rsin_v, rsin2, dxa, dya,              &
               delpc(isd,jsd,k), delp(isd,jsd,k), ptc(isd,jsd,k),    &
               pt(isd,jsd,k), u(isd,jsd,k), v(isd,jsd,k),            &
               w(isd,jsd,k), uc(isd,jsd,k), vc(isd,jsd,k),           &
               ua(isd,jsd,k), va(isd,jsd,k), wc(isd,jsd,k),          &
               ut(isd,jsd,k), vt(isd,jsd,k), divg_d(isd,jsd,k), dt2)
  enddo

#ifdef ENABLE_GPTL
  if (do_profile == 1) then
     ret = gptlstop('kernel')
  end if
#endif

  ! Get the stop time
  call system_clock(count_end, count_rate)

  ! Print the output state
  if(rank == 0) then
    call print_state("Output State")
  endif

  ! Write the output state to test_file for each rank
#ifdef ENABLE_MPI
  call write_state_test("Output State",test_file_unit)
#endif

  ! Write the output state to the NetCDF output file
  if(rank == 0) then
    call write_state(TRIM(output_file))
  endif

  ! Write timing information
  if(rank == 0) then
    write(*,*)
    write(*,'(A,F12.9)') "Total time (in seconds)=" ,  ((count_end - count_start) * 1.0) / count_rate
  endif

  ! Deallocate the state variables
  call deallocate_state()

  ! Turn off GPTL if enabled
#ifdef ENABLE_GPTL
  if (do_profile == 1) then
    ret = gptlpr(0)
    ret = gptlfinalize()
  end if
#endif

end program sw_driver
