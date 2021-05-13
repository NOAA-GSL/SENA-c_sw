!------------------------------------------------------------------
! Program to test the MPI halo exchange
! This program must be run using 4 MPI tasks
! This program runs 3 tests with (rows,cols) = (1,4) (2,2) (4,1)
!------------------------------------------------------------------
program test_exchange_driver

  use sw_core_mod
  use mpi

  implicit none

  ! Local variables
  integer :: i, j, k    ! Indexes
  integer :: test       ! Index for the three tests to be run
  integer :: ierr       ! Error return
  integer :: errorcode  ! Error return for MPI_Abort
  real, allocatable :: usave(:,:,:) ! Array for saving u for testing.

  call MPI_Init(ierr)
  if( ierr /= 0 ) then
    print*,'Error initializing MPI. Error =', ierr
    call MPI_Abort(MPI_COMM_WORLD, errorcode, ierr)
    stop 1
  endif

  ! Get the MPI rank
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  if( ierr /= 0 ) then
    print*,'Error getting the MPI rank',ierr,rank
  endif
  ! Get the number of MPI tasks
  call MPI_Comm_size(MPI_COMM_WORLD, nTasks, ierr)
  if( ierr /= 0 ) then
    print*,'Error getting the number of MPI tasks',ierr,nTasks
    call MPI_Abort(MPI_COMM_WORLD, errorcode, ierr)
    stop 2
  endif
  if ( nTasks /= 4 ) then
      print*,'Error: The number of MPI tasks must equal 4'
      print*,'The number of MPI tasks =',nTasks
      call MPI_Abort(MPI_COMM_WORLD, errorcode, ierr)
      stop 3
    endif
  if( rank == 0 ) then
    print*,'Running in parallel mode with 4 MPI tasks.'
  endif

  !Set indexes for a 48x48 case.
  isd =  -2
  jsd =  -2
  ied =  51
  jed =  51
  npz = 127

  !Allocate variables
  call allocate_state
  allocate(usave (isd:ied, jsd:jed+1, npz))

  ! Run 3 test using 4 MPI tasks with (rows,cols) = (1,4) (2,2) (4,1)
  rows = 1
  do test = 1, 3

    cols = 4/rows

    ! Get the MPI task neighbors for the exchange
    call get_neighbors

    !Fill the u array
    do k = 1, npz
      do j = jsd, jed+1
        do i = isd, ied
          u(i, j, k) = 10000*k + 100*j + i
        enddo
      enddo
    enddo

    ! Copy u to usave for testing the exchange
    usave = u

    ! Exchange the halo for the variable u
    call exchange

    !Test the exchange
    do k = 1, npz
      do j = jsd, jed+1
        do i = isd, ied
          if( u(i, j, k) /= usave(i, j, k) ) then
            print*,'Error: u /= usave at i,j,k=',i,j,k,u(i, j, k),usave(i, j, k)
            print*,'Error: u(i, j, k),usave(i, j, k)=',u(i, j, k),usave(i, j, k)
            call MPI_Abort(MPI_COMM_WORLD, errorcode, ierr)
            stop 4
          endif
        enddo
      enddo
    enddo

  enddo !test=1, 3

  call MPI_Barrier(MPI_COMM_WORLD,ierr)

  if( rank == 0 ) then
    print*,'Halo exchange passed all three tests.'
  endif

  call MPI_Finalize(ierr)

end program test_exchange_driver
