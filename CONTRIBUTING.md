# Contributing to c_sw

NOTE: If you are reading this with a plain text editor, please note that this 
document is formatted with Markdown syntax elements.  See 
https://www.markdownguide.org/cheat-sheet/ for more information. It is
recommended to view this document on [GitHub](https://github.com/NOAA-GSL/SENA-c_sw)


## Contents

[How to Contribute](#how-to-contribute)

[Branch Management](#branch-management)

[Pull Request Rules and Guidlines](#pull-request-rules-and-guidelines)

[Fortran Style Guide](#fortran-style-guide)

## How to Contribute

Contributions to c_sw will be accepted via [pull request](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request),
either from branches in this repository, or from a fork of this repository. Pull
requests will be reviewed and evaluated based on the technical merit of the
proposed changes as well as conformance to the style guidelines outlined in this
document. Code reviewers may request changes as a condition of acceptance of the
pull request.

## Branch Management

This repository follows the [GitHub Flow](https://guides.github.com/introduction/flow/)
branching model with the following modifications borrowed from
[Git Flow](https://nvie.com/posts/a-successful-git-branching-model/):

* All branches that add new features, capabilities, or enhancements must be named:
`feature/name-of-my-feature`
* All branches that fix defects must be named: `bugfix/name-of-my-bugfix`

In addition to the naming conventions, all branches must have a clearly defined,
singular purpose, described by their name. It is also prefered, but not required,
that branches correspond to an issue documented in the issue tracking system. (Issues
can be added after branch creation, if necessary.)  All branches shall result in a
pull request and shall be deleted immediately after that pull request is merged.

## Pull Request Rules and Guidelines

We ask contributors to please be mindful of the following when submitting a Pull
Request. The following will help reviewers give meaningful and timely feedback.

* Pull requests will not be accepted for branches that are not up-to-date with
the current `develop` branch.
* Pull requests will not be accepted unless all tests pass.
* Pull requests will not be accepted without the addition of tests for any new
code or feature being proposed.
* Provide a good description of the changes you are proposing and reference any
issues they resolve. Also link to other related or dependent pull requests in
the description.
* Pull requests should be as small as possible. Whenever possible, break large
changes into multiple smaller changes. If your pull request is too big, reviewers
may ask you to break it into smaller, more digestable, pieces.
* Group changes that logically contribute to the branch's singular purpose
together.
* Do not group unrelated changes together; create separate branches and submit
pull requests for them separately.

## Fortran Style Guide

Unfortunately, there appears not to be a reliable linter for Fortran that can be
used to automate conformance of Fortran coding style. The code in this repository
should use a consistent style throughout such that it appears it was written by a
single person.

The following rules apply when modifying existing code, or contributing new code to this repository.

* Do not use upper case

  ```
  ! Use this
  program foo
    integer :: foobar
    integer :: foo_bar  ! This is also okay
    integer :: fooBar   ! This is also okay
  end program foo

  ! Instead of this
  PROGRAM FOO
    INTEGER :: FOOBAR
  END PROGRAM FOO
  ```

* Use two spaces to indent all code inside `program`

  ```
  ! Use this
  program foo
    integer :: foobar
  end program foo

  ! Instead of this
  program foo
  integer :: foobar
  end program foo

  ! Please do not use this, either
  program foo
        integer :: foobar
  end program foo
  ```

* Use two spaces to indent all code inside `module`

  ```
  ! Use this
  module foo
    foo :: integer
  contains
    subroutine bar
    end subroutine bar
  end module foo
  
  ! Instead of this
  module foo
  foo :: integer
  contains
  subroutine bar
  end subroutine bar
  end module foo

* Use two spaces to indent all code inside `subroutine` and `function`, `if`, `do`, `while`, etc.

  ```
  ! Use this
  subroutine foo(bar)
    bar :: integer
    bar = bar + 1
  end subroutine bar

  ! Instead of
  subroutine foo(bar)
  bar :: integer
  bar = bar + 1
       bar = bar + 1  ! Please do not do this, either
  end subroutine bar
  ```

* Use two spaces to indent all code inside `if`, `do`, `while`, etc.

  ```
  ! Use this
  if (bar > 1) then
    do while (bar < 10)
      write(*, *) "Bar"
      bar = bar + 1
    end do
    end if
  end if

  ! Instead of
  if (bar > 1) then
  do while (bar < 10)
  write(*, *) "Bar"
  bar = bar + 1
  end do
  end if
  end if

  ! Please do not do this, either
  if (bar > 1) then
       do while (bar < 10)
            write(*, *) "Bar"
            bar = bar + 1
       end do
  end if
  ```

* Use spaces after commas

  ```
  ! Use this
  write(*, '(A, I)') "The number is", a(i, j)

  ! Instead of
  write(*,'(A,I)') "The number is",a(i,j)
  ```
  
* Use spaces around operators

  ```
  ! Use this
  x = a(i, j) * 1.0 - pi / (rho + phi)

  ! Instead of
  x=a(i,j)*1.0-pi/(rho+phi)
  ```

* Do NOT use spaces before the open parenthesis when calling a function

  ```
  ! Use this
  write(*, *) "Foo"
  call bar(x)

  ! Instead of
  write (*, *) "Foo"
  call bar (x)
  ```

* Align variable and intent declarations

  ```
  ! Use this
  subroutine foo(x, y, z)
    integer, intent(   in) :: x
    real,    intent(  out) :: y
    logical, intent(inout) :: z

    real, allocatable :: foobar(:,:)
    real              :: baz
    integer           :: zap

  ! Instead of
  subroutine foo(x, y, z)
    integer, intent(in) :: x
    real, intent(out) :: y
    logical, intent(inout) :: z

    real, allocatable :: fobar(:,:)
    real :: baz
    integer :: zap
  ```

* Declare subroutine arguments in the same order they appear in the argument list

  ```
  ! Use this
  subroutine foo(a, b, c)
    integer :: a
    real    :: b
    logical :: c

  ! Instead of
  subroutine foo(a, b, c)
    logical :: c
    integer :: a
    real    :: b    
  ```

* Specify full name in `end` statements

  ```
  ! Use this
  program foo

  end program foo

  module bar
 
  contains

    subroutine alpha

    end subroutine alpha

  end module bar


  ! Instead of
  program foo

  end program

  module bar
 
  contains

    subroutine alpha

    end subroutine

  end module
  ```
