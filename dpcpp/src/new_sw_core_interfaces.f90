! Copyright (C) 2020-2021 Intel Corporation
 
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
 
! http://www.apache.org/licenses/LICENSE-2.0
 
! Unless required by applicable law or agreed to in writing,
! software distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions
! and limitations under the License.
 

! SPDX-License-Identifier: Apache-2.0


      module new_sw_core_interfaces
      implicit none
      interface
        subroutine c_sw_loop(isd, ied, jsd, jed, &
          is, ie, js, je, nord, npx, &
          npy, npz, do_profile, dt2, &
          sw_corner, se_corner, nw_corner, ne_corner, &
          rarea, rarea_c, &
          sin_sg, cos_sg, sina_v, &
          cosa_v, sina_u, cosa_u, &
          fC, rdxc, rdyc, dx, &
          dy, dxc, dyc, cosa_s, &
          rsin_u, rsin_v, rsin2, &
          dxa, dya, delpc, delp, &
          ptc, pt, u, v, w, &
          uc, vc, ua, va, wc, &
          ut, vt, divg_d) bind(C, name='c_sw_loop')
          integer, intent(   in) :: isd, ied, jsd, jed, is, ie, js, je
          integer, intent(   in) :: nord, npx, npy, npz
          integer, intent(   in) :: do_profile
          real,    intent(   in) :: dt2
          logical, intent(   in) :: sw_corner, se_corner, ne_corner, nw_corner
          real,    intent(inout), dimension(*) :: rarea, rarea_c, sin_sg, cos_sg, sina_v
          real,    intent(inout), dimension(*) :: cosa_v, sina_u, cosa_u, fC, rdxc, rdyc, dx
          real,    intent(inout), dimension(*) :: dy, dxc, dyc, cosa_s, rsin_u, rsin_v, rsin2
          real,    intent(inout), dimension(*) :: dxa, dya
          real,    intent(inout), dimension(*) :: u, vc
          real,    intent(inout), dimension(*) :: v, uc
          real,    intent(inout), dimension(*) :: delp, pt, ua
          real,    intent(inout), dimension(*) :: va, ut, vt, w
          real,    intent(  out), dimension(*) :: delpc, ptc, wc
          real,    intent(  out), dimension(*) :: divg_d
        end subroutine c_sw_loop

        subroutine d2a2c_vect_from_f(sw_corner, se_corner, ne_corner, nw_corner, &
              sin_sg,cosa_u,cosa_v,cosa_s,rsin_u,rsin_v,rsin2,dxa,dya, &
              u, v, ua, va, uc, vc, ut, vt) bind(C, name='d2a2c_vect_from_f')
          logical, intent(in) :: sw_corner, se_corner, ne_corner, nw_corner
          real, intent(in),  dimension(*) :: sin_sg
          real, intent(in),  dimension(*) :: cosa_u,rsin_u
          real, intent(in),  dimension(*) :: cosa_v,rsin_v
          real, intent(in),  dimension(*) :: cosa_s,rsin2,dxa,dya
          real, intent(in),  dimension(*) :: u
          real, intent(in),  dimension(*) :: v
          real, intent(out), dimension(*) :: uc
          real, intent(out), dimension(*) :: vc
          real, intent(out), dimension(*) :: ua, va, ut, vt
        end subroutine d2a2c_vect_from_f
        subroutine fill_4corners_from_f(q, dir, sw_corner, se_corner, &
              ne_corner, nw_corner) bind(C, name='fill_4corners_from_f')
          integer, intent(   in) :: dir     ! 1: x-dir; 2: y-dir
          logical, intent(   in) :: sw_corner, se_corner, ne_corner, nw_corner
          real,    intent(inout), dimension(*) :: q
        end subroutine fill_4corners_from_f
        subroutine fill2_4corners_from_f(q1, q2, dir, sw_corner, se_corner, &
              ne_corner, nw_corner) bind(C, name='fill2_4corners_from_f')
          integer, intent(   in) :: dir     ! 1: x-dir; 2: y-dir
          logical, intent(   in) :: sw_corner, se_corner, ne_corner, nw_corner
          real,    intent(inout), dimension(*) :: q1, q2
        end subroutine fill2_4corners_from_f
        subroutine divergence_corner_from_f(sw_corner, se_corner, ne_corner, nw_corner, &
              rarea_c, sin_sg, cos_sg, cosa_u, cosa_v, sina_u, sina_v, &
              dxc, dyc, u, v, ua, va, divg_d) bind(C, name='divergence_corner_from_f')
          logical, intent( in) :: sw_corner, se_corner, ne_corner, nw_corner
          real,    intent( in), dimension(*) :: rarea_c
          real,    intent( in), dimension(*) :: sin_sg, cos_sg
          real,    intent( in), dimension(*) :: sina_v, cosa_v
          real,    intent( in), dimension(*) :: sina_u, cosa_u
          real,    intent( in), dimension(*) :: dxc
          real,    intent( in), dimension(*) :: dyc
          real,    intent( in), dimension(*) :: u
          real,    intent( in), dimension(*) :: v
          real,    intent( in), dimension(*) :: ua, va
          real,    intent(out), dimension(*) :: divg_d
        end subroutine divergence_corner_from_f
      end interface
      end module