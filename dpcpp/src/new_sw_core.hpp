/*
Copyright (C) 2020-2021 Intel Corporation
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
http://www.apache.org/licenses/LICENSE-2.0
 
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions
and limitations under the License.
 

SPDX-License-Identifier: Apache-2.0
*/

#ifndef NEW_SW_CORE_HPP
#define NEW_SW_CORE_HPP


#include <CL/sycl.hpp>
   
extern "C" {
  ///Function called by Fortran driver that decides the testcase
  void c_sw_loop(int* isd, int* ied, int* jsd, int* jed, 
                int *is, int *ie, int *js, int *je, int *nord, int *npx, 
                int *npy, int *npz, int *do_profile, double *dt2, 
                bool *sw_corner, bool *se_corner, bool *nw_corner, bool *ne_corner,
                double *rarea, double *rarea_c,
                double *sin_sg, double *cos_sg, double *sina_v,
                double *cosa_v, double *sina_u, double *cosa_u,
                double *fC, double *rdxc, double *rdyc, double *dx,
                double *dy, double *dxc, double *dyc, double *cosa_s,
                double *rsin_u, double *rsin_v, double *rsin2,
                double *dxa, double *dya, double *delpc, double *delp, 
                double *ptc, double *pt, double *u, double *v, double *w,
                double *uc, double *vc, double *ua, double *va, double *wc,
                double *ut, double *vt, double *divg_d);
}

///Implements static member functions which are reposible for the allocation,
///copy and dealltion of device memory. Each new code version (that requires different allocations)
///needs to implement a new function here. THe functions have as parameters
///the device selector for the device where this should be run and the allocation
///type (host, device, shared)
///The actual computations, i.e. the 1:1 translation of the previous new_sw_core.cpp
///is implemented is CSWMixed.hpp and CSWOuter.hpp (depending on the case)
class CSWVariants
{
public:
  static void c_sw_outer(const sycl::device_selector& selector, 
    sycl::usm::alloc alloc_type, int isd, int ied, int jsd, int jed, 
    int is, int ie, int js, int je, int nord, int npx, 
    int npy, int npz, int do_profile, double dt2, 
    bool sw_corner, bool se_corner, bool nw_corner, bool ne_corner,
    double *rarea_in, double *rarea_c_in,
    double *sin_sg_in, double *cos_sg_in, double *sina_v_in,
    double *cosa_v_in, double *sina_u_in, double *cosa_u_in,
    double *fC_in, double *rdxc_in, double *rdyc_in, double *dx_in,
    double *dy_in, double *dxc_in, double *dyc_in, double *cosa_s_in,
    double *rsin_u_in, double *rsin_v_in, double *rsin2_in,
    double *dxa_in, double *dya_in, double *delpc_in, double *delp_in, 
    double *ptc_in, double *pt_in, double *u_in, double *v_in, double *w_in,
    double *uc_in, double *vc_in, double *ua_in, double *va_in, double *wc_in,
    double *ut_in, double *vt_in, double *divg_d_in);

  static void c_sw_mixed(const sycl::device_selector& selector, 
    sycl::usm::alloc alloc_type, int isd, int ied, int jsd, int jed, 
    int is, int ie, int js, int je, int nord, int npx, 
    int npy, int npz, int do_profile, double dt2, 
    bool sw_corner, bool se_corner, bool nw_corner, bool ne_corner,
    double *rarea_in, double *rarea_c_in,
    double *sin_sg_in, double *cos_sg_in, double *sina_v_in,
    double *cosa_v_in, double *sina_u_in, double *cosa_u_in,
    double *fC_in, double *rdxc_in, double *rdyc_in, double *dx_in,
    double *dy_in, double *dxc_in, double *dyc_in, double *cosa_s_in,
    double *rsin_u_in, double *rsin_v_in, double *rsin2_in,
    double *dxa_in, double *dya_in, double *delpc_in, double *delp_in, 
    double *ptc_in, double *pt_in, double *u_in, double *v_in, double *w_in,
    double *uc_in, double *vc_in, double *ua_in, double *va_in, double *wc_in,
    double *ut_in, double *vt_in, double *divg_d_in);
};

#endif /*NEW_SW_CORE_HPP*/





