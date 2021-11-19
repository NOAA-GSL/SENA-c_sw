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

#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <chrono>

#include "new_sw_core.hpp"
#include <CL/sycl.hpp>
#include "arrayfuncs.hpp"
#include "CSWOuter.hpp"
#include "CSWMixed.hpp"

#ifdef ENABLE_MPI
    #include <mpi.h>
#endif

using namespace sycl;

///Enum to define which version to use. At some point we want to move all c++
///versions to this implementation and have a switch between them All
///The 'Outer' case is the offload only in the 'k' variable.
///The 'Mixed' Testcase if the offload of collapsed k,j,i loops
enum Testcase
{
    OUTER = 0,
    //ORIGINAL = 1,
    MIXED = 2,
    //LOOP = 3
};

///Global variable defining the used testcase.
///Could be moved to a command line flag at some point.
constexpr Testcase tcase = MIXED;



///handle sycl errors as proposed by reinders book
auto handle_async_error = [](exception_list elist) {
  for (auto &e : elist) {
    try{ std::rethrow_exception(e); }
    catch ( sycl::exception& e ) {
      std::cout << "ASYNC EXCEPTION!!\n";
      std::cout << e.what() << "\n";
    }
  }
};


///Calls the actual case to run. The actual allocations are performed in an implementation
///of a static member function of the CSWVariants class.
///The actual calls to the device code is implemented in CSWOuter.hpp and CSWMixed.hpp
void c_sw_loop(int* isd_in, int* ied_in, int* jsd_in, int* jed_in, 
                int *is_in, int *ie_in, int *js_in, int *je_in, int *nord_in, int *npx_in, 
                int *npy_in, int *npz_in, int *do_profile_in, double *dt2_in, 
                bool *sw_corner_in, bool *se_corner_in, bool *nw_corner_in, bool *ne_corner_in,
                double *rarea_in, double *rarea_c_in,
                double *sin_sg_in, double *cos_sg_in, double *sina_v_in,
                double *cosa_v_in, double *sina_u_in, double *cosa_u_in,
                double *fC_in, double *rdxc_in, double *rdyc_in, double *dx_in,
                double *dy_in, double *dxc_in, double *dyc_in, double *cosa_s_in,
                double *rsin_u_in, double *rsin_v_in, double *rsin2_in,
                double *dxa_in, double *dya_in, double *delpc_in, double *delp_in, 
                double *ptc_in, double *pt_in, double *u_in, double *v_in, double *w_in,
                double *uc_in, double *vc_in, double *ua_in, double *va_in, double *wc_in,
                double *ut_in, double *vt_in, double *divg_d_in)
{
    ///Check which testcase to run
    if (tcase == OUTER) 
    {
        #ifdef ENABLE_MPI
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        #endif
        std::cout << "--------- RUNNING OUTER TESTCASE ---------" << std::endl;
        CSWVariants::c_sw_outer(default_selector{}, usm::alloc::device, *isd_in, *ied_in, *jsd_in, *jed_in, 
                *is_in, *ie_in, *js_in, *je_in, *nord_in, *npx_in, 
                *npy_in, *npz_in, *do_profile_in, *dt2_in, 
                *sw_corner_in, *se_corner_in, *nw_corner_in, *ne_corner_in,
                rarea_in, rarea_c_in,
                sin_sg_in, cos_sg_in, sina_v_in,
                cosa_v_in, sina_u_in, cosa_u_in,
                fC_in, rdxc_in, rdyc_in, dx_in,
                dy_in, dxc_in, dyc_in, cosa_s_in,
                rsin_u_in, rsin_v_in, rsin2_in,
                dxa_in, dya_in, delpc_in, delp_in, 
                ptc_in, pt_in, u_in, v_in, w_in,
                uc_in, vc_in, ua_in, va_in, wc_in,
                ut_in, vt_in, divg_d_in);
    }
    else if (tcase == MIXED) 
    {
        #ifdef ENABLE_MPI
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        #endif
        std::cout << "--------- RUNNING MIXED TESTCASE ---------" << std::endl;
        CSWVariants::c_sw_mixed(default_selector{}, usm::alloc::device, *isd_in, *ied_in, *jsd_in, *jed_in, 
                *is_in, *ie_in, *js_in, *je_in, *nord_in, *npx_in, 
                *npy_in, *npz_in, *do_profile_in, *dt2_in, 
                *sw_corner_in, *se_corner_in, *nw_corner_in, *ne_corner_in,
                rarea_in, rarea_c_in,
                sin_sg_in, cos_sg_in, sina_v_in,
                cosa_v_in, sina_u_in, cosa_u_in,
                fC_in, rdxc_in, rdyc_in, dx_in,
                dy_in, dxc_in, dyc_in, cosa_s_in,
                rsin_u_in, rsin_v_in, rsin2_in,
                dxa_in, dya_in, delpc_in, delp_in, 
                ptc_in, pt_in, u_in, v_in, w_in,
                uc_in, vc_in, ua_in, va_in, wc_in,
                ut_in, vt_in, divg_d_in);
    }
}


void CSWVariants::c_sw_outer(const sycl::device_selector& selector, 
    usm::alloc alloc_type, int isd, int ied, int jsd, int jed, 
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
    double *ut_in, double *vt_in, double *divg_d_in)
{
    //std::cout << "Calling Outer" << std::endl;
    queue Q{selector, handle_async_error};

    #ifdef ENABLE_MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    #endif
    

    const int isize = ied-isd+1;
    const int jsize = jed-jsd+1;
    const int ksize = 9;
    const int isizep1 = isize+1;
    const int jsizep1 = jsize+1;
    const int bufisize = ie+1-(is-1)+1;
    const int bufjsize = je+1-(js-1)+1;
    const int bufisizep1 = bufisize+1;
    const int bufjsizep1 = bufjsize+1;

    #ifdef ENABLE_MPI
    if (rank == 0)
    #endif
    std::cout << "Device : "<< Q.get_device().get_info<info::device::name>() << std::endl <<
        "Memory Address Alignment : " << Q.get_device().get_info<info::device::mem_base_addr_align>() << std::endl <<
        "Slice size isize x jsize = " << isize << " x " << jsize << " = " << isize*jsize << 
        " doubles (=" << isize*jsize*sizeof(double) << " bytes)" << std::endl <<
        "npz = " << npz << std::endl;
    

    //usm_allocator<const double, usm::alloc::device> const_device_allocator(Q);
    Allocator memallocator(usm::alloc::device);
    const Offset2DArray<double, true> rarea = memallocator.Allocate<double, true>(isd, ied, jsd, jed, rarea_in, Q);
    const Offset2DArray<double, true> rarea_c = memallocator.Allocate<double, true>(isd, ied+1, jsd, jed+1, rarea_c_in, Q);
    const Offset3DArray<double, true> sin_sg = memallocator.Allocate<double, true>(isd, ied, jsd, jed, 1, 9, sin_sg_in, Q);
    const Offset3DArray<double, true> cos_sg = memallocator.Allocate<double, true>(isd, ied, jsd, jed, 1, 9, cos_sg_in, Q);
    const Offset2DArray<double, true> sina_v = memallocator.Allocate<double, true>(isd, ied,  jsd, jed+1, sina_v_in, Q);
    const Offset2DArray<double, true> cosa_v = memallocator.Allocate<double, true>(isd, ied,  jsd, jed+1, cosa_v_in, Q);
    const Offset2DArray<double, true> sina_u = memallocator.Allocate<double, true>(isd, ied+1, jsd, jed, sina_u_in, Q);
    const Offset2DArray<double, true> cosa_u = memallocator.Allocate<double, true>(isd, ied+1, jsd, jed, cosa_u_in, Q);
    const Offset2DArray<double, true> fC = memallocator.Allocate<double, true>(isd, ied+1, jsd, jed+1, fC_in, Q);
    const Offset2DArray<double, true> rdxc = memallocator.Allocate<double, true>(isd, ied+1, jsd, jed, rdxc_in, Q);
    const Offset2DArray<double, true> rdyc = memallocator.Allocate<double, true>(isd, ied,  jsd, jed+1, rdyc_in, Q);
    const Offset2DArray<double, true> dx = memallocator.Allocate<double, true>(isd, ied,  jsd, jed+1, dx_in, Q);
    const Offset2DArray<double, true> dy = memallocator.Allocate<double, true>(isd, ied+1, jsd, jed, dy_in, Q);
    const Offset2DArray<double, true> dxc = memallocator.Allocate<double, true>(isd, ied+1, jsd, jed, dxc_in, Q);
    const Offset2DArray<double, true> dyc = memallocator.Allocate<double, true>(isd, ied,  jsd, jed+1, dyc_in, Q);
    const Offset2DArray<double, true> cosa_s = memallocator.Allocate<double, true>(isd, ied,  jsd, jed, cosa_s_in, Q);
    const Offset2DArray<double, true> rsin_u = memallocator.Allocate<double, true>(isd, ied+1, jsd, jed, rsin_u_in, Q);
    const Offset2DArray<double, true> rsin_v = memallocator.Allocate<double, true>(isd, ied,  jsd, jed+1, rsin_v_in, Q);
    const Offset2DArray<double, true> rsin2 = memallocator.Allocate<double, true>(isd, ied,  jsd, jed, rsin2_in, Q);
    const Offset2DArray<double, true> dxa = memallocator.Allocate<double, true>(isd, ied,  jsd, jed, dxa_in, Q);
    const Offset2DArray<double, true> dya = memallocator.Allocate<double, true>(isd, ied,  jsd, jed, dya_in, Q);

    ///temporary storage
    
    double *pvort  = memallocator.Allocate<double>(bufisize*bufjsize*npz, nullptr, Q);
    double *pke    = memallocator.Allocate<double>(bufisize*bufjsize*npz, nullptr, Q);
    double *pfx    = memallocator.Allocate<double>(bufisizep1*bufjsize*npz, nullptr, Q);
    double *pfx1   = memallocator.Allocate<double>(bufisizep1*bufjsize*npz, nullptr, Q);
    double *pfx2   = memallocator.Allocate<double>(bufisizep1*bufjsize*npz, nullptr, Q);
    double *pfy    = memallocator.Allocate<double>(bufisize*bufjsizep1*npz, nullptr, Q);
    double *pfy1   = memallocator.Allocate<double>(bufisize*bufjsizep1*npz, nullptr, Q);
    double *pfy2   = memallocator.Allocate<double>(bufisize*bufjsizep1*npz, nullptr, Q);
    double *putmp  = memallocator.Allocate<double>(isize*jsize*npz, nullptr, Q);
    double *pvtmp  = memallocator.Allocate<double>(isize*jsize*npz, nullptr, Q);
    double *puf    = memallocator.Allocate<double>((bufisize+2)*bufjsizep1*npz, nullptr, Q);
    double *pvf    = memallocator.Allocate<double>(bufisizep1*(bufjsize+2)*npz, nullptr, Q);

    //move data to device
    double *pdelpc = memallocator.Allocate<double>(isize*jsize*npz, delpc_in, Q);
    double *pdelp = memallocator.Allocate<double>(isize*jsize*npz, delp_in, Q);
    double *pptc = memallocator.Allocate<double>(isize*jsize*npz, ptc_in, Q);
    double *ppt = memallocator.Allocate<double>(isize*jsize*npz, pt_in, Q);
    double *pu = memallocator.Allocate<double>(isize*jsizep1*npz, u_in, Q);
    double *pv = memallocator.Allocate<double>(isizep1*jsize*npz, v_in, Q);
    double *pw = memallocator.Allocate<double>(isize*jsize*npz, w_in, Q);
    double *puc = memallocator.Allocate<double>(isizep1*jsize*npz, uc_in, Q);
    double *pvc = memallocator.Allocate<double>(isize*jsizep1*npz, vc_in, Q);
    double *pua = memallocator.Allocate<double>(isize*jsize*npz, ua_in, Q);
    double *pva = memallocator.Allocate<double>(isize*jsize*npz, va_in, Q);
    double *pwc = memallocator.Allocate<double>(isize*jsize*npz, wc_in, Q);
    double *put = memallocator.Allocate<double>(isize*jsize*npz, ut_in, Q);
    double *pvt = memallocator.Allocate<double>(isize*jsize*npz, vt_in, Q);
    double *pdivg_d = memallocator.Allocate<double>(isizep1*jsizep1*npz, divg_d_in, Q); 
    Q.wait();
        Q.submit([&](handler &h) {
        CSWOuter Foo(isd, ied, jsd, jed, is, ie, js, je, nord, npx, npy, npz, do_profile, 
            dt2, sw_corner, se_corner, nw_corner, ne_corner, rarea, rarea_c, sin_sg,
            cos_sg, sina_v, cosa_v, sina_u, cosa_u, fC, rdxc, rdyc, dx, dy, dxc, dyc,
            cosa_s, rsin_u, rsin_v, rsin2, dxa, dya);
        
        h.parallel_for(npz, [=](auto k) {
            Offset2DArray<double, true> delpc(isd, ied, jsd, jed, pdelpc+k*isize*jsize);
            Offset2DArray<double, true> delp(isd, ied, jsd, jed, pdelp+k*isize*jsize);
            Offset2DArray<double, true> ptc(isd, ied, jsd, jed, pptc+k*isize*jsize);
            Offset2DArray<double, true> pt(isd, ied, jsd, jed, ppt+k*isize*jsize);
            Offset2DArray<double, true> u(isd, ied, jsd, jed+1, pu+k*isize*jsizep1);
            Offset2DArray<double, true> v(isd, ied+1, jsd, jed, pv+k*isizep1*jsize);
            Offset2DArray<double, true> w(isd, ied, jsd, jed, pw+k*isize*jsize);
            Offset2DArray<double, true> uc(isd, ied+1, jsd, jed, puc+k*isizep1*jsize);
            Offset2DArray<double, true> vc(isd, ied, jsd, jed+1, pvc+k*isize*jsizep1);
            Offset2DArray<double, true> ua(isd, ied, jsd, jed, pua+k*isize*jsize);
            Offset2DArray<double, true> va(isd, ied, jsd, jed, pva+k*isize*jsize);
            Offset2DArray<double, true> wc(isd, ied, jsd, jed, pwc+k*isize*jsize);
            Offset2DArray<double, true> ut(isd, ied, jsd, jed, put+k*isize*jsize);
            Offset2DArray<double, true> vt(isd, ied, jsd, jed, pvt+k*isize*jsize);
            Offset2DArray<double, true> divg_d(isd, ied+1, jsd, jed+1, pdivg_d+k*isizep1*jsizep1);

            Offset2DArray<double, true> vort(is-1, ie+1, js-1, je+1, pvort+k*bufisize*bufjsize);
            Offset2DArray<double, true> ke(is-1, ie+1, js-1, je+1, pke+k*bufisize*bufjsize);
            Offset2DArray<double, true> fx(is-1, ie+2, js-1, je+1, pfx+k*bufisizep1*bufjsize);
            Offset2DArray<double, true> fx1(is-1, ie+2, js-1, je+1, pfx1+k*bufisizep1*bufjsize);
            Offset2DArray<double, true> fx2(is-1, ie+2, js-1, je+1, pfx2+k*bufisizep1*bufjsize);
            Offset2DArray<double, true> fy(is-1, ie+1, js-1, je+2, pfy+k*bufisize*bufjsizep1);
            Offset2DArray<double, true> fy1(is-1, ie+1, js-1, je+2, pfy1+k*bufisize*bufjsizep1);
            Offset2DArray<double, true> fy2(is-1, ie+1, js-1, je+2, pfy2+k*bufisize*bufjsizep1);
            Offset2DArray<double, true> utmp(isd, ied, jsd, jed, putmp+k*isize*jsize);
            Offset2DArray<double, true> vtmp(isd, ied, jsd, jed, pvtmp+k*isize*jsize);
            Offset2DArray<double, true> uf(is-2, ie+2, js-1, je+2, puf+k*(bufisize+2)*bufjsizep1);
            Offset2DArray<double, true> vf(is-1, ie+2, js-2, je+2, pvf+k*bufisizep1*(bufjsize+2));

            Foo.c_sw(delpc, delp, ptc, pt, u, v, w, uc, vc, ua, va, wc, ut, vt, divg_d,
                vort, ke, fx, fx1, fx2, fy, fy1, fy2, utmp, vtmp, uf, vf);
        });
    }).wait();

    auto t1 = std::chrono::high_resolution_clock::now();
    
    Q.submit([&](handler &h) {
        CSWOuter Foo(isd, ied, jsd, jed, is, ie, js, je, nord, npx, npy, npz, do_profile, 
            dt2, sw_corner, se_corner, nw_corner, ne_corner, rarea, rarea_c, sin_sg,
            cos_sg, sina_v, cosa_v, sina_u, cosa_u, fC, rdxc, rdyc, dx, dy, dxc, dyc,
            cosa_s, rsin_u, rsin_v, rsin2, dxa, dya);
        
        h.parallel_for(npz, [=](auto k) {
            Offset2DArray<double, true> delpc(isd, ied, jsd, jed, pdelpc+k*isize*jsize);
            Offset2DArray<double, true> delp(isd, ied, jsd, jed, pdelp+k*isize*jsize);
            Offset2DArray<double, true> ptc(isd, ied, jsd, jed, pptc+k*isize*jsize);
            Offset2DArray<double, true> pt(isd, ied, jsd, jed, ppt+k*isize*jsize);
            Offset2DArray<double, true> u(isd, ied, jsd, jed+1, pu+k*isize*jsizep1);
            Offset2DArray<double, true> v(isd, ied+1, jsd, jed, pv+k*isizep1*jsize);
            Offset2DArray<double, true> w(isd, ied, jsd, jed, pw+k*isize*jsize);
            Offset2DArray<double, true> uc(isd, ied+1, jsd, jed, puc+k*isizep1*jsize);
            Offset2DArray<double, true> vc(isd, ied, jsd, jed+1, pvc+k*isize*jsizep1);
            Offset2DArray<double, true> ua(isd, ied, jsd, jed, pua+k*isize*jsize);
            Offset2DArray<double, true> va(isd, ied, jsd, jed, pva+k*isize*jsize);
            Offset2DArray<double, true> wc(isd, ied, jsd, jed, pwc+k*isize*jsize);
            Offset2DArray<double, true> ut(isd, ied, jsd, jed, put+k*isize*jsize);
            Offset2DArray<double, true> vt(isd, ied, jsd, jed, pvt+k*isize*jsize);
            Offset2DArray<double, true> divg_d(isd, ied+1, jsd, jed+1, pdivg_d+k*isizep1*jsizep1);

            Offset2DArray<double, true> vort(is-1, ie+1, js-1, je+1, pvort+k*bufisize*bufjsize);
            Offset2DArray<double, true> ke(is-1, ie+1, js-1, je+1, pke+k*bufisize*bufjsize);
            Offset2DArray<double, true> fx(is-1, ie+2, js-1, je+1, pfx+k*bufisizep1*bufjsize);
            Offset2DArray<double, true> fx1(is-1, ie+2, js-1, je+1, pfx1+k*bufisizep1*bufjsize);
            Offset2DArray<double, true> fx2(is-1, ie+2, js-1, je+1, pfx2+k*bufisizep1*bufjsize);
            Offset2DArray<double, true> fy(is-1, ie+1, js-1, je+2, pfy+k*bufisize*bufjsizep1);
            Offset2DArray<double, true> fy1(is-1, ie+1, js-1, je+2, pfy1+k*bufisize*bufjsizep1);
            Offset2DArray<double, true> fy2(is-1, ie+1, js-1, je+2, pfy2+k*bufisize*bufjsizep1);
            Offset2DArray<double, true> utmp(isd, ied, jsd, jed, putmp+k*isize*jsize);
            Offset2DArray<double, true> vtmp(isd, ied, jsd, jed, pvtmp+k*isize*jsize);
            Offset2DArray<double, true> uf(is-2, ie+2, js-1, je+2, puf+k*(bufisize+2)*bufjsizep1);
            Offset2DArray<double, true> vf(is-1, ie+2, js-2, je+2, pvf+k*bufisizep1*(bufjsize+2));

            Foo.c_sw(delpc, delp, ptc, pt, u, v, w, uc, vc, ua, va, wc, ut, vt, divg_d,
                vort, ke, fx, fx1, fx2, fy, fy1, fy2, utmp, vtmp, uf, vf);
        });
    }).wait();

    auto t2 = std::chrono::high_resolution_clock::now();
    #ifdef ENABLE_MPI
    const double time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/1000.0;
    MPI_Barrier(MPI_COMM_WORLD);
    std::vector<double> alltimes(world_size);
    MPI_Gather(&time, 1, MPI_DOUBLE, &alltimes[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0)
        std::cout << "Time in kernel = " << *std::max_element(alltimes.begin(), alltimes.end()) << " s" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    #else
    std::cout << "Time in kernel = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/1000.0 
            << " s" << std::endl;
    #endif

        Q.submit([&](handler &h) {
        CSWOuter Foo(isd, ied, jsd, jed, is, ie, js, je, nord, npx, npy, npz, do_profile, 
            dt2, sw_corner, se_corner, nw_corner, ne_corner, rarea, rarea_c, sin_sg,
            cos_sg, sina_v, cosa_v, sina_u, cosa_u, fC, rdxc, rdyc, dx, dy, dxc, dyc,
            cosa_s, rsin_u, rsin_v, rsin2, dxa, dya);
        
        h.parallel_for(npz, [=](auto k) {
            Offset2DArray<double, true> delpc(isd, ied, jsd, jed, pdelpc+k*isize*jsize);
            Offset2DArray<double, true> delp(isd, ied, jsd, jed, pdelp+k*isize*jsize);
            Offset2DArray<double, true> ptc(isd, ied, jsd, jed, pptc+k*isize*jsize);
            Offset2DArray<double, true> pt(isd, ied, jsd, jed, ppt+k*isize*jsize);
            Offset2DArray<double, true> u(isd, ied, jsd, jed+1, pu+k*isize*jsizep1);
            Offset2DArray<double, true> v(isd, ied+1, jsd, jed, pv+k*isizep1*jsize);
            Offset2DArray<double, true> w(isd, ied, jsd, jed, pw+k*isize*jsize);
            Offset2DArray<double, true> uc(isd, ied+1, jsd, jed, puc+k*isizep1*jsize);
            Offset2DArray<double, true> vc(isd, ied, jsd, jed+1, pvc+k*isize*jsizep1);
            Offset2DArray<double, true> ua(isd, ied, jsd, jed, pua+k*isize*jsize);
            Offset2DArray<double, true> va(isd, ied, jsd, jed, pva+k*isize*jsize);
            Offset2DArray<double, true> wc(isd, ied, jsd, jed, pwc+k*isize*jsize);
            Offset2DArray<double, true> ut(isd, ied, jsd, jed, put+k*isize*jsize);
            Offset2DArray<double, true> vt(isd, ied, jsd, jed, pvt+k*isize*jsize);
            Offset2DArray<double, true> divg_d(isd, ied+1, jsd, jed+1, pdivg_d+k*isizep1*jsizep1);

            Offset2DArray<double, true> vort(is-1, ie+1, js-1, je+1, pvort+k*bufisize*bufjsize);
            Offset2DArray<double, true> ke(is-1, ie+1, js-1, je+1, pke+k*bufisize*bufjsize);
            Offset2DArray<double, true> fx(is-1, ie+2, js-1, je+1, pfx+k*bufisizep1*bufjsize);
            Offset2DArray<double, true> fx1(is-1, ie+2, js-1, je+1, pfx1+k*bufisizep1*bufjsize);
            Offset2DArray<double, true> fx2(is-1, ie+2, js-1, je+1, pfx2+k*bufisizep1*bufjsize);
            Offset2DArray<double, true> fy(is-1, ie+1, js-1, je+2, pfy+k*bufisize*bufjsizep1);
            Offset2DArray<double, true> fy1(is-1, ie+1, js-1, je+2, pfy1+k*bufisize*bufjsizep1);
            Offset2DArray<double, true> fy2(is-1, ie+1, js-1, je+2, pfy2+k*bufisize*bufjsizep1);
            Offset2DArray<double, true> utmp(isd, ied, jsd, jed, putmp+k*isize*jsize);
            Offset2DArray<double, true> vtmp(isd, ied, jsd, jed, pvtmp+k*isize*jsize);
            Offset2DArray<double, true> uf(is-2, ie+2, js-1, je+2, puf+k*(bufisize+2)*bufjsizep1);
            Offset2DArray<double, true> vf(is-1, ie+2, js-2, je+2, pvf+k*bufisizep1*(bufjsize+2));

            Foo.c_sw(delpc, delp, ptc, pt, u, v, w, uc, vc, ua, va, wc, ut, vt, divg_d,
                vort, ke, fx, fx1, fx2, fy, fy1, fy2, utmp, vtmp, uf, vf);
        });
    }).wait();


    //copy results back
    Q.memcpy(delpc_in, pdelpc, sizeof(double)*isize*jsize*npz);
    Q.memcpy(delp_in, pdelp, sizeof(double)*isize*jsize*npz);
    Q.memcpy(ptc_in, pptc, sizeof(double)*isize*jsize*npz);
    Q.memcpy(pt_in, ppt, sizeof(double)*isize*jsize*npz);
    Q.memcpy(u_in, pu, sizeof(double)*isize*jsizep1*npz);
    Q.memcpy(v_in, pv, sizeof(double)*isizep1*jsize*npz);
    Q.memcpy(w_in, pw, sizeof(double)*isize*jsize*npz);
    Q.memcpy(uc_in, puc, sizeof(double)*isizep1*jsize*npz);
    Q.memcpy(vc_in, pvc, sizeof(double)*isize*jsizep1*npz);
    Q.memcpy(ua_in, pua, sizeof(double)*isize*jsize*npz);
    Q.memcpy(va_in, pva, sizeof(double)*isize*jsize*npz);
    Q.memcpy(wc_in, pwc, sizeof(double)*isize*jsize*npz);
    Q.memcpy(ut_in, put, sizeof(double)*isize*jsize*npz);
    Q.memcpy(vt_in, pvt, sizeof(double)*isize*jsize*npz);
    Q.memcpy(divg_d_in, pdivg_d, sizeof(double)*isizep1*jsizep1*npz); 
    Q.wait();

    //Free memory
    memallocator.Deallocate<double, true>(rarea, nullptr, Q);
    memallocator.Deallocate<double, true>(rarea_c, nullptr, Q);
    memallocator.Deallocate<double, true>(sin_sg, nullptr, Q);
    memallocator.Deallocate<double, true>(cos_sg, nullptr, Q);
    memallocator.Deallocate<double, true>(sina_v, nullptr, Q);
    memallocator.Deallocate<double, true>(cosa_v, nullptr, Q);
    memallocator.Deallocate<double, true>(sina_u, nullptr, Q);
    memallocator.Deallocate<double, true>(cosa_u, nullptr, Q);
    memallocator.Deallocate<double, true>(fC, nullptr, Q);
    memallocator.Deallocate<double, true>(rdxc, nullptr, Q);
    memallocator.Deallocate<double, true>(rdyc, nullptr, Q);
    memallocator.Deallocate<double, true>(dx, nullptr, Q);
    memallocator.Deallocate<double, true>(dy, nullptr, Q);
    memallocator.Deallocate<double, true>(dxc, nullptr, Q);
    memallocator.Deallocate<double, true>(dyc, nullptr, Q);
    memallocator.Deallocate<double, true>(cosa_s, nullptr, Q);
    memallocator.Deallocate<double, true>(rsin_u, nullptr, Q);
    memallocator.Deallocate<double, true>(rsin_v, nullptr, Q);
    memallocator.Deallocate<double, true>(rsin2, nullptr, Q);
    memallocator.Deallocate<double, true>(dxa, nullptr, Q);
    memallocator.Deallocate<double, true>(dya, nullptr, Q);
    


    free(pdelpc, Q);
    free(pdelp, Q);
    free(pptc, Q);
    free(ppt, Q);
    free(pu, Q);
    free(pv, Q);
    free(pw, Q);
    free(puc, Q);
    free(pvc, Q);
    free(pua, Q);
    free(pva, Q);
    free(pwc, Q);
    free(put, Q);
    free(pvt, Q);
    free(pdivg_d, Q);
    free(pvort, Q);
    free(pke, Q);
    free(pfx, Q);
    free(pfx1, Q);
    free(pfx2, Q);
    free(pfy, Q);
    free(pfy1, Q);
    free(pfy2, Q);
    free(putmp, Q);
    free(pvtmp, Q);
    free(puf, Q);
    free(pvf, Q);
}

void CSWVariants::c_sw_mixed(const sycl::device_selector& selector, 
    usm::alloc alloc_type, int isd, int ied, int jsd, int jed, 
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
    double *ut_in, double *vt_in, double *divg_d_in)
{
    //std::cout << "Calling Mixed" << std::endl;
    queue Q{selector, handle_async_error};

    #ifdef ENABLE_MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    #endif

    const int isize = ied-isd+1;
    const int jsize = jed-jsd+1;
    const int ksize = 9;
    const int isizep1 = isize+1;
    const int jsizep1 = jsize+1;
    const int bufisize = ie+1-(is-1)+1;
    const int bufjsize = je+1-(js-1)+1;
    const int bufisizep1 = bufisize+1;
    const int bufjsizep1 = bufjsize+1;

    #ifdef ENABLE_MPI
    if (rank == 0)
    #endif
    std::cout << "Device : "<< Q.get_device().get_info<info::device::name>() << std::endl <<
        "Memory Address Alignment : " << Q.get_device().get_info<info::device::mem_base_addr_align>() << std::endl <<
        "Slice size isize x jsize = " << isize << " x " << jsize << " = " << isize*jsize << 
        " doubles (=" << isize*jsize*sizeof(double) << " bytes)" << std::endl <<
        "npz = " << npz << std::endl;
    
  double dt4;
  int i, j;
  int iep1, jep1;
  int ret;

#ifdef ENABLE_GPTL
  if (do_profile == 1)
  {
    ret = gptlstart('c_sw');
  }
#endif

  iep1 = ie + 1;
  jep1 = je + 1;

  d2a2c_vect(sw_corner, se_corner, ne_corner, nw_corner, sin_sg, cosa_u,
      cosa_v, cosa_s, rsin_u, rsin_v, rsin2, dxa, dya, u, v, ua, va, uc, vc,
      ut, vt);

  if (nord > 0)
  {
/*
    divergence_corner(sw_corner, se_corner, ne_corner, nw_corner,
                      rarea_c, sin_sg, cos_sg, cosa_u, cosa_v, sina_u,
                      sina_v, dxc, dyc, u, v, ua, va, divg_d);
*/
    divergence_corner(sw_corner, se_corner, ne_corner, nw_corner,
                      rarea_c, sin_sg, cos_sg, dxc, dyc, u, v, ua, va, divg_d);
  }
  
  for (j=js-1; j<=jep1; j++)
  {
    for (i=is-1; i<=iep1+1; i++)
    {
      if (ut(i,j)>0.0)
      {
        ut(i,j)=dt2*ut(i,j)*dy(i,j)*sin_sg(i-1,j,3);
      } else
      {
        ut(i,j)=dt2*ut(i,j)*dy(i,j)*sin_sg(i,j,1);
      }
    }
  }

  for (j=js-1; j<=je+2; j++)
  {
    for (i=is-1; i<=iep1; i++)
    {
      if (vt(i,j)>0.0)
      {
        vt(i, j) = dt2 * vt(i, j) * dx(i, j) * sin_sg(i, j-1, 4);
      } else
      {
        vt(i, j) = dt2 * vt(i, j) * dx(i, j) * sin_sg(i, j, 2);
      }
    }
  }

  //----------------
  // Transport delp:
  //----------------
  // Xdir:
  fill2_4corners(delp, pt, 1, sw_corner, se_corner, ne_corner, nw_corner);
  fill_4corners(w, 1, sw_corner, se_corner, ne_corner, nw_corner);
  for (j=js-1; j<=je+1; j++)
  {
    for (i=is-1; i<=ie+2; i++)
    {
      if (ut(i,j)>0.0)
      {
        fx1(i, j) = delp(i-1, j);
        fx(i, j) = pt(i-1, j);
        fx2(i, j) = w(i-1, j);
      } else
      {
        fx1(i, j) = delp(i, j);
        fx(i, j) = pt(i, j);
        fx2(i, j) = w(i, j);
      }
      fx1(i, j) = ut(i, j) * fx1(i, j);
      fx(i, j) = fx1(i, j) * fx(i, j);
      fx2(i, j) = fx1(i, j) * fx2(i, j);
    }
  }

  // Ydir:
  fill2_4corners(delp, pt, 2, sw_corner, se_corner, ne_corner, nw_corner);
  fill_4corners(w, 2, sw_corner, se_corner, ne_corner, nw_corner);
  for (j=js-1; j<=je+2; j++)
  {
    for (i=is-1; i<=ie+1; i++)
    {
      if (vt(i,j)>0)
      {
        fy1(i, j) = delp(i, j-1);
        fy(i, j) = pt(i, j-1);
        fy2(i, j) = w(i, j-1);
      } else
      {
        fy1(i, j) = delp(i, j);
        fy(i, j) = pt(i, j);
        fy2(i, j) = w(i, j);
      }
      fy1(i, j) =  vt(i, j) * fy1(i, j);
      fy(i, j) = fy1(i, j) * fy(i, j);
      fy2(i, j) = fy1(i, j) * fy2(i, j);
    }
  }  
  
  for (j=js-1; j<=je+1; j++)
  {
    for (i=is-1; i<=ie+1; i++)
    {
      delpc(i, j) = delp(i, j) + (fx1(i, j) - fx1(i+1, j) + fy1(i, j) - fy1(i, j+1)) * rarea(i, j);
      ptc(i, j) = (pt(i, j) * delp(i, j) + (fx(i, j) - fx(i+1, j) + fy(i, j) - fy(i, j+1)) * rarea(i, j)) / delpc(i,j);
      wc(i, j) = (w(i, j) * delp(i, j) + (fx2(i, j) - fx2(i+1, j) + fy2(i, j) - fy2(i, j+1)) * rarea(i, j)) / delpc(i, j);
    }
  }

  //------------
  // Compute KE:
  //------------

  // Since uc = u*, i.e. the covariant wind perpendicular to the face edge, if
  // we want to compute kinetic energy we will need the true coordinate-parallel
  // covariant wind, computed through u = uc*sina + v*cosa.
  //
  // Use the alpha for the cell KE is being computed in.

    
    // std::vector<CSWMixed::CSWData> vdata;
    // vdata.emplace_back(isd, ied, jsd, jed, 
    //     is, ie, js, je, nord, npx, 
    //     npy, npz, do_profile, dt2, 
    //     sw_corner, se_corner, nw_corner, ne_corner,
    //     rarea, rarea_c, sin_sg, cos_sg, sina_v, cosa_v, sina_u, cosa_u,
    //     fC, rdxc, rdyc, dx, dy, dxc, dyc, cosa_s, rsin_u, rsin_v,
    //     rsin2, dxa, dya, delpc, delp, ptc, pt, u, v, 
    //     w, uc, vc, ua, va, wc, ut, vt, divg_d, vort, ke, 
    //     fx, fx1, fx2, fy, fy1, fy2, utmp, vtmp, uf, vf);
    // buffer bdata(vdata);
    // bdata.set_final_data(nullptr);
    // host_accessor accdata(bdata);
    // CSWMixed::CSWData Foo = accdata[0];

    CSWMixed::CSWData Foo(isd, ied, jsd, jed, 
        is, ie, js, je, nord, npx, 
        npy, npz, do_profile, dt2, 
        sw_corner, se_corner, nw_corner, ne_corner,
        rarea, rarea_c, sin_sg, cos_sg, sina_v, cosa_v, sina_u, cosa_u,
        fC, rdxc, rdyc, dx, dy, dxc, dyc, cosa_s, rsin_u, rsin_v,
        rsin2, dxa, dya, delpc, delp, ptc, pt, u, v, 
        w, uc, vc, ua, va, wc, ut, vt, divg_d, vort, ke, 
        fx, fx1, fx2, fy, fy1, fy2, utmp, vtmp, uf, vf);

     

    //for timings activate this to get rid of the zeCreateModule in the measuremtn
    //ATTENTION: uncommenting this might yield wrong results
    CSWMixed::c_sw(Foo, Q);

    #ifdef ENABLE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
    #endif
    auto t1 = std::chrono::high_resolution_clock::now();

    CSWMixed::c_sw(Foo, Q);

    auto t2 = std::chrono::high_resolution_clock::now();
    #ifdef ENABLE_MPI
    const double time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/1000.0;
    MPI_Barrier(MPI_COMM_WORLD);
    std::vector<double> alltimes(world_size);
    MPI_Gather(&time, 1, MPI_DOUBLE, &alltimes[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0)
        std::cout << "Time in kernel = " << *std::max_element(alltimes.begin(), alltimes.end()) << " s" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    #else
    std::cout << "Time in kernel = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/1000.0 
            << " s" << std::endl;
    #endif

    //for timings activate this to get rid of the zeCreateModule in the measuremtn
    //ATTENTION: uncommenting this might yield wrong results
    CSWMixed::c_sw(Foo, Q);

    //Free memory and copy results back
    memallocator.Deallocate<double, true>(delpc, delpc_in, Q);
    memallocator.Deallocate<double, true>(delp, delp_in, Q);
    memallocator.Deallocate<double, true>(ptc, ptc_in, Q);
    memallocator.Deallocate<double, true>(pt, pt_in, Q);
    memallocator.Deallocate<double, true>(u, u_in, Q);
    memallocator.Deallocate<double, true>(v, v_in, Q);
    memallocator.Deallocate<double, true>(w, w_in, Q);
    memallocator.Deallocate<double, true>(uc, uc_in, Q);
    memallocator.Deallocate<double, true>(vc, vc_in, Q);
    memallocator.Deallocate<double, true>(ua, ua_in, Q);
    memallocator.Deallocate<double, true>(va, va_in, Q);
    memallocator.Deallocate<double, true>(wc, wc_in, Q);
    memallocator.Deallocate<double, true>(ut, ut_in, Q);
    memallocator.Deallocate<double, true>(vt, vt_in, Q);
    memallocator.Deallocate<double, true>(divg_d, divg_d_in, Q); 

    memallocator.Deallocate<double, true>(vort, nullptr, Q);
    memallocator.Deallocate<double, true>(ke, nullptr, Q);
    memallocator.Deallocate<double, true>(fx, nullptr, Q);
    memallocator.Deallocate<double, true>(fx1, nullptr, Q);
    memallocator.Deallocate<double, true>(fx2, nullptr, Q);
    memallocator.Deallocate<double, true>(fy, nullptr, Q);
    memallocator.Deallocate<double, true>(fy1, nullptr, Q);
    memallocator.Deallocate<double, true>(fy2, nullptr, Q);
    memallocator.Deallocate<double, true>(utmp, nullptr, Q);
    memallocator.Deallocate<double, true>(vtmp, nullptr, Q);
    memallocator.Deallocate<double, true>(uf, nullptr, Q);
    memallocator.Deallocate<double, true>(vf, nullptr, Q);

    memallocator.Deallocate<double, true>(rarea, nullptr, Q);
    memallocator.Deallocate<double, true>(rarea_c, nullptr, Q);
    memallocator.Deallocate<double, true>(sin_sg, nullptr, Q);
    memallocator.Deallocate<double, true>(cos_sg, nullptr, Q);
    memallocator.Deallocate<double, true>(sina_v, nullptr, Q);
    memallocator.Deallocate<double, true>(cosa_v, nullptr, Q);
    memallocator.Deallocate<double, true>(sina_u, nullptr, Q);
    memallocator.Deallocate<double, true>(cosa_u, nullptr, Q);
    memallocator.Deallocate<double, true>(fC, nullptr, Q);
    memallocator.Deallocate<double, true>(rdxc, nullptr, Q);
    memallocator.Deallocate<double, true>(rdyc, nullptr, Q);
    memallocator.Deallocate<double, true>(dx, nullptr, Q);
    memallocator.Deallocate<double, true>(dy, nullptr, Q);
    memallocator.Deallocate<double, true>(dxc, nullptr, Q);
    memallocator.Deallocate<double, true>(dyc, nullptr, Q);
    memallocator.Deallocate<double, true>(cosa_s, nullptr, Q);
    memallocator.Deallocate<double, true>(rsin_u, nullptr, Q);
    memallocator.Deallocate<double, true>(rsin_v, nullptr, Q);
    memallocator.Deallocate<double, true>(rsin2, nullptr, Q);
    memallocator.Deallocate<double, true>(dxa, nullptr, Q);
    memallocator.Deallocate<double, true>(dya, nullptr, Q);
    
}