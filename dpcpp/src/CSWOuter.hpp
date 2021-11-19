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

#pragma once


#include "arrayfuncs.hpp"
#include <CL/sycl.hpp>

///The outer implementation does not offload j and i loops, just the k loop which iterates
///typically 0 <= k < 127.
class CSWOuter
{
//member variables
private:
    static constexpr double big_number=1.0e8;
    static constexpr double a1=0.5625;
    static constexpr double a2=-0.0625;
    static constexpr double c1=-2./14.;
    static constexpr double c2=11./14.;
    static constexpr double c3=5./14.;

    const int isd;
    const int ied;
    const int jsd;
    const int jed;
    const int is;
    const int ie;
    const int js;
    const int je;
    const int nord;
    const int npx;
    const int npy;
    const size_t npz;
    const int do_profile;

    const double dt2;
    const bool sw_corner, se_corner, nw_corner, ne_corner;

    //some small helper values to reduce code
    const int isize; //ied-isd+1
    const int jsize; //jed-jsd+1
    const int ksize; //9
    const int isizep1; //iszie+1
    const int jsizep1; //jsize +1
    
    const Offset2DArray<double, true> rarea;
    const Offset2DArray<double, true> rarea_c;
    const Offset3DArray<double, true> sin_sg;
    const Offset3DArray<double, true> cos_sg;
    const Offset2DArray<double, true> sina_v;
    const Offset2DArray<double, true> cosa_v;
    const Offset2DArray<double, true> sina_u;
    const Offset2DArray<double, true> cosa_u;
    const Offset2DArray<double, true> fC;
    const Offset2DArray<double, true> rdxc;
    const Offset2DArray<double, true> rdyc;
    const Offset2DArray<double, true> dx;
    const Offset2DArray<double, true> dy;
    const Offset2DArray<double, true> dxc;
    const Offset2DArray<double, true> dyc;
    const Offset2DArray<double, true> cosa_s;
    const Offset2DArray<double, true> rsin_u;
    const Offset2DArray<double, true> rsin_v;
    const Offset2DArray<double, true> rsin2;
    const Offset2DArray<double, true> dxa;
    const Offset2DArray<double, true> dya;

//functions
public:
    SYCL_EXTERNAL CSWOuter(const int isd_in, const int ied_in, const int jsd_in, const int jed_in, 
        const int is_in, const int ie_in, const int js_in, const int je_in, const int nord_in, const int npx_in, 
        const int npy_in, const int npz_in, const int do_profile_in, double dt2_in, 
        const bool sw_corner_in, const bool se_corner_in, const bool nw_corner_in, const bool ne_corner_in,
        double *prarea, double *prarea_c,
        double *psin_sg, double *pcos_sg, double *psina_v,
        double *pcosa_v,  double *psina_u,  double *pcosa_u,
        double *pfC,  double *prdxc,  double *prdyc,  double *pdx,
        double *pdy,  double *pdxc,  double *pdyc,  double *pcosa_s,
        double *prsin_u,  double *prsin_v,  double *prsin2,
        double *pdxa,  double *pdya);

    SYCL_EXTERNAL CSWOuter(const int isd_in, const int ied_in, const int jsd_in, const int jed_in, 
        const int is_in, const int ie_in, const int js_in, const int je_in, const int nord_in, const int npx_in, 
        const int npy_in, const int npz_in, const int do_profile_in, double dt2_in, 
        const bool sw_corner_in, const bool se_corner_in, const bool nw_corner_in, const bool ne_corner_in,
        const Offset2DArray<double, true>& rarea, const Offset2DArray<double, true>& rarea_c,
        const Offset3DArray<double, true>& sin_sg, const Offset3DArray<double, true>& cos_sg,
        const Offset2DArray<double, true>& sina_v, const Offset2DArray<double, true>& cosa_v,
        const Offset2DArray<double, true>& sina_u, const Offset2DArray<double, true>& cosa_u,
        const Offset2DArray<double, true>& fC, const Offset2DArray<double, true>& rdxc,
        const Offset2DArray<double, true>& rdyc, const Offset2DArray<double, true>& dx,
        const Offset2DArray<double, true>& dy, const Offset2DArray<double, true>& dxc,
        const Offset2DArray<double, true>& dyc, const Offset2DArray<double, true>& cosa_s,
        const Offset2DArray<double, true>& rsin_u, const Offset2DArray<double, true>& rsin_v,
        const Offset2DArray<double, true>& rsin2, const Offset2DArray<double, true>& dxa,
        const Offset2DArray<double, true>& dya);

    SYCL_EXTERNAL void c_sw(Offset2DArray<double, true>& delpc, Offset2DArray<double, true>& delp, 
        Offset2DArray<double, true>& ptc, Offset2DArray<double, true>& pt, 
        Offset2DArray<double, true>& u, Offset2DArray<double, true>& v, 
        Offset2DArray<double, true>& w, Offset2DArray<double, true>& uc,
        Offset2DArray<double, true>& vc, Offset2DArray<double, true>& ua, 
        Offset2DArray<double, true>& va, Offset2DArray<double, true>& wc,
        Offset2DArray<double, true>& ut, Offset2DArray<double, true>& vt, 
        Offset2DArray<double, true>& divg_d,
        Offset2DArray<double, true>& vort, Offset2DArray<double, true>& ke, 
        Offset2DArray<double, true>& fx, Offset2DArray<double, true>& fx1, 
        Offset2DArray<double, true>& fx2, Offset2DArray<double, true>& fy, 
        Offset2DArray<double, true>& fy1, Offset2DArray<double, true>& fy2, 
        Offset2DArray<double, true>& utmp, Offset2DArray<double, true>& vtmp, 
        Offset2DArray<double, true>& uf, Offset2DArray<double, true>& vf) const;

private:

    SYCL_EXTERNAL void d2a2c_vect(Offset2DArray<double, true>& u, Offset2DArray<double, true>& v, 
        Offset2DArray<double, true>& ua, Offset2DArray<double, true>& va, 
        Offset2DArray<double, true>& uc, Offset2DArray<double, true>& vc, 
        Offset2DArray<double, true>& ut, Offset2DArray<double, true>& vt,
        Offset2DArray<double, true>& utmp, Offset2DArray<double, true>& vtmp) const;

    SYCL_EXTERNAL double edge_interpolate4(double *ua, double *dxa) const;

    SYCL_EXTERNAL void fill_4corners(Offset2DArray<double, true>& q, int dir) const;

    SYCL_EXTERNAL void fill2_4corners(Offset2DArray<double, true>& q1, Offset2DArray<double, true>& q2, int dir) const;

    SYCL_EXTERNAL void divergence_corner(Offset2DArray<double, true>& u, Offset2DArray<double, true>& v, 
        Offset2DArray<double, true>& ua, Offset2DArray<double, true>& va, 
        Offset2DArray<double, true>& divg_d,
        Offset2DArray<double, true>& uf, Offset2DArray<double, true>& vf) const;
};