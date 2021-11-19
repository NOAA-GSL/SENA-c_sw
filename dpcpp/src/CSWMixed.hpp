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

#ifndef CSWMIXED_HPP
#define CSWMIXED_HPP


#include "arrayfuncs.hpp"
#include <CL/sycl.hpp>


///Namespace for the actual dpc++ kernel implementations and calls.
///THe mixed code splits the k loop and collapsed it with j and i loops and offloads those loops 
///separately.
namespace CSWMixed
{
    ///Class that holds the data and used as input to all functions for easier-to-read code.
    class CSWData
    {
    public:
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
        const Offset3DArray<double, true> delpc;
        const Offset3DArray<double, true> delp;
        const Offset3DArray<double, true> ptc;
        const Offset3DArray<double, true> pt; 
        const Offset3DArray<double, true> u; 
        const Offset3DArray<double, true> v; 
        const Offset3DArray<double, true> w; 
        const Offset3DArray<double, true> uc;
        const Offset3DArray<double, true> vc; 
        const Offset3DArray<double, true> ua; 
        const Offset3DArray<double, true> va; 
        const Offset3DArray<double, true> wc;
        const Offset3DArray<double, true> ut; 
        const Offset3DArray<double, true> vt; 
        const Offset3DArray<double, true> divg_d;
        const Offset3DArray<double, true> vort; 
        const Offset3DArray<double, true> ke; 
        const Offset3DArray<double, true> fx; 
        const Offset3DArray<double, true> fx1; 
        const Offset3DArray<double, true> fx2; 
        const Offset3DArray<double, true> fy; 
        const Offset3DArray<double, true> fy1; 
        const Offset3DArray<double, true> fy2; 
        const Offset3DArray<double, true> utmp; 
        const Offset3DArray<double, true> vtmp; 
        const Offset3DArray<double, true> uf; 
        const Offset3DArray<double, true> vf;

        CSWData(const int isd_in, const int ied_in, const int jsd_in, const int jed_in, 
            const int is_in, const int ie_in, const int js_in, const int je_in, const int nord_in, const int npx_in, 
            const int npy_in, const int npz_in, const int do_profile_in, double dt2_in, 
            const bool sw_corner_in, const bool se_corner_in, const bool nw_corner_in, const bool ne_corner_in,
            const Offset2DArray<double, true>& rarea_in, const Offset2DArray<double, true>& rarea_c_in,
            const Offset3DArray<double, true>& sin_sg_in, const Offset3DArray<double, true>& cos_sg_in,
            const Offset2DArray<double, true>& sina_v_in, const Offset2DArray<double, true>& cosa_v_in,
            const Offset2DArray<double, true>& sina_u_in, const Offset2DArray<double, true>& cosa_u_in,
            const Offset2DArray<double, true>& fC_in, const Offset2DArray<double, true>& rdxc_in,
            const Offset2DArray<double, true>& rdyc_in, const Offset2DArray<double, true>& dx_in,
            const Offset2DArray<double, true>& dy_in, const Offset2DArray<double, true>& dxc_in,
            const Offset2DArray<double, true>& dyc_in, const Offset2DArray<double, true>& cosa_s_in,
            const Offset2DArray<double, true>& rsin_u_in, const Offset2DArray<double, true>& rsin_v_in,
            const Offset2DArray<double, true>& rsin2_in, const Offset2DArray<double, true>& dxa_in,
            const Offset2DArray<double, true>& dya_in,
            Offset3DArray<double, true>& delpc_in, Offset3DArray<double, true>& delp_in, 
            Offset3DArray<double, true>& ptc_in, Offset3DArray<double, true>& pt_in, 
            Offset3DArray<double, true>& u_in, Offset3DArray<double, true>& v_in, 
            Offset3DArray<double, true>& w_in, Offset3DArray<double, true>& uc_in,
            Offset3DArray<double, true>& vc_in, Offset3DArray<double, true>& ua_in, 
            Offset3DArray<double, true>& va_in, Offset3DArray<double, true>& wc_in,
            Offset3DArray<double, true>& ut_in, Offset3DArray<double, true>& vt_in, 
            Offset3DArray<double, true>& divg_d_in,
            Offset3DArray<double, true>& vort_in, Offset3DArray<double, true>& ke_in, 
            Offset3DArray<double, true>& fx_in, Offset3DArray<double, true>& fx1_in, 
            Offset3DArray<double, true>& fx2_in, Offset3DArray<double, true>& fy_in, 
            Offset3DArray<double, true>& fy1_in, Offset3DArray<double, true>& fy2_in, 
            Offset3DArray<double, true>& utmp_in, Offset3DArray<double, true>& vtmp_in, 
            Offset3DArray<double, true>& uf_in, Offset3DArray<double, true>& vf_in)  :
            isd(isd_in), ied(ied_in), jsd(jsd_in), jed(jed_in),
            is(is_in), ie(ie_in), js(js_in), je(je_in), nord(nord_in), npx(npx_in), 
            npy(npy_in), npz(npz_in), do_profile(do_profile_in), dt2(dt2_in),
            sw_corner(sw_corner_in), se_corner(se_corner_in), nw_corner(nw_corner_in), ne_corner(ne_corner_in),
            isize(ied-isd+1), jsize(jed-jsd+1), ksize(9), isizep1(isize+1), jsizep1(jsize+1),
            rarea(rarea_in),
            rarea_c(rarea_c_in),
            sin_sg(sin_sg_in),
            cos_sg(cos_sg_in),
            sina_v(sina_v_in),
            cosa_v(cosa_v_in),
            sina_u(sina_u_in),
            cosa_u(cosa_u_in),
            fC(fC_in),
            rdxc(rdxc_in),
            rdyc(rdyc_in),
            dx(dx_in),
            dy(dy_in),
            dxc(dxc_in),
            dyc(dyc_in),
            cosa_s(cosa_s_in),
            rsin_u(rsin_u_in),
            rsin_v(rsin_v_in),
            rsin2(rsin2_in),
            dxa(dxa_in),
            dya(dya_in),
            delpc(delpc_in), delp(delp_in), ptc(ptc_in), pt(pt_in), u(u_in), v(v_in), w(w_in), uc(uc_in),
            vc(vc_in), ua(ua_in), va(va_in), wc(wc_in), ut(ut_in), vt(vt_in), divg_d(divg_d_in), 
            vort(vort_in), ke(ke_in), fx(fx_in), fx1(fx1_in), fx2(fx2_in), fy(fy_in), 
            fy1(fy1_in), fy2(fy2_in), utmp(utmp_in), vtmp(vtmp_in), uf(uf_in), vf(vf_in)
        {}
    };


    double edge_interpolate4(double *ua, double *dxa)
    {
        double u0L, u0R;
        constexpr double half=static_cast<double>(0.5);
        constexpr double two=static_cast<double>(2.0);
        u0L = half*((two*dxa[1]+dxa[0])*ua[1] - dxa[1]*ua[0]) / ( dxa[0]+dxa[1] );
        u0R = half*((two*dxa[2]+dxa[3])*ua[2] - dxa[2]*ua[3]) / ( dxa[2]+dxa[3] );
        return (u0L+u0R);
    }

    void fill_4corners(const int k, const Offset3DArray<double, true>& q, int dir, const CSWData& d)
    {
        // This routine fill the 4 corners of the scalar fields only as needed by c_core
        
        #ifdef ENABLE_GPTL
        if (do_profile == 1)
        {
            ret = gptlstart('fill_4corners');
        }
        #endif
        
        switch(dir)
        {
            case(1): // x-dir
            if (d.sw_corner)
            {
                q(-1, 0,k)      = q(0, 2,k);
                q(0, 0,k)       = q(0, 1,k);
            }
            if (d.se_corner)
            {
                q(d.npx+1, 0,k)   = q(d.npx, 2,k);
                q(d.npx, 0,k)     = q(d.npx, 1,k);
            }
            if (d.nw_corner)
            {
                q(0, d.npy,k)     = q(0, d.npy-1,k);
                q(-1, d.npy,k)    = q(0, d.npy-2,k);
            }
            if (d.ne_corner)
            {
                q(d.npx, d.npy,k)   = q(d.npx, d.npy-1,k);
                q(d.npx+1, d.npy,k) = q(d.npx, d.npy-2,k);
            }
            break;
            case(2): // y-dir
            if (d.sw_corner)
            {
                q(0, 0,k)       = q(1, 0,k);
                q(0, -1,k)      = q(2, 0,k);
            }
            if (d.se_corner)
            {
                q(d.npx, 0,k)     = q(d.npx-1, 0,k);
                q(d.npx, -1,k)    = q(d.npx-2, 0,k);
            }
            if (d.nw_corner)
            {
                q(0, d.npy,k)     = q(1, d.npy,k);
                q(0, d.npy+1,k)   = q(2, d.npy,k);
            }
            if (d.ne_corner)
            {
                q(d.npx, d.npy,k)   = q(d.npx-1, d.npy,k);
                q(d.npx, d.npy+1,k) = q(d.npx-2, d.npy,k);
            }
            break;
        }

        #ifdef ENABLE_GPTL
        if (do_profile == 1)
        {
            ret = gptlstop('fill_4corners');
        }
        #endif

    }

    void fill2_4corners(const int k, const Offset3DArray<double, true>& q1, const Offset3DArray<double, true>& q2, int dir, const CSWData& d)
    {
    // This routine fills the 4 corners of the scalar fields only as needed by c_core

    #ifdef ENABLE_GPTL
    if (do_profile == 1)
    {
        ret = gptlstart('fill2_4corners');
    }
    #endif

    switch(dir)
    {
        case(1): // x-dir
        if (d.sw_corner)
        {
            q1(-1, 0,k) = q1(0, 2,k);
            q1( 0, 0,k) = q1(0, 1,k);
            q2(-1, 0,k) = q2(0, 2,k);
            q2( 0, 0,k) = q2(0, 1,k);
        }
        if ( d.se_corner )
        {
            q1(d.npx+1, 0,k) = q1(d.npx, 2,k);
            q1(d.npx,   0,k) = q1(d.npx, 1,k);
            q2(d.npx+1, 0,k) = q2(d.npx, 2,k);
            q2(d.npx,   0,k) = q2(d.npx, 1,k);
        }
        if ( d.nw_corner )
        {
            q1( 0, d.npy,k) = q1(0, d.npy-1,k);
            q1(-1, d.npy,k) = q1(0, d.npy-2,k);
            q2( 0, d.npy,k) = q2(0, d.npy-1,k);
            q2(-1, d.npy,k) = q2(0, d.npy-2,k);
        }
        if ( d.ne_corner )
        {
            q1(d.npx,   d.npy,k) = q1(d.npx, d.npy-1,k);
            q1(d.npx+1, d.npy,k) = q1(d.npx, d.npy-2,k);
            q2(d.npx,   d.npy,k) = q2(d.npx, d.npy-1,k);
            q2(d.npx+1, d.npy,k) = q2(d.npx, d.npy-2,k);
        }
        break;
        case(2): // y-dir
        if ( d.sw_corner )
        {
            q1(0,  0,k) = q1(1, 0,k);
            q1(0, -1,k) = q1(2, 0,k);
            q2(0,  0,k) = q2(1, 0,k);
            q2(0, -1,k) = q2(2, 0,k);
        }
        if ( d.se_corner )
        {
            q1(d.npx,  0,k) = q1(d.npx-1, 0,k);
            q1(d.npx, -1,k) = q1(d.npx-2, 0,k);
            q2(d.npx,  0,k) = q2(d.npx-1, 0,k);
            q2(d.npx, -1,k) = q2(d.npx-2, 0,k);
        }
        if ( d.nw_corner )
        {
            q1(0, d.npy,k)   = q1(1, d.npy,k);
            q1(0, d.npy+1,k) = q1(2, d.npy,k);
            q2(0, d.npy,k)   = q2(1, d.npy,k);
            q2(0, d.npy+1,k) = q2(2, d.npy,k);
        }
        if ( d.ne_corner )
        {
            q1(d.npx, d.npy,k)   = q1(d.npx-1, d.npy,k);
            q1(d.npx, d.npy+1,k) = q1(d.npx-2, d.npy,k);
            q2(d.npx, d.npy,k)   = q2(d.npx-1, d.npy,k);
            q2(d.npx, d.npy+1,k) = q2(d.npx-2, d.npy,k);
        }
        break;
    }

    #ifdef ENABLE_GPTL
    if (do_profile == 1)
    {
        ret = gptlstop('fill2_4corners');
    }
    #endif

    }

    void d2a2c_vect(const CSWData& d, queue& Q)
    {
        const double zero=static_cast<double>(0.0);

        #ifdef ENABLE_GPTL
        int ret;
        if (do_profile == 1)
        {
            ret = gptlstart ('d2a2c_vect');
        }
        #endif

        //for (int k = 0; k < d.npz; k++) {
        Q.parallel_for(range(d.npz, d.jsize, d.isize), [=](auto idx) {
            // Initialize the non-existing corner regions
            const int k = idx[0];
            const int j = idx[1]+d.jsd;
            const int i = idx[2]+d.isd;
            //for (int j=d.jsd; j<=d.jed; j++)
            //{
            //    for (int i=d.isd; i<=d.ied; i++) 
            //    {
                    d.utmp(i,j,k)=d.big_number;
                    d.vtmp(i,j,k)=d.big_number;
            //    }
            //}
        }).wait();

            //----------
            // Interior:
            //----------
        Q.parallel_for(range(d.npz, std::min(d.npy-4,d.je+1)-std::max(4, d.js-1)+1, std::min(d.npx-4,d.ied)-std::max(4, d.isd)+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+std::max(4, d.js-1);
            const int i = idx[2]+std::max(4, d.isd);
            //for (int j = std::max(4, d.js-1); j<=std::min(d.npy-4,d.je+1); j++)
            //{
            //    for (int i = std::max(4, d.isd); i<=std::min(d.npx-4,d.ied); i++)
            //    {
                    d.utmp(i, j,k) = d.a2 * (d.u(i, j-1,k) + d.u(i, j+2,k)) + d.a1 * (d.u(i, j,k) + d.u(i, j+1,k));
            //    }
            //}
        });
            
        Q.parallel_for(range(d.npz, std::min(d.npy-4,d.jed)-std::max(4, d.jsd)+1, std::min(d.npx-4,d.ie+1)-std::max(4, d.is-1)+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+std::max(4, d.jsd);
            const int i = idx[2]+std::max(4, d.is-1);
            //for (int j = std::max(4, d.jsd); j<=std::min(d.npy-4,d.jed); j++)
            //{
            //    for (int i = std::max(4, d.is-1); i<=std::min(d.npx-4,d.ie+1); i++)
            //    {
                    d.vtmp(i, j,k) = d.a2 * (d.v(i-1, j,k) + d.v(i+2, j,k)) + d.a1 * (d.v(i, j,k) + d.v(i+1, j,k));
            //    }
            //}
        }).wait();

        //----------
        // edges:
        //----------
        if ( d.js == 1 || d.jsd < 4)
        {
            Q.parallel_for(range(d.npz, 3-d.jsd+1, d.ied-d.isd+1), [=](auto idx) {
                const int k = idx[0];
                const int j = idx[1]+d.jsd;
                const int i = idx[2]+d.isd;
                // for (int j = d.jsd; j<= 3; j++)
                // {
                //     for (int i = d.isd; i<= d.ied; i++)
                //     {
                        d.utmp(i,j,k) = 0.5 * (d.u(i,j,k) + d.u(i,j+1,k));
                        d.vtmp(i,j,k) = 0.5 * (d.v(i,j,k) + d.v(i+1,j,k));
                //     }
                // }
            }).wait();
        }
        

        if ( (d.je + 1) == d.npy || d.jed >= (d.npy - 4) )
        {
            Q.parallel_for(range(d.npz, d.jed-(d.npy-3)+1, d.ied-d.isd+1), [=](auto idx) {
                const int k = idx[0];
                const int j = idx[1]+d.npy-3;
                const int i = idx[2]+d.isd;
                // for (int j = d.npy-3; j<= d.jed; j++)
                // {
                //     for (int i = d.isd; i<= d.ied; i++)
                //     {
                        d.utmp(i,j,k) = 0.5 * (d.u(i,j,k) + d.u(i,j+1,k));
                        d.vtmp(i,j,k) = 0.5 * (d.v(i,j,k) + d.v(i+1,j,k));
                //     }
                // }
            }).wait();
        }
        

        if ( d.is == 1 || d.isd < 4 )
        {
            Q.parallel_for(range(d.npz, std::min(d.npy-4, d.jed)-std::max(4, d.jsd)+1, 3-d.isd+1), [=](auto idx) {
                const int k = idx[0];
                const int j = idx[1]+std::max(4, d.jsd);
                const int i = idx[2]+d.isd;
                // for (int j = std::max(4, d.jsd); j<=std::min(d.npy-4, d.jed); j++)
                // {
                //     for (int i = d.isd; i<= 3; i++)
                //     {
                        d.utmp(i,j,k) = 0.5 * (d.u(i,j,k) + d.u(i,j+1,k));
                        d.vtmp(i,j,k) = 0.5 * (d.v(i,j,k) + d.v(i+1,j,k));
                //     }
                // }
            }).wait();
        }

        if ( (d.ie+1) == d.npx || d.ied >= (d.npx-4) )
        {
            Q.parallel_for(range(d.npz, std::min(d.npy-4, d.jed)-std::max(4, d.jsd)+1, d.ied-(d.npx-3)+1), [=](auto idx) {
                const int k = idx[0];
                const int j = idx[1]+std::max(4, d.jsd);
                const int i = idx[2]+d.npx-3;
                // for (int j = std::max(4, d.jsd); j<=std::min(d.npy-4, d.jed); j++)
                // {
                //     for (int i = d.npx-3; i<= d.ied; i++)
                //     {
                        d.utmp(i,j,k) = 0.5 * (d.u(i,j,k) + d.u(i, j+1,k));
                        d.vtmp(i,j,k) = 0.5 * (d.v(i,j,k) + d.v(i+1, j,k));
                //     }
                // }
            }).wait();
        }

        Q.parallel_for(range(d.npz, d.je+2-(d.js-2)+1, d.ie+2-(d.is-2)+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js-2;
            const int i = idx[2]+d.is-2;
            // for (int j = d.js-2; j<= d.je+2; j++)
            // {
            //     for (int i = d.is-2; i<= d.ie+2; i++)
            //     {
                    d.ua(i,j,k) = (d.utmp(i,j,k) - d.vtmp(i,j,k) * d.cosa_s(i,j)) * d.rsin2(i,j);
                    d.va(i,j,k) = (d.vtmp(i,j,k) - d.utmp(i,j,k) * d.cosa_s(i,j)) * d.rsin2(i,j);
            //     }
            // }
        }).wait();


        // A -> C
        //--------------
        // Fix the edges
        //--------------
        // Xdir:
        if ( d.sw_corner )
        {
            Q.parallel_for(d.npz, [=](int k) {
                for (int i = -2; i<= 0; i++)
                {
                    d.utmp(i, 0,k) = -d.vtmp(0, 1-i,k);
                }
            });
        }

        if ( d.se_corner )
        {
            Q.parallel_for(d.npz, [=](int k) {
                for (int i = 0; i<= 2; i++)
                {
                    d.utmp(d.npx+i, 0,k) = d.vtmp(d.npx, i+1,k);
                }
            });
        }
        
        if ( d.ne_corner )
        {
            Q.parallel_for(d.npz, [=](int k) {
                for (int i = 0; i<= 2; i++)
                {
                    d.utmp(d.npx+i, d.npy,k) = -d.vtmp(d.npx, d.je-i,k);
                }
            });
        }

        if ( d.nw_corner )
        {
            Q.parallel_for(d.npz, [=](int k) {
                for (int i = -2; i<= 0; i++)
                {
                    d.utmp(i,d.npy,k) = d.vtmp(0,d.je+i,k);
                }
            });
        }
        Q.wait();

        const int ifirst = std::max(3,     d.is-1);
        const int ilast  = std::min(d.npx-2, d.ie+2);


        //---------------------------------------------
        // 4th order interpolation for interior points:
        //---------------------------------------------
        Q.parallel_for(range(d.npz, d.je+1-(d.js-1)+1, ilast-ifirst+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js-1;
            const int i = idx[2]+ifirst;
            // for (int j = d.js-1; j<= d.je+1; j++)
            // {
            //     for (int i = ifirst; i<= ilast; i++)
            //     {
                    d.uc(i, j,k) = d.a1 * (d.utmp(i-1, j,k) + d.utmp(i, j,k)) + d.a2 * (d.utmp(i-2, j,k) + d.utmp(i+1, j,k));
                    d.ut(i, j,k) = (d.uc(i, j,k) - d.v(i, j,k) * d.cosa_u(i, j)) * d.rsin_u(i, j);
            //     }
            // }
        });

        // Xdir:
        Q.parallel_for(d.npz, [=](int k) {
            if ( d.sw_corner )
            {
                d.ua(-1, 0,k) = -d.va(0, 2,k);
                d.ua( 0, 0,k) = -d.va(0, 1,k);
            }

            if ( d.se_corner )
            {
                d.ua(d.npx,   0,k) = d.va(d.npx, 1,k);
                d.ua(d.npx+1, 0,k) = d.va(d.npx, 2,k);
            }
            
            if ( d.ne_corner )
            {
                d.ua(d.npx,   d.npy,k) = -d.va(d.npx, d.npy-1,k);
                d.ua(d.npx+1, d.npy,k) = -d.va(d.npx, d.npy-2,k);
            }

            if ( d.nw_corner )
            {
                d.ua(-1, d.npy,k) = d.va(0, d.npy-2,k);
                d.ua( 0, d.npy,k) = d.va(0, d.npy-1,k);
            }
        });


        if ( d.is == 1 )
        {
            Q.parallel_for(range(d.npz,d.je+1-(d.js-1)+1), [=](auto idx) {
                const int k = idx[0];
                const int j = idx[1]+d.js-1;
                //for (int j = d.js-1; j<= d.je+1; j++)
                //{
                    d.uc(0, j,k) = d.c1 * d.utmp(-2, j,k) + d.c2 * d.utmp(-1, j,k) + d.c3 * d.utmp(0, j,k);

                    double ua_interp[4];
                    double dxa_interp[4];
                    for (int iter=0;iter<=3;iter++)
                    {
                        ua_interp[iter] = d.ua(iter-1,j,k);
                        dxa_interp[iter] = d.dxa(iter-1,j);
                    }
                    d.ut(1,j,k)=edge_interpolate4(ua_interp, dxa_interp);
                    //Want to use the UPSTREAM value
                    if (d.ut(1, j,k) < zero)
                        d.uc(1, j,k) = d.ut(1, j,k) * d.sin_sg(0, j, 3);
                    else
                        d.uc(1, j,k) = d.ut(1, j,k) * d.sin_sg(1, j, 1);

                    d.uc(2, j,k) = d.c1 * d.utmp(3, j,k) + d.c2 * d.utmp(2, j,k) + d.c3 * d.utmp(1, j,k);
                    d.ut(0, j,k) = (d.uc(0, j,k) - d.v(0, j,k) * d.cosa_u(0, j)) * d.rsin_u(0, j);
                    d.ut(2, j,k) = (d.uc(2, j,k) - d.v(2, j,k) * d.cosa_u(2, j)) * d.rsin_u(2, j);
                //}
            }).wait();

            if ( (d.ie+1) == d.npx )
            {
                Q.parallel_for(range(d.npz,d.je+1-(d.js-1)+1), [=](auto idx) {
                    const int k = idx[0];
                    const int j = idx[1]+d.js-1;
                    //for (int j = d.js-1; j<= d.je+1; j++)
                    //{
                        d.uc(d.npx-1, j,k) = d.c1 * d.utmp(d.npx-3, j,k) + d.c2 * d.utmp(d.npx-2, j,k) + d.c3 * d.utmp(d.npx-1, j,k);
                        int i = d.npx;
                        d.ut(i,j,k) = 0.25 * (-d.ua(i-2,j,k) + 3. * (d.ua(i-1,j,k) + d.ua(i,j,k)) - d.ua(i+1,j,k));
                        //ut(i, j) = edge_interpolate4(ua(i-2:i+1, j), dxa(i-2:i+1, j))
                        double ua_interp[4];
                        double dxa_interp[4];
                        for (int iter=0;iter<=3;iter++)
                        {
                            ua_interp[iter]=d.ua(i-2+iter,j,k);
                            dxa_interp[iter]=d.dxa(i-2+iter,j);
                        }

                        d.ut(i,j,k)=edge_interpolate4(ua_interp, dxa_interp);

                        if ( d.ut(i,j,k) < zero )
                            d.uc(i,j,k) = d.ut(i,j,k) * d.sin_sg(i-1,j,3);
                        else
                            d.uc(i,j,k) = d.ut(i,j,k) * d.sin_sg(i, j, 1);

                        d.uc(d.npx+1, j,k) = d.c3 * d.utmp(d.npx, j,k) + d.c2 * d.utmp(d.npx+1, j,k) + d.c1 * d.utmp(d.npx+2, j,k);
                        d.ut(d.npx-1, j,k) = (d.uc(d.npx-1, j,k) - d.v(d.npx-1, j,k) * d.cosa_u(d.npx-1, j)) * d.rsin_u(d.npx-1, j);
                        d.ut(d.npx+1, j,k) = (d.uc(d.npx+1, j,k) - d.v(d.npx+1, j,k) * d.cosa_u(d.npx+1, j)) * d.rsin_u(d.npx+1, j);
                    //}
                }).wait();
            }
        }

        //------
        // Ydir:
        //------
        if ( d.sw_corner )
        {
            Q.parallel_for(d.npz, [=](int k) {
                for (int j = -2; j<= 0; j++)
                {
                    d.vtmp(0, j,k) = -d.utmp(1-j, 0,k);
                }
            });
        }

        if ( d.nw_corner )
        {
            Q.parallel_for(d.npz, [=](int k) {
                for (int j = 0; j<= 2; j++)
                {
                    d.vtmp(0, d.npy+j,k) = d.utmp(j+1, d.npy,k);
                }
            });
        }

        if ( d.se_corner )
        {
            Q.parallel_for(d.npz, [=](int k) {
                for (int j = -2; j<= 0; j++)
                {
                    d.vtmp(d.npx, j,k) = d.utmp(d.ie+j, 0,k);
                }
            });
        }

        if ( d.ne_corner )
        {
            Q.parallel_for(d.npz, [=](int k) {
                for (int j = 0; j<= 2; j++)
                {
                    d.vtmp(d.npx, d.npy+j,k) = -d.utmp(d.ie-j, d.npy,k);
                }
            });
        }

        Q.parallel_for(d.npz, [=](int k) {
            if ( d.sw_corner )
            {
                d.va(0, -1,k) = -d.ua(2, 0,k);
                d.va(0,  0,k) = -d.ua(1, 0,k);
            }

            if ( d.se_corner )
            {
                d.va(d.npx,  0,k) = d.ua(d.npx-1, 0,k);
                d.va(d.npx, -1,k) = d.ua(d.npx-2, 0,k);
            }

            if ( d.ne_corner )
            {
                d.va(d.npx, d.npy  ,k) = -d.ua(d.npx-1, d.npy,k);
                d.va(d.npx, d.npy+1,k) = -d.ua(d.npx-2, d.npy,k);
            }

            if ( d.nw_corner )
            {
                d.va(0, d.npy,k)   = d.ua(1, d.npy,k);
                d.va(0, d.npy+1,k) = d.ua(2, d.npy,k);
            }
        }).wait();
        //}


        //for (k = 0; k < d.npz; k++)
        //{
        //    for (j = d.js-1; j<= d.je+2; j++)
        //    {
        Q.parallel_for(range(d.npz, d.je+2-(d.js-1)+1, d.ie+1-(d.is-1)+1), [=](auto idx) {
            const int j = idx[1] + d.js-1;
            const int k = idx[0];
            const int i = idx[2]+d.is-1;
            if ( j == 1 )
            {
                //for (int i = d.is-1; i<= d.ie+1; i++)
                //{
                    //vt(i, j) = edge_interpolate4(va(i, -1:2), dya(i, -1:2))
                    double va_interp[4];
                    double dya_interp[4];
                    for (int iter=0;iter<=3;iter++)
                    {
                        va_interp[iter]=d.va(i,iter-1,k);
                        dya_interp[iter]=d.dya(i,iter-1);
                    }
                    d.vt(i,j,k)=edge_interpolate4(va_interp, dya_interp);

                    if ( d.vt(i,j,k) < 0. )
                        d.vc(i,j,k) = d.vt(i,j,k) * d.sin_sg(i, j-1, 4);
                    else
                        d.vc(i,j,k) = d.vt(i,j,k) * d.sin_sg(i, j, 2);
                //}
            } 
            else if ( j == 0 || j == (d.npy-1) )
            {
                //for (int i = d.is-1; i<= d.ie+1; i++)
                //{
                    d.vc(i,j,k) = d.c1 * d.vtmp(i, j-2,k) + d.c2 * d.vtmp(i, j-1,k) + d.c3 * d.vtmp(i,j,k);
                    d.vt(i,j,k) = (d.vc(i,j,k) - d.u(i,j,k) * d.cosa_v(i,j)) * d.rsin_v(i,j);
                //}
            } 
            else if ( j == 2 || j == (d.npy+1) )
            {
                //for (int i = d.is-1; i<= d.ie+1; i++)
                //{
                    d.vc(i,j,k) = d.c1 * d.vtmp(i, j+1,k) + d.c2 * d.vtmp(i, j,k) + d.c3 * d.vtmp(i, j-1,k);
                    d.vt(i,j,k) = (d.vc(i,j,k) - d.u(i,j,k) * d.cosa_v(i, j)) * d.rsin_v(i, j);
                //}
            } 
            else if ( j == d.npy )
            {
                //for (int i = d.is-1; i<= d.ie+1; i++)
                //{
                    d.vt(i,j,k) = 0.25 * (-d.va(i, j-2,k) + 3. * (d.va(i, j-1,k) + d.va(i, j,k)) - d.va(i, j+1,k));
                    //vt(i, j) = edge_interpolate4(va(i, j-2:j+1), dya(i, j-2:j+1))
                    double va_interp[4];
                    double dya_interp[4];
                    for (int iter=0; iter<=3; iter++)
                    {
                        va_interp[iter]=d.va(i,j-2+iter,k);
                        dya_interp[iter]=d.dya(i,j-2+iter);
                    }
                    d.vt(i,j,k)=edge_interpolate4(va_interp, dya_interp);

                    if ( d.vt(i,j,k) < 0. )
                        d.vc(i,j,k) = d.vt(i,j,k) * d.sin_sg(i, j-1, 4);
                    else
                        d.vc(i,j,k) = d.vt(i,j,k) * d.sin_sg(i, j, 2);
                //}
            } 
            else
            {
                // 4th order interpolation for interior points:
                //for (int i = d.is-1; i<= d.ie+1; i++)
                //{
                    d.vc(i,j,k) = d.a2 * (d.vtmp(i, j-2,k) + d.vtmp(i, j+1,k)) + d.a1 * (d.vtmp(i, j-1,k) + d.vtmp(i, j,k));
                    d.vt(i,j,k) = (d.vc(i, j,k) - d.u(i, j,k) * d.cosa_v(i, j)) * d.rsin_v(i, j);
                //}
            }
        }).wait();
        //}
        #ifdef ENABLE_GPTL
        if (do_profile == 1)
        {
            ret = gptlstop ('d2a2c_vect');
        }
        #endif
    }

    void divergence_corner(const CSWData& d, queue& Q)
    {  
        const int is2 = std::max(2, d.is);
        const int ie1 = std::min(d.npx-1, d.ie+1);
        
        #ifdef ENABLE_GPTL
        if (do_profile == 1)
        {
            int ret = gptlstart('divergence_corner');
        }
        #endif


        //     9---4---8
        //     |       |
        //     1   5   3
        //     |       |
        //     6---2---7
        
        //for (int k = 0; k < d.npz; k++) {
        Q.parallel_for(range(d.npz, d.je+1-d.js+1, d.ie+1-(d.is-1)+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js;
            const int i = idx[2]+d.is-1;
            //for (int j=d.js; j<=d.je+1; j++)
            //{
                if ( j == 1 || j == d.npy )
                {
                    //for (int i = d.is-1; i<=d.ie+1; i++)
                    //{
                        d.uf(i,j,k) = d.u(i,j,k) * d.dyc(i,j) * 0.5 * (d.sin_sg(i, j-1, 4) + d.sin_sg(i, j, 2));
                    //}
                } 
                else
                {
                    //for (int i = d.is-1; i<=d.ie+1; i++)
                    //{
                        d.uf(i,j,k) = (d.u(i,j,k) - 0.25 * (d.va(i,j-1,k) + d.va(i, j,k)) *
                                (d.cos_sg(i, j-1, 4) + d.cos_sg(i, j, 2))) *
                                d.dyc(i, j) * 0.5 * (d.sin_sg(i, j-1, 4) + d.sin_sg(i, j, 2));
                    //}
                }
            //}
        });

        Q.parallel_for(range(d.npz, d.je+1-(d.js-1)+1, ie1-is2+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js-1;
            const int i = idx[2]+is2;
            //for (int j = d.js-1; j<=d.je+1; j++)
            //{
                //for (int i = is2; i<=ie1; i++)
                //{
                    d.vf(i,j,k) = (d.v(i,j,k) - 0.25 * (d.ua(i-1,j,k) + d.ua(i,j,k)) *
                            (d.cos_sg(i-1, j, 3) + d.cos_sg(i, j, 1))) *
                                d.dxc(i,j) * 0.5 * (d.sin_sg(i-1, j, 3) + d.sin_sg(i, j, 1));
                //}
        });

         Q.parallel_for(range(d.npz, d.je+1-(d.js-1)+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js-1;

                if (d.is==1)
                    d.vf(1, j,k) = d.v(1, j,k) * d.dxc(1, j) * 0.5 * (d.sin_sg(0, j, 3) + d.sin_sg(1, j, 1));

                if ( (d.ie+1) == d.npx )
                    d.vf(d.npx,j,k) = d.v(d.npx,j,k) * d.dxc(d.npx, j) * 0.5 * (d.sin_sg(d.npx-1, j, 3) + d.sin_sg(d.npx, j, 1));
                
            //}
        }).wait();
            

        Q.parallel_for(range(d.npz, d.je+1-d.js+1, d.ie+1-d.is+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js;
            const int i = idx[2]+d.is;
            //for (int j = d.js; j<=d.je+1; j++)
            //{
                //for (int i = d.is; i<=d.ie+1; i++)
                //{
                    d.divg_d(i, j,k) = d.vf(i, j-1,k) - d.vf(i, j,k) + d.uf(i-1, j,k) - d.uf(i, j,k);
                //}
            //}
        }).wait();

            
            // Remove the extra term at the corners:
        Q.parallel_for(d.npz, [=](int k) {

            if (d.sw_corner)
            {
                d.divg_d(1,     1,k) = d.divg_d(1,     1,k) - d.vf(1,     0,k);
            }
            if (d.se_corner)
            {
                d.divg_d(d.npx,   1,k) = d.divg_d(d.npx,   1,k) - d.vf(d.npx,   0,k);
            }
            if (d.ne_corner)
            {
                d.divg_d(d.npx, d.npy,k) = d.divg_d(d.npx, d.npy,k) + d.vf(d.npx, d.npy,k);
            }
            if (d.nw_corner)
            {
                d.divg_d(1,   d.npy,k) = d.divg_d(1,   d.npy,k) + d.vf(1,   d.npy,k);
            }
        }).wait();
            
        Q.parallel_for(range(d.npz, d.je+1-d.js+1, d.ie+1-d.is+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js;
            const int i = idx[2]+d.is;
            //for (int j=d.js; j<=d.je+1; j++)
            //{
                //for (int i=d.is; i<=d.ie+1; i++)
                //{
                    d.divg_d(i,j,k) = d.rarea_c(i, j) * d.divg_d(i, j,k);
                //}
            //}
        //}
        }).wait();


        #ifdef ENABLE_GPTL
        if (do_profile == 1)
        {
            ret = gptlstop('divergence_corner');
        }
        #endif
    }

    void c_sw(const CSWData& d, queue& Q)
    {
        const double dt4 = 0.5 * d.dt2;
        const int iep1 = d.ie+1;
        const int jep1 = d.je+1;

        #ifdef ENABLE_GPTL
        int ret;
        if (do_profile == 1)
        {
            ret = gptlstart('c_sw');
        }
        #endif

        
        d2a2c_vect(d, Q);

        if (d.nord > 0)
            divergence_corner(d, Q);
        
        Q.parallel_for(range(d.npz, jep1-(d.js-1)+1, iep1+1-(d.is-1)+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js-1;
            const int i = idx[2]+d.is-1;
            //for (int j=d.js-1; j<=jep1; j++)
            //{
              //for (int i=d.is-1; i<=iep1+1; i++)
              //{
                if (d.ut(i,j,k)>0.0)
                  d.ut(i,j,k)=d.dt2*d.ut(i,j,k)*d.dy(i,j)*d.sin_sg(i-1,j,3);
                else
                  d.ut(i,j,k)=d.dt2*d.ut(i,j,k)*d.dy(i,j)*d.sin_sg(i,j,1);
              //}
            //}
        });

        //for (int k = 0; k < d.npz; k++) {
        Q.parallel_for(range(d.npz, d.je+2-(d.js-1)+1, iep1-(d.is-1)+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js-1;
            const int i = idx[2]+d.is-1;
            //for (int j=d.js-1; j<=d.je+2; j++)
            //{
                //for (int i=d.is-1; i<=iep1; i++)
                //{
                    if (d.vt(i,j,k)>0.0)
                        d.vt(i,j,k) = d.dt2 * d.vt(i,j,k) * d.dx(i, j) * d.sin_sg(i, j-1, 4);
                    else
                        d.vt(i,j,k) = d.dt2 * d.vt(i,j,k) * d.dx(i, j) * d.sin_sg(i, j, 2);
                //}
            //}
        });

            //----------------
            // Transport delp:
            //----------------
            // Xdir:
        Q.parallel_for(d.npz, [=](int k) {
            fill2_4corners(k, d.delp, d.pt, 1, d);
        });

        Q.parallel_for(d.npz, [=](int k) {
            fill_4corners(k, d.w, 1, d);
        }).wait();

        Q.parallel_for(range(d.npz, d.je+1-(d.js-1)+1, d.ie+2-(d.is-1)+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js-1;
            const int i = idx[2]+d.is-1;
            //for (int j=d.js-1; j<=d.je+1; j++)
            //{
                //for (int i=d.is-1; i<=d.ie+2; i++)
                //{
                    if (d.ut(i,j,k)>0.0)
                    {
                        d.fx1(i,j,k) = d.delp(i-1,j,k);
                        d.fx(i,j,k) = d.pt(i-1,j,k);
                        d.fx2(i,j,k) = d.w(i-1,j,k);
                    } 
                    else
                    {
                        d.fx1(i,j,k) = d.delp(i,j,k);
                        d.fx(i,j,k) = d.pt(i,j,k);
                        d.fx2(i,j,k) = d.w(i,j,k);
                    }
                    d.fx1(i,j,k) = d.ut(i,j,k) * d.fx1(i,j,k);
                    d.fx(i,j,k) = d.fx1(i,j,k) * d.fx(i,j,k);
                    d.fx2(i,j,k) = d.fx1(i,j,k) * d.fx2(i,j,k);
                //}
            //}
        }).wait();

        // Ydir:
        Q.parallel_for(d.npz, [=](int k) {
            fill2_4corners(k, d.delp, d.pt, 2, d);
        });
        Q.parallel_for(d.npz, [=](int k) {
            fill_4corners(k, d.w, 2, d);
        }).wait();

        Q.parallel_for(range(d.npz, d.je+2-(d.js-1)+1, d.ie+1-(d.is-1)+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js-1;
            const int i = idx[2]+d.is-1;
            //for (int j=d.js-1; j<=d.je+2; j++)
            //{
                //for (int i=d.is-1; i<=d.ie+1; i++)
                //{
                    if (d.vt(i,j,k)>0)
                    {
                        d.fy1(i,j,k) = d.delp(i,j-1,k);
                        d.fy(i,j,k) = d.pt(i,j-1,k);
                        d.fy2(i,j,k) = d.w(i,j-1,k);
                    } 
                    else
                    {
                        d.fy1(i,j,k) = d.delp(i,j,k);
                        d.fy(i,j,k) = d.pt(i,j,k);
                        d.fy2(i,j,k) = d.w(i,j,k);
                    }
                    d.fy1(i,j,k) =  d.vt(i,j,k) * d.fy1(i,j,k);
                    d.fy(i,j,k) = d.fy1(i,j,k) * d.fy(i,j,k);
                    d.fy2(i,j,k) = d.fy1(i,j,k) * d.fy2(i,j,k);
                //}
            //}
        }).wait();  
            
        Q.parallel_for(range(d.npz, jep1-(d.js-1)+1, iep1-(d.is-1)+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js-1;
            const int i = idx[2]+d.is-1;
            //for (int j=d.js-1; j<=d.je+1; j++)
            //{
                //for (int i=d.is-1; i<=d.ie+1; i++)
                //{
                    d.delpc(i,j,k) = d.delp(i,j,k) + 
                        (d.fx1(i,j,k) - d.fx1(i+1,j,k) + d.fy1(i,j,k) - d.fy1(i,j+1,k)) * d.rarea(i,j);
                    d.ptc(i,j,k) = (d.pt(i,j,k) * d.delp(i,j,k) + 
                        (d.fx(i,j,k) - d.fx(i+1,j,k) + d.fy(i,j,k) - d.fy(i,j+1,k)) * d.rarea(i,j)) / d.delpc(i,j,k);
                    d.wc(i,j,k) = (d.w(i,j,k) * d.delp(i,j,k) + 
                        (d.fx2(i,j,k) - d.fx2(i+1,j,k) + d.fy2(i,j,k) - d.fy2(i,j+1,k)) * d.rarea(i,j)) / d.delpc(i,j,k);
                //}
            //}
        });

            //------------
            // Compute KE:
            //------------

            // Since uc = u*, i.e. the covariant wind perpendicular to the face edge, if
            // we want to compute kinetic energy we will need the true coordinate-parallel
            // covariant wind, computed through u = uc*sina + v*cosa.
            //
            // Use the alpha for the cell KE is being computed in.

        Q.parallel_for(range(d.npz, jep1-(d.js-1)+1, iep1-(d.is-1)+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js-1;
            const int i = idx[2]+d.is-1;
            //for (int j=d.js-1; j<= jep1; j++)
            //{
                //for (int i=d.is-1; i<= iep1; i++)
                //{
                    if ( d.ua(i,j,k) > 0. )
                    {
                        if (i==1)
                            d.ke(1,j,k)=d.uc(1,j,k)*d.sin_sg(1,j,1) + d.v(1,j,k)*d.cos_sg(1,j,1);
                        else if (i==d.npx)
                            d.ke(i,j,k)=d.uc(d.npx,j,k)*d.sin_sg(d.npx,j,1) + d.v(d.npx,j,k)*d.cos_sg(d.npx,j,1);
                        else
                            d.ke(i,j,k) = d.uc(i,j,k);
                    } 
                    else
                    {
                        if (i==0)
                            d.ke(0,j,k)=d.uc(1,j,k)*d.sin_sg(0,j,3) + d.v(1,j,k)*d.cos_sg(0,j,3);
                        else if (i==(d.npx-1))
                            d.ke(i,j,k)=d.uc(d.npx,j,k)*d.sin_sg(d.npx-1,j,3) + d.v(d.npx,j,k)*d.cos_sg(d.npx-1,j,3);
                        else
                            d.ke(i,j,k)=d.uc(i+1,j,k);
                    }
                //}
            //}
            
            //for (int j=d.js-1; j<= jep1; j++)
            //{
                //for (int i=d.is-1; i<= iep1; i++)
                //{
                    if ( d.va(i,j,k) > 0. )
                    {
                        if (j==1)
                            d.vort(i,1,k)=d.vc(i,1,k)*d.sin_sg(i,1,2) + d.u(i,1,k)*d.cos_sg(i,1,2);
                        else if (j==d.npy)
                            d.vort(i,j,k)=d.vc(i,d.npy,k)*d.sin_sg(i,d.npy,2) + d.u(i,d.npy,k)*d.cos_sg(i,d.npy,2);
                        else
                            d.vort(i,j,k) = d.vc(i,j,k);
                    } 
                    else
                    {
                        if ( j == 0 )
                            d.vort(i,0,k) = d.vc(i,1,k) * d.sin_sg(i, 0, 4) + d.u(i,1,k) * d.cos_sg(i, 0, 4);
                        else if ( j == (d.npy - 1) )
                            d.vort(i,j,k) = d.vc(i,d.npy,k) * d.sin_sg(i, d.npy-1, 4) + d.u(i,d.npy,k) * d.cos_sg(i, d.npy-1, 4);
                        else
                            d.vort(i,j,k) = d.vc(i,j+1,k);
                    }
                //}
            //}

            //for (int j=d.js-1; j<= jep1; j++)
            //{
                //for (int i=d.is-1; i<= iep1; i++)
                //{
                    d.ke(i,j,k) = dt4 * (d.ua(i,j,k) * d.ke(i,j,k) + d.va(i,j,k) * d.vort(i,j,k));
                //}
            //}
        }).wait();

            //------------------------------
            // Compute circulation on C grid
            //------------------------------
            // To consider using true co-variant winds at face edges?
        Q.parallel_for(range(d.npz, jep1-(d.js-1)+1, d.ie+1-d.is+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js-1;
            const int i = idx[2]+d.is;
            //for (int j=d.js-1; j<= d.je+1; j++)
            //{
                //for (int i=d.is; i<= d.ie+1; i++)
                //{
                    d.fx(i,j,k) = d.uc(i,j,k) * d.dxc(i,j);
                //}
            //}
        });

        Q.parallel_for(range(d.npz, jep1-(d.js)+1, d.ie+1-(d.is-1)+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js;
            const int i = idx[2]+d.is-1;
            //for (int j=d.js; j<= d.je+1; j++)
            //{
                //for (int i=d.is-1; i<= d.ie+1; i++)
                //{
                    d.fy(i,j,k) = d.vc(i,j,k) * d.dyc(i,j);
                //}
            //}
        }).wait();

        Q.parallel_for(range(d.npz, jep1-(d.js)+1, d.ie+1-d.is+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js;
            const int i = idx[2]+d.is;
            //for (int j=d.js; j<= d.je+1; j++)
            //{
                //for (int i=d.is; i<= d.ie+1; i++)
                //{
                    d.vort(i,j,k) =  d.fx(i,j-1,k) - d.fx(i,j,k) - d.fy(i-1,j,k) + d.fy(i,j,k);
                //}
            //}
        }).wait();

            // Remove the extra term at the corners:
        Q.parallel_for(d.npz, [=](int k) {
            if ( d.sw_corner )
                d.vort(1,     1,k) = d.vort(1,     1,k) + d.fy(0,     1,k);

            if ( d.se_corner )
                d.vort(d.npx,   1,k) = d.vort(d.npx,   1,k) - d.fy(d.npx,   1,k);

            if ( d.ne_corner )
                d.vort(d.npx, d.npy,k) = d.vort(d.npx, d.npy,k) - d.fy(d.npx, d.npy,k);

            if ( d.nw_corner )
                d.vort(1,   d.npy,k) = d.vort(1,   d.npy,k) + d.fy(0,   d.npy,k);
        }).wait();


            //----------------------------
            // Compute absolute vorticity
            //----------------------------
        Q.parallel_for(range(d.npz, jep1-(d.js)+1, d.ie+1-d.is+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js;
            const int i = idx[2]+d.is;
            //for (int j=d.js; j<= d.je+1; j++)
            //{
                //for (int i=d.is; i<= d.ie+1; i++)
                //{
                    d.vort(i,j,k) = d.fC(i,j) + d.rarea_c(i, j) * d.vort(i,j,k);
                //}
            //}
        }).wait();

            //----------------------------------
            // Transport absolute vorticity:
            //----------------------------------
            // To go from v to contravariant v at the edges, we divide by sin_sg;
            // but we then must multiply by sin_sg to get the proper flux.
            // These cancel, leaving us with fy1 = dt2*v at the edges.
            // (For the same reason we only divide by sin instead of sin**2 in the interior)

        Q.parallel_for(range(d.npz, d.je-(d.js)+1, iep1-d.is+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js;
            const int i = idx[2]+d.is;
            //for (int j=d.js; j<=d.je; j++)
            //{
            //    #pragma omp simd
                //for (int i=d.is; i<=iep1; i++)
                //{
                    if (i == 1 || i == d.npx )
                        d.fy1(i,j,k) = d.dt2 * d.v(i,j,k);
                    else
                        d.fy1(i,j,k) = d.dt2 * (d.v(i,j,k) - d.uc(i,j,k) * d.cosa_u(i,j)) / d.sina_u(i,j);

                    if ( d.fy1(i,j,k) > 0.0 )
                        d.fy(i,j,k) = d.vort(i,j,k);
                    else
                        d.fy(i,j,k) = d.vort(i,j+1,k);
                //}
            //}
        });

        Q.parallel_for(range(d.npz, jep1-(d.js)+1, d.ie-d.is+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js;
            const int i = idx[2]+d.is;
            //for (int j=d.js; j<=jep1; j++)
            //{
                if (j==1 || j==d.npy )
                {
                    //#pragma omp simd
                    //for (int i=d.is; i<=d.ie; i++)
                    //{
                        d.fx1(i,j,k) = d.dt2 * d.u(i,j,k);
                        if ( d.fx1(i,j,k) > 0.0 )
                            d.fx(i,j,k) = d.vort(i,j,k);
                        else
                            d.fx(i,j,k) = d.vort(i+1,j,k);
                    //}
                } 
                else
                {
                    //#pragma omp simd
                    //for (int i=d.is; i<=d.ie; i++)
                    //{
                        d.fx1(i,j,k) = d.dt2 * (d.u(i,j,k) - d.vc(i,j,k) * d.cosa_v(i,j)) / d.sina_v(i,j);
                        if ( d.fx1(i,j,k) > 0.0 )
                            d.fx(i,j,k) = d.vort(i,j,k);
                        else
                            d.fx(i,j,k) = d.vort(i+1,j,k);
                    //}
                }
            //}
        }).wait();

            // Update time-centered winds on the C-Grid
        Q.parallel_for(range(d.npz, d.je-(d.js)+1, iep1-d.is+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js;
            const int i = idx[2]+d.is;
            //for (int j=d.js; j<= d.je; j++)
            //{
                //for (int i=d.is; i<= iep1; i++)
                //{
                    d.uc(i,j,k) = d.uc(i,j,k) + d.fy1(i,j,k) * d.fy(i,j,k) + d.rdxc(i,j) * (d.ke(i-1,j,k) - d.ke(i,j,k));
                //}
            //}
        });
        Q.parallel_for(range(d.npz, jep1-(d.js)+1, d.ie-d.is+1), [=](auto idx) {
            const int k = idx[0];
            const int j = idx[1]+d.js;
            const int i = idx[2]+d.is;
            //for (int j=d.js; j<= jep1; j++)
            //{
                //for (int i=d.is; i<= d.ie; i++)
                //{
                    d.vc(i,j,k) = d.vc(i,j,k) - d.fx1(i,j,k) * d.fx(i,j,k) + d.rdyc(i,j) * (d.ke(i,j-1,k) - d.ke(i,j,k));
                //}
            //}
        }).wait();
        //}

        #ifdef ENABLE_GPTL
        if (do_profile == 1)
        {
            ret = gptlstop('c_sw');
        }
        #endif
    }

}


#endif /*CSWMIXED_HPP*/
