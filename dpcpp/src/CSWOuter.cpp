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

#include "CSWOuter.hpp"

using namespace sycl;


CSWOuter::CSWOuter(const int isd_in, const int ied_in, const int jsd_in, const int jed_in, 
        const int is_in, const int ie_in, const int js_in, const int je_in, const int nord_in, const int npx_in, 
        const int npy_in, const int npz_in, const int do_profile_in, double dt2_in, 
        const bool sw_corner_in, const bool se_corner_in, const bool nw_corner_in, const bool ne_corner_in,
        double *prarea, double *prarea_c,
        double *psin_sg, double *pcos_sg, double *psina_v,
        double *pcosa_v,  double *psina_u,  double *pcosa_u,
        double *pfC,  double *prdxc,  double *prdyc,  double *pdx,
        double *pdy,  double *pdxc,  double *pdyc,  double *pcosa_s,
        double *prsin_u,  double *prsin_v,  double *prsin2,
        double *pdxa,  double *pdya) :  
    isd(isd_in), ied(ied_in), jsd(jsd_in), jed(jed_in),
    is(is_in), ie(ie_in), js(js_in), je(je_in), nord(nord_in), npx(npx_in), 
    npy(npy_in), npz(npz_in), do_profile(do_profile_in), dt2(dt2_in),
    sw_corner(sw_corner_in), se_corner(se_corner_in), nw_corner(nw_corner_in), ne_corner(ne_corner_in),
    isize(ied-isd+1), jsize(jed-jsd+1), ksize(9), isizep1(isize+1), jsizep1(jsize+1),
    rarea(isd, ied, jsd, jed, prarea),
    rarea_c(isd, ied+1, jsd, jed+1, prarea_c),
    sin_sg(isd, ied, jsd, jed, 1, 9, psin_sg),
    cos_sg(isd, ied, jsd, jed, 1, 9, pcos_sg),
    sina_v(isd, ied,  jsd, jed+1, psina_v),
    cosa_v(isd, ied,  jsd, jed+1, pcosa_v),
    sina_u(isd, ied+1, jsd, jed, psina_u),
    cosa_u(isd, ied+1, jsd, jed, pcosa_u),
    fC(isd, ied+1, jsd, jed+1, pfC),
    rdxc(isd, ied+1, jsd, jed, prdxc),
    rdyc(isd, ied,  jsd, jed+1, prdyc),
    dx(isd, ied,  jsd, jed+1, pdx),
    dy(isd, ied+1, jsd, jed, pdy),
    dxc(isd, ied+1, jsd, jed, pdxc),
    dyc(isd, ied,  jsd, jed+1, pdyc),
    cosa_s(isd, ied,  jsd, jed, pcosa_s),
    rsin_u(isd, ied+1, jsd, jed, prsin_u),
    rsin_v(isd, ied,  jsd, jed+1, prsin_v),
    rsin2(isd, ied,  jsd, jed, prsin2),
    dxa(isd, ied,  jsd, jed, pdxa),
    dya(isd, ied,  jsd, jed, pdya)
{}

CSWOuter::CSWOuter(const int isd_in, const int ied_in, const int jsd_in, const int jed_in, 
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
    const Offset2DArray<double, true>& dya_in) :
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
    dya(dya_in)
{}

void CSWOuter::c_sw(Offset2DArray<double, true>& delpc, Offset2DArray<double, true>& delp, 
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
        Offset2DArray<double, true>& uf, Offset2DArray<double, true>& vf) const
{
  const double dt4 = 0.5 * dt2;
  int i, j;
  const int iep1 = ie+1;
  const int jep1 = je+1;

  // double *vort  = malloc_device<double>(is-1, ie+1, js-1, je+1);
  // double *ke    = malloc_device<double>(is-1, ie+1, js-1, je+1);
  // double *fx    = malloc_device<double>(is-1, ie+2, js-1, je+1);
  // double *fx1   = malloc_device<double>(is-1, ie+2, js-1, je+1);
  // double *fx2   = malloc_device<double>(is-1, ie+2, js-1, je+1);
  // double *fy    = malloc_device<double>(is-1, ie+1, js-1, je+2);
  // double *fy1   = malloc_device<double>(is-1, ie+1, js-1, je+2);
  // double *fy2   = malloc_device<double>(is-1, ie+1, js-1, je+2);

#ifdef ENABLE_GPTL
  int ret;
  if (do_profile == 1)
  {
    ret = gptlstart('c_sw');
  }
#endif

  d2a2c_vect(u, v, ua, va, uc, vc, ut, vt, utmp, vtmp);


  if (nord > 0)
  {
    divergence_corner(u, v, ua, va, divg_d, uf, vf);
  }
  
  for (j=js-1; j<=jep1; j++)
  {
    for (i=is-1; i<=iep1+1; i++)
    {
      if (ut(i,j)>0.0)
        ut(i,j)=dt2*ut(i,j)*dy(i,j)*sin_sg(i-1,j,3);
      else
        ut(i,j)=dt2*ut(i,j)*dy(i,j)*sin_sg(i,j,1);
    }
  }  


  for (j=js-1; j<=je+2; j++)
  {
    for (i=is-1; i<=iep1; i++)
    {
      if (vt(i,j)>0.0)
        vt(i,j) = dt2 * vt(i,j) * dx(i, j) * sin_sg(i, j-1, 4);
      else
        vt(i,j) = dt2 * vt(i,j) * dx(i, j) * sin_sg(i, j, 2);
    }
  }

  //----------------
  // Transport delp:
  //----------------
  // Xdir:
  fill2_4corners(delp, pt, 1);

  fill_4corners(w, 1);

  for (j=js-1; j<=je+1; j++)
  {
    for (i=is-1; i<=ie+2; i++)
    {
      if (ut(i,j)>0.0)
      {
        fx1(i, j) = delp(i-1, j);
        fx(i, j) = pt(i-1, j);
        fx2(i, j) = w(i-1, j);
      } 
      else
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
  fill2_4corners(delp, pt, 2);
  fill_4corners(w, 2);
  for (j=js-1; j<=je+2; j++)
  {
    for (i=is-1; i<=ie+1; i++)
    {
      if (vt(i,j)>0)
      {
        fy1(i, j) = delp(i, j-1);
        fy(i, j) = pt(i, j-1);
        fy2(i, j) = w(i, j-1);
      } 
      else
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

  for (j=js-1; j<= jep1; j++)
  {
    for (i=is-1; i<= iep1; i++)
    {
      if ( ua(i, j) > 0. )
      {
        if (i==1)
          ke(1,j)=uc(1,j)*sin_sg(1,j,1) + v(1,j)*cos_sg(1,j,1);
        else if (i==npx)
          ke(i,j)=uc(npx,j)*sin_sg(npx,j,1) + v(npx,j)*cos_sg(npx,j,1);
        else
          ke(i, j) = uc(i, j);
      } 
      else
      {
        if (i==0)
          ke(0,j)=uc(1,j)*sin_sg(0,j,3) + v(1,j)*cos_sg(0,j,3);
        else if (i==(npx-1))
          ke(i,j)=uc(npx,j)*sin_sg(npx-1,j,3) + v(npx,j)*cos_sg(npx-1,j,3);
        else
          ke(i,j)=uc(i+1,j);
      }
    }
  }
  
  for (j=js-1; j<= jep1; j++)
  {
    for (i=is-1; i<= iep1; i++)
    {
      if ( va(i, j) > 0. )
      {
        if (j==1)
          vort(i,1)=vc(i,1)*sin_sg(i,1,2) + u(i,1)*cos_sg(i,1,2);
        else if (j==npy)
          vort(i, j)=vc(i,npy)*sin_sg(i,npy,2) + u(i,npy)*cos_sg(i,npy,2);
        else
          vort(i, j) = vc(i, j);
      } 
      else
      {
        if ( j == 0 )
          vort(i, 0) = vc(i, 1) * sin_sg(i, 0, 4) + u(i, 1) * cos_sg(i, 0, 4);
        else if ( j == (npy - 1) )
          vort(i, j) = vc(i, npy) * sin_sg(i, npy-1, 4) + u(i, npy) * cos_sg(i, npy-1, 4);
        else
          vort(i, j) = vc(i, j+1);
      }
    }
  }

  for (j=js-1; j<= jep1; j++)
  {
    for (i=is-1; i<= iep1; i++)
    {
      ke(i, j) = dt4 * (ua(i, j) * ke(i, j) + va(i, j) * vort(i, j));
    }
  }

  //------------------------------
  // Compute circulation on C grid
  //------------------------------
  // To consider using true co-variant winds at face edges?

  for (j=js-1; j<= je+1; j++)
  {
    for (i=is; i<= ie+1; i++)
    {
      fx(i, j) = uc(i, j) * dxc(i, j);
    }
  }

  for (j=js; j<= je+1; j++)
  {
    for (i=is-1; i<= ie+1; i++)
    {
      fy(i, j) = vc(i, j) * dyc(i, j);
    }
  }

  for (j=js; j<= je+1; j++)
  {
    for (i=is; i<= ie+1; i++)
    {
      vort(i, j) =  fx(i, j-1) - fx(i, j) - fy(i-1, j) + fy(i, j);
    }
  }

  // Remove the extra term at the corners:

  if ( sw_corner )
    vort(1,     1) = vort(1,     1) + fy(0,     1);

  if ( se_corner )
    vort(npx,   1) = vort(npx,   1) - fy(npx,   1);

  if ( ne_corner )
    vort(npx, npy) = vort(npx, npy) - fy(npx, npy);

  if ( nw_corner )
    vort(1,   npy) = vort(1,   npy) + fy(0,   npy);


  //----------------------------
  // Compute absolute vorticity
  //----------------------------
  for (j=js; j<= je+1; j++)
  {
    for (i=is; i<= ie+1; i++)
    {
      vort(i, j) = fC(i, j) + rarea_c(i, j) * vort(i, j);
    }
  }

  //----------------------------------
  // Transport absolute vorticity:
  //----------------------------------
  // To go from v to contravariant v at the edges, we divide by sin_sg;
  // but we then must multiply by sin_sg to get the proper flux.
  // These cancel, leaving us with fy1 = dt2*v at the edges.
  // (For the same reason we only divide by sin instead of sin**2 in the interior)

  for (j=js; j<=je; j++)
  {
    #pragma omp simd
    for (i=is; i<=iep1; i++)
    {
      if (i == 1 || i == npx )
        fy1(i, j) = dt2 * v(i, j);
      else
        fy1(i, j) = dt2 * (v(i, j) - uc(i, j) * cosa_u(i, j)) / sina_u(i, j);

      if ( fy1(i, j) > 0.0 )
        fy(i, j) = vort(i, j);
      else
        fy(i, j) = vort(i, j+1);
    }
  }

  for (j=js; j<=jep1; j++)
  {
    if (j==1 || j==npy )
    {
      #pragma omp simd
      for (i=is; i<=ie; i++)
      {
        fx1(i, j) = dt2 * u(i, j);
        if ( fx1(i, j) > 0.0 )
          fx(i, j) = vort(i, j);
        else
          fx(i, j) = vort(i+1, j);
      }
    } else
    {
      #pragma omp simd
      for (i=is; i<=ie; i++)
      {
        fx1(i, j) = dt2 * (u(i, j) - vc(i, j) * cosa_v(i, j)) / sina_v(i, j);
        if ( fx1(i, j) > 0.0 )
          fx(i, j) = vort(i, j);
        else
          fx(i, j) = vort(i+1, j);
      }
    }
  }

  // Update time-centered winds on the C-Grid
  for (j=js; j<= je; j++)
  {
    for (i=is; i<= iep1; i++)
    {
      uc(i, j) = uc(i, j) + fy1(i, j) * fy(i, j) + rdxc(i, j) * (ke(i-1, j) - ke(i, j));
    }
  }
  for (j=js; j<= jep1; j++)
  {
    for (i=is; i<= ie; i++)
    {
      vc(i, j) = vc(i, j) - fx1(i, j) * fx(i, j) + rdyc(i, j) * (ke(i, j-1) - ke(i, j));
    }
  }

#ifdef ENABLE_GPTL
  if (do_profile == 1)
  {
    ret = gptlstop('c_sw');
  }
#endif
}

void CSWOuter::d2a2c_vect(Offset2DArray<double, true>& u, Offset2DArray<double, true>& v, 
        Offset2DArray<double, true>& ua, Offset2DArray<double, true>& va, 
        Offset2DArray<double, true>& uc, Offset2DArray<double, true>& vc, 
        Offset2DArray<double, true>& ut, Offset2DArray<double, true>& vt,
        Offset2DArray<double, true>& utmp, Offset2DArray<double, true>& vtmp) const
{
  //Offset2DArray<double, true> utmp(isd, ied, jsd, jed);
  //Offset2DArray<double, true> vtmp(isd, ied, jsd, jed);

  int i, j, ifirst, ilast;
  double zero=static_cast<double>(0.0);

#ifdef ENABLE_GPTL
  int ret;
  if (do_profile == 1)
  {
     ret = gptlstart ('d2a2c_vect');
  }
#endif

  // Initialize the non-existing corner regions
  for (j=jsd; j<=jed; j++)
  {
    for (i=isd; i<=ied; i++) 
    {
      utmp(i,j)=big_number;
      vtmp(i,j)=big_number;
    }
  }

  //----------
  // Interior:
  //----------

  for (j = std::max(4, js-1); j<=std::min(npy-4,je+1); j++)
  {
    for (i = std::max(4, isd); i<=std::min(npx-4,ied); i++)
    {
      utmp(i, j) = a2 * (u(i, j-1) + u(i, j+2)) + a1 * (u(i, j) + u(i, j+1));
    }
  }
  
  for (j = std::max(4, jsd); j<=std::min(npy-4,jed); j++)
  {
    for (i = std::max(4, is-1); i<=std::min(npx-4,ie+1); i++)
    {
      vtmp(i, j) = a2 * (v(i-1, j) + v(i+2, j)) + a1 * (v(i, j) + v(i+1, j));
    }
  }

  //----------
  // edges:
  //----------
  if ( js == 1 || jsd < 4)
  {
    for (j = jsd; j<= 3; j++)
    {
      for (i = isd; i<= ied; i++)
      {
        utmp(i, j) = 0.5 * (u(i, j) + u(i, j+1));
        vtmp(i, j) = 0.5 * (v(i, j) + v(i+1, j));
      }
    }
  }
  

  if ( (je + 1) == npy || jed >= (npy - 4) )
  {
    for (j = npy-3; j<= jed; j++)
    {
      for (i = isd; i<= ied; i++)
      {
        utmp(i, j) = 0.5 * (u(i, j) + u(i, j+1));
        vtmp(i, j) = 0.5 * (v(i, j) + v(i+1,j) );
      }
    }
  }
  

  if ( is == 1 || isd < 4 )
  {
    for (j = std::max(4, jsd); j<=std::min(npy-4, jed); j++)
    {
      for (i = isd; i<= 3; i++)
      {
        utmp(i, j) = 0.5 * (u(i, j) + u(i, j+1));
        vtmp(i, j) = 0.5 * (v(i, j) + v(i+1, j));
      }
    }
  }

  if ( (ie+1) == npx || ied >= (npx-4) )
  {
    for (j = std::max(4, jsd); j<=std::min(npy-4, jed); j++)
    {
      for (i = npx-3; i<= ied; i++)
      {
        utmp(i, j) = 0.5 * (u(i, j) + u(i, j+1));
        vtmp(i, j) = 0.5 * (v(i, j) + v(i+1, j));
      }
    }
  }

  for (j = js-2; j<= je+2; j++)
  {
    for (i = is-2; i<= ie+2; i++)
    {
      ua(i, j) = (utmp(i, j) - vtmp(i, j) * cosa_s(i, j)) * rsin2(i, j);
      va(i, j) = (vtmp(i, j) - utmp(i, j) * cosa_s(i, j)) * rsin2(i, j);
    }
  }


  // A -> C
  //--------------
  // Fix the edges
  //--------------
  // Xdir:
  if ( sw_corner )
  {
    for (i = -2; i<= 0; i++)
    {
      utmp(i, 0) = -vtmp(0, 1-i);
    }
  }

  if ( se_corner )
  {
    for (i = 0; i<= 2; i++)
    {
      utmp(npx+i, 0) = vtmp(npx, i+1);
    }
  }
    
  if ( ne_corner )
  {
    for (i = 0; i<= 2; i++)
    {
      utmp(npx+i, npy) = -vtmp(npx, je-i);
    }
  }

  if ( nw_corner )
  {
    for (i = -2; i<= 0; i++)
    {
      utmp(i, npy) = vtmp(0, je+i);
    }
  }

  ifirst = std::max(3,     is-1);
  ilast  = std::min(npx-2, ie+2);

  //---------------------------------------------
  // 4th order interpolation for interior points:
  //---------------------------------------------
  for (j = js-1; j<= je+1; j++)
  {
    for (i = ifirst; i<= ilast; i++)
    {
      uc(i, j) = a1 * (utmp(i-1, j) + utmp(i, j)) + a2 * (utmp(i-2, j) + utmp(i+1, j));
      ut(i, j) = (uc(i, j) - v(i, j) * cosa_u(i, j)) * rsin_u(i, j);
    }
  }

  // Xdir:
  if ( sw_corner )
  {
    ua(-1, 0) = -va(0, 2);
    ua( 0, 0) = -va(0, 1);
  }

  if ( se_corner )
  {
    ua(npx,   0) = va(npx, 1);
    ua(npx+1, 0) = va(npx, 2);
  }
  
  if ( ne_corner )
  {
    ua(npx,   npy) = -va(npx, npy-1);
    ua(npx+1, npy) = -va(npx, npy-2);
  }

  if ( nw_corner )
  {
    ua(-1, npy) = va(0, npy-2);
    ua( 0, npy) = va(0, npy-1);
  }

  if ( is == 1 )
  {
    for (j = js-1; j<= je+1; j++)
    {
      uc(0, j) = c1 * utmp(-2, j) + c2 * utmp(-1, j) + c3 * utmp(0, j);

      double ua_interp[4];
      double dxa_interp[4];
      for (int iter=0;iter<=3;iter++)
      {
        ua_interp[iter] = ua(iter-1,j);
        dxa_interp[iter] = dxa(iter-1,j);
      }
      ut(1,j)=edge_interpolate4(ua_interp, dxa_interp);
      //Want to use the UPSTREAM value
      if (ut(1, j) < zero)
        uc(1, j) = ut(1, j) * sin_sg(0, j, 3);
      else
        uc(1, j) = ut(1, j) * sin_sg(1, j, 1);

      uc(2, j) = c1 * utmp(3, j) + c2 * utmp(2, j) + c3 * utmp(1, j);
      ut(0, j) = (uc(0, j) - v(0, j) * cosa_u(0, j)) * rsin_u(0, j);
      ut(2, j) = (uc(2, j) - v(2, j) * cosa_u(2, j)) * rsin_u(2, j);
    }

    if ( (ie+1) == npx )
    {
      for (j = js-1; j<= je+1; j++)
      {
        uc(npx-1, j) = c1 * utmp(npx-3, j) + c2 * utmp(npx-2, j) + c3 * utmp(npx-1, j);
        i = npx;
        ut(i, j) = 0.25 * (-ua(i-2, j) + 3. * (ua(i-1, j) + ua(i, j)) - ua(i+1, j));
        //ut(i, j) = edge_interpolate4(ua(i-2:i+1, j), dxa(i-2:i+1, j))
        double ua_interp[4];
        double dxa_interp[4];
        for (int iter=0;iter<=3;iter++)
        {
          ua_interp[iter]=ua(i-2+iter,j);
          dxa_interp[iter]=dxa(i-2+iter,j);
        }

        ut(i,j)=edge_interpolate4(ua_interp, dxa_interp);

        if ( ut(i,j) < zero )
          uc(i, j) = ut(i, j) * sin_sg(i-1, j, 3);
        else
          uc(i, j) = ut(i, j) * sin_sg(i, j, 1);

        uc(npx+1, j) = c3 * utmp(npx, j) + c2 * utmp(npx+1, j) + c1 * utmp(npx+2, j);
        ut(npx-1, j) = (uc(npx-1, j) - v(npx-1, j) * cosa_u(npx-1, j)) * rsin_u(npx-1, j);
        ut(npx+1, j) = (uc(npx+1, j) - v(npx+1, j) * cosa_u(npx+1, j)) * rsin_u(npx+1, j);
      }
    }
  }

  //------
  // Ydir:
  //------
  if ( sw_corner )
  {
    for (j = -2; j<= 0; j++)
    {
      vtmp(0, j) = -utmp(1-j, 0);
    }
  }

  if ( nw_corner )
  {
    for (j = 0; j<= 2; j++)
    {
      vtmp(0, npy+j) = utmp(j+1, npy);
    }
  }

  if ( se_corner )
  {
    for (j = -2; j<= 0; j++)
    {
      vtmp(npx, j) = utmp(ie+j, 0);
    }
  }

  if ( ne_corner )
  {
    for (j = 0; j<= 2; j++)
    {
      vtmp(npx, npy+j) = -utmp(ie-j, npy);
    }
  }

  if ( sw_corner )
  {
    va(0, -1) = -ua(2, 0);
    va(0,  0) = -ua(1, 0);
  }

  if ( se_corner )
  {
    va(npx,  0) = ua(npx-1, 0);
    va(npx, -1) = ua(npx-2, 0);
  }

  if ( ne_corner )
  {
    va(npx, npy  ) = -ua(npx-1, npy);
    va(npx, npy+1) = -ua(npx-2, npy);
  }

  if ( nw_corner )
  {
    va(0, npy)   = ua(1, npy);
    va(0, npy+1) = ua(2, npy);
  }

  for (j = js-1; j<= je+2; j++)
  {
    if ( j == 1 )
    {
      for (i = is-1; i<= ie+1; i++)
      {
        //vt(i, j) = edge_interpolate4(va(i, -1:2), dya(i, -1:2))
        double va_interp[4];
        double dya_interp[4];
        for (int iter=0;iter<=3;iter++)
        {
          va_interp[iter]=va(i,iter-1);
          dya_interp[iter]=dya(i,iter-1);
        }
        vt(i,j)=edge_interpolate4(va_interp, dya_interp);

        if ( vt(i, j) < 0. )
          vc(i, j) = vt(i, j) * sin_sg(i, j-1, 4);
        else
          vc(i, j) = vt(i, j) * sin_sg(i, j, 2);
      }
    } else if ( j == 0 || j == (npy-1) )
    {
      for (i = is-1; i<= ie+1; i++)
      {
        vc(i, j) = c1 * vtmp(i, j-2) + c2 * vtmp(i, j-1) + c3 * vtmp(i, j);
        vt(i, j) = (vc(i, j) - u(i, j) * cosa_v(i, j)) * rsin_v(i, j);
      }
    } else if ( j == 2 || j == (npy+1) )
    {
      for (i = is-1; i<= ie+1; i++)
      {
        vc(i, j) = c1 * vtmp(i, j+1) + c2 * vtmp(i, j) + c3 * vtmp(i, j-1);
        vt(i, j) = (vc(i, j) - u(i, j) * cosa_v(i, j)) * rsin_v(i, j);
      }
    } else if ( j == npy )
    {
      for (i = is-1; i<= ie+1; i++)
      {
        vt(i, j) = 0.25 * (-va(i, j-2) + 3. * (va(i, j-1) + va(i, j)) - va(i, j+1));
        //vt(i, j) = edge_interpolate4(va(i, j-2:j+1), dya(i, j-2:j+1))
        double va_interp[4];
        double dya_interp[4];
        for (int iter=0; iter<=3; iter++)
        {
          va_interp[iter]=va(i,j-2+iter);
          dya_interp[iter]=dya(i,j-2+iter);
        }
        vt(i,j)=edge_interpolate4(va_interp, dya_interp);

        if ( vt(i, j) < 0. )
          vc(i, j) = vt(i, j) * sin_sg(i, j-1, 4);
        else
          vc(i, j) = vt(i, j) * sin_sg(i, j, 2);
      }
    } else
    {
      // 4th order interpolation for interior points:
      for (i = is-1; i<= ie+1; i++)
      {
        vc(i, j) = a2 * (vtmp(i, j-2) + vtmp(i, j+1)) + a1 * (vtmp(i, j-1) + vtmp(i, j));
        vt(i, j) = (vc(i, j) - u(i, j) * cosa_v(i, j)) * rsin_v(i, j);
      }
    }
  }

#ifdef ENABLE_GPTL
  if (do_profile == 1)
  {
     ret = gptlstop ('d2a2c_vect');
  }
#endif

}


double CSWOuter::edge_interpolate4(double *ua, double *dxa) const
{
  double u0L, u0R;
  constexpr double half=static_cast<double>(0.5);
  constexpr double two=static_cast<double>(2.0);
  u0L = half*((two*dxa[1]+dxa[0])*ua[1] - dxa[1]*ua[0]) / ( dxa[0]+dxa[1] );
  u0R = half*((two*dxa[2]+dxa[3])*ua[2] - dxa[2]*ua[3]) / ( dxa[2]+dxa[3] );
  return (u0L+u0R);
}


void CSWOuter::fill_4corners(Offset2DArray<double, true>& q, int dir) const
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
      if (sw_corner)
      {
        q(-1, 0)      = q(0, 2);
        q(0, 0)       = q(0, 1);
      }
      if (se_corner)
      {
        q(npx+1, 0)   = q(npx, 2);
        q(npx, 0)     = q(npx, 1);
      }
      if (nw_corner)
      {
        q(0, npy)     = q(0, npy-1);
        q(-1, npy)    = q(0, npy-2);
      }
      if (ne_corner)
      {
        q(npx, npy)   = q(npx, npy-1);
        q(npx+1, npy) = q(npx, npy-2);
      }
      break;
    case(2): // y-dir
      if (sw_corner)
      {
        q(0, 0)       = q(1, 0);
        q(0, -1)      = q(2, 0);
      }
      if (se_corner)
      {
        q(npx, 0)     = q(npx-1, 0);
        q(npx, -1)    = q(npx-2, 0);
      }
      if (nw_corner)
      {
        q(0, npy)     = q(1, npy);
        q(0, npy+1)   = q(2, npy);
      }
      if (ne_corner)
      {
        q(npx, npy)   = q(npx-1, npy);
        q(npx, npy+1) = q(npx-2, npy);
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

void CSWOuter::fill2_4corners(Offset2DArray<double, true>& q1, Offset2DArray<double, true>& q2, int dir) const
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
      if (sw_corner)
      {
        q1(-1, 0) = q1(0, 2);
        q1( 0, 0) = q1(0, 1);
        q2(-1, 0) = q2(0, 2);
        q2( 0, 0) = q2(0, 1);
      }
      if ( se_corner )
      {
        q1(npx+1, 0) = q1(npx, 2);
        q1(npx,   0) = q1(npx, 1);
        q2(npx+1, 0) = q2(npx, 2);
        q2(npx,   0) = q2(npx, 1);
      }
      if ( nw_corner )
      {
        q1( 0, npy) = q1(0, npy-1);
        q1(-1, npy) = q1(0, npy-2);
        q2( 0, npy) = q2(0, npy-1);
        q2(-1, npy) = q2(0, npy-2);
      }
      if ( ne_corner )
      {
        q1(npx,   npy) = q1(npx, npy-1);
        q1(npx+1, npy) = q1(npx, npy-2);
        q2(npx,   npy) = q2(npx, npy-1);
        q2(npx+1, npy) = q2(npx, npy-2);
      }
      break;
    case(2): // y-dir
      if ( sw_corner )
      {
        q1(0,  0) = q1(1, 0);
        q1(0, -1) = q1(2, 0);
        q2(0,  0) = q2(1, 0);
        q2(0, -1) = q2(2, 0);
      }
      if ( se_corner )
      {
        q1(npx,  0) = q1(npx-1, 0);
        q1(npx, -1) = q1(npx-2, 0);
        q2(npx,  0) = q2(npx-1, 0);
        q2(npx, -1) = q2(npx-2, 0);
      }
      if ( nw_corner )
      {
        q1(0, npy)   = q1(1, npy);
        q1(0, npy+1) = q1(2, npy);
        q2(0, npy)   = q2(1, npy);
        q2(0, npy+1) = q2(2, npy);
      }
      if ( ne_corner )
      {
        q1(npx, npy)   = q1(npx-1, npy);
        q1(npx, npy+1) = q1(npx-2, npy);
        q2(npx, npy)   = q2(npx-1, npy);
        q2(npx, npy+1) = q2(npx-2, npy);
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


void CSWOuter::divergence_corner(Offset2DArray<double, true>& u, Offset2DArray<double, true>& v, 
        Offset2DArray<double, true>& ua, Offset2DArray<double, true>& va, 
        Offset2DArray<double, true>& divg_d,
        Offset2DArray<double, true>& uf, Offset2DArray<double, true>& vf) const
{
  // Offset2DArray<double, true> uf(is-2, ie+2, js-1, je+2);
  // Offset2DArray<double, true> vf(is-1, ie+2, js-2, je+2);
  
  const int is2 = std::max(2, is);
  const int ie1 = std::min(npx-1, ie+1);
  int i, j;
  
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
  
  for (j=js; j<=je+1; j++)
  {
    if ( j == 1 || j == npy )
    {
      for (i = is-1; i<=ie+1; i++)
      {
        uf(i, j) = u(i, j) * dyc(i, j) * 0.5 * (sin_sg(i, j-1, 4) + sin_sg(i, j, 2));
      }
    } 
    else
    {
      for (i = is-1; i<=ie+1; i++)
      {
        uf(i, j) = (u(i, j) - 0.25 * (va(i, j-1) + va(i, j)) *
                   (cos_sg(i, j-1, 4) + cos_sg(i, j, 2))) *
                    dyc(i, j) * 0.5 * (sin_sg(i, j-1, 4) + sin_sg(i, j, 2));
      }
    }
  }

  for (j = js-1; j<=je+1; j++)
  {
    for (i = is2; i<=ie1; i++)
    {
      vf(i, j) = (v(i, j) - 0.25 * (ua(i-1, j) + ua(i, j)) *
                 (cos_sg(i-1, j, 3) + cos_sg(i, j, 1))) *
                  dxc(i, j) * 0.5 * (sin_sg(i-1, j, 3) + sin_sg(i, j, 1));
    }
    if (is==1)
      vf(1, j) = v(1, j) * dxc(1, j) * 0.5 * (sin_sg(0, j, 3) + sin_sg(1, j, 1));

    if ( (ie+1) == npx )
      vf(npx, j) = v(npx, j) * dxc(npx, j) * 0.5 * (sin_sg(npx-1, j, 3) + sin_sg(npx, j, 1));
    
  }
  

  for (j = js; j<=je+1; j++)
  {
    for (i = is; i<=ie+1; i++)
    {
      divg_d(i, j) = vf(i, j-1) - vf(i, j) + uf(i-1, j) - uf(i, j);
    }
  }

    
  // Remove the extra term at the corners:
  if (sw_corner)
  {
    divg_d(1,     1) = divg_d(1,     1) - vf(1,     0);
  }
  if (se_corner)
  {
    divg_d(npx,   1) = divg_d(npx,   1) - vf(npx,   0);
  }
  if (ne_corner)
  {
    divg_d(npx, npy) = divg_d(npx, npy) + vf(npx, npy);
  }
  if (nw_corner)
  {
    divg_d(1,   npy) = divg_d(1,   npy) + vf(1,   npy);
  }
  

  for (j=js; j<=je+1; j++)
  {
    for (i=is; i<=ie+1; i++)
    {
      divg_d(i, j) = rarea_c(i, j) * divg_d(i, j);
    }
  }


#ifdef ENABLE_GPTL
  if (do_profile == 1)
  {
     ret = gptlstop('divergence_corner');
  }
#endif

}