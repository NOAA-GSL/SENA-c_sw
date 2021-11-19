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

// Templates for mapping n-dimensional arrays with non-zero index into
// one-dimensional, zero-indexed arrays

#include <stdexcept>
#include <algorithm>
#include <CL/sycl.hpp>

using namespace sycl;

//#define OFFSETRANGECHECK
#define RANGE_ERROR_MSG "Index out of OffsetArray range"

///1D array class. Either takes a pointer to an array with the 'correct' size 
///and does not take care of the memory, or gets size of the requested memory and
///allocated memory in constructor and destroys it in destructor.

template <class T>
class OffsetArray
{
    const int istart;
    const int iend;
    T * const values;

  public:
    OffsetArray() = delete;

    OffsetArray(int istart_in, int iend_in, T * values_in) :
      istart(istart_in),
      iend(istart_in),
      values(values_in)
    {
    }

    T &operator()(int i) const
    {
      return values[i-istart];
    }

    size_t size() const {return iend-istart+1;}
};


/////////////////////////////////////////2D/////////////////////////////////////



template <class T, bool f_order=false>
class Offset2DArray
{
public:
    const int istart;
    const int iend;
    const int jstart;
    const int jend;
    const int isize;
    const int jsize;
    T * const values;

    Offset2DArray() = delete;
    
    ///constructor. The memory is owned by outside source. Note that the given sizes defined by i/j-start/end 
    ///need to coincide with the size of values. Otherwise the behaviour is undefined
    Offset2DArray(int istart_in, int iend_in, int jstart_in, int jend_in, T * values_in) :
      istart(std::min(istart_in, iend_in)),
      iend(std::max(istart_in, iend_in)),
      jstart(std::min(jstart_in, jend_in)),
      jend(std::max(jstart_in, jend_in)),
      isize(iend-istart+1),
      jsize(jend-jstart+1),
      values(values_in)
    {
    }

    #pragma omp declare simd
    T &operator()(int i, int j) const
    {
      int index;
      if (f_order)
        index=(j-jstart)*isize + (i-istart);
      else
        index=(i-istart)*jsize + (j-jstart);

      return values[index];
    }

    size_t size() const {return isize*jsize;}
};


//////////////////////////////////3D ///////////////////////////////////////////



template <class T, bool f_order=false>
class Offset3DArray
{
public:
    const int istart;
    const int iend;
    const int jstart;
    const int jend;
    const int kstart;
    const int kend;
    const int isize;
    const int jsize;
    const int ksize;
    T * const values;

    Offset3DArray() = delete;

    ///constructor. Caller owns data. Assumes the size of data and the given start and end points match.
    Offset3DArray(int istart_in, int iend_in, int jstart_in, int jend_in, int kstart_in, int kend_in, T * values_in) :
      istart(std::min(istart_in, iend_in)),
      iend(std::max(istart_in, iend_in)),
      jstart(std::min(jstart_in, jend_in)),
      jend(std::max(jstart_in, jend_in)),
      kstart(std::min(kstart_in, kend_in)),
      kend(std::max(kstart_in, kend_in)),
      isize(iend-istart+1),
      jsize(jend-jstart+1),
      ksize(kend-kstart+1),
      values(values_in)
    {
    }   

    #pragma omp declare simd
    T &operator()(int i, int j, int k) const
    {
      int index;
      if (f_order)
        index=(k-kstart)*jsize*isize + (j-jstart)*isize + (i-istart);
      else
        index=(i-istart)*jsize*ksize + (j-jstart)*ksize + (k-kstart);

      return values[index];
    }

    size_t size() const {return isize*jsize*ksize;}
};



///Allocator class for the generation of Offset Arrays on devices. 
///Takes a host array T* and generates a DPC++ allocated array (host, device, shared).
///Allocate offset array with that data.
///If T* is a nullptr, no memory copies are done.
///Also implements Deallocators which can copy data back from the device to the host
///and then frees device memory.
class Allocator
{
private:
    usm::alloc alloct_;

public:
    Allocator(usm::alloc alloct) : alloct_(alloct) {}

    template <typename T>
    T* Allocate(size_t n, T* vals, queue Q) const
    {
        T* ret = nullptr;
        if (alloct_ == usm::alloc::shared)
            ret = malloc_shared<T>(n, Q);
        else if (alloct_ == usm::alloc::device)
            ret = malloc_device<T>(n, Q);
        else if (alloct_ == usm::alloc::host)
            ret = malloc_host<T>(n, Q);
        else throw std::invalid_argument("Array can only be shared, host or device.\n");

        if (vals != nullptr)
            Q.memcpy(ret, vals, n*sizeof(T));

        return ret;
    }

    template <typename T>
    OffsetArray<T> Allocate(int istart, int iend, T * values, queue Q) const
    {
        return OffsetArray<T>(istart, iend, Allocate<T>(iend-istart+1, values, Q));
    }

    template <typename T, bool order>
    Offset2DArray<T, order> Allocate(int istart, int iend, int jstart, int jend, T * values, queue Q) const
    {
        const int size = (iend-istart+1)*(jend-jstart+1);
        return Offset2DArray<T, order>(istart, iend, jstart, jend, Allocate<T>(size, values, Q));
    }

    template <typename T, bool order>
    Offset3DArray<T, order> Allocate(int istart, int iend, int jstart, int jend, int kstart, int kend,
        T * values, queue Q) const
    {
        const int size = (iend-istart+1)*(jend-jstart+1)*(kend-kstart+1);
        return Offset3DArray<T, order>(istart, iend, jstart, jend, kstart, kend, Allocate<T>(size, values, Q));
    }

    ///Deallocations
    template<typename T>
    void Deallocate(size_t n, T* in, T* out, queue Q) const
    {
        if (out != nullptr)
            Q.memcpy(out, in, n*sizeof(T)).wait();

        free(in, Q);
    }

    template <typename T>
    void Deallocate(const OffsetArray<T>& in, T* out, queue Q ) const
    {
        Deallocate(in.size(), in.values, out, Q);
    }

    template <typename T, bool order>
    void Deallocate(const Offset2DArray<T, order>& in, T* out, queue Q ) const
    {
        Deallocate(in.size(), in.values, out, Q);
    }

    template <typename T, bool order>
    void Deallocate(const Offset3DArray<T, order>& in, T* out, queue Q ) const
    {
        Deallocate(in.size(), in.values, out, Q);
    }
};