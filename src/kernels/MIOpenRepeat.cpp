/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

__device__ static inline __half __ushort_as___half(ushort x)
{
    static_assert(sizeof(ushort) == sizeof(__half), "");

    __half tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ static inline ushort ____half_as_ushort(__half x)
{
    static_assert(sizeof(ushort) == sizeof(__half), "");

    ushort tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ inline void atomic_add_g(ushort* addr, const float val)
{
    float val_       = bfloat16_to_float(val);
    size_t offset    = reinterpret_cast<size_t>(addr) & 0x2;
    bool is_32_align = offset;
    uint32_t* addr_as_uint32_t =
        reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(addr) - offset);
    uint32_t current = *addr_as_uint32_t;

    uint32_t expected, next;
    ushort current_ushort, next_ushort;
    float next_float;

    do
    {
        expected       = current;
        current_ushort = is_32_align ? current >> 16 : current & 0xffff;

        next_float  = __uint_as_float(static_cast<uint32_t>(current_ushort) << 16) + val_;
        next_ushort = static_cast<ushort>(__float_as_uint(next_float) >> 16);
        next        = is_32_align ? (current & 0xffff) | (next_ushort << 16)
                                  : (current & 0xffff0000) | next_ushort;

        current = atomicCAS(addr_as_uint32_t, expected, next);
    } while(current != expected);
}

__device__ inline void atomic_add_g(__half* addr, const __half val)
{
    size_t offset    = reinterpret_cast<size_t>(addr) & 0x2;
    bool is_32_align = offset;
    uint32_t* addr_as_uint32_t =
        reinterpret_cast<uint32_t*>(reinterpret_cast<size_t>(addr) - offset);
    uint32_t current = *addr_as_uint32_t;

    uint32_t expected, next;
    ushort current_ushort, next_ushort;

    do
    {
        expected       = current;
        current_ushort = is_32_align ? current >> 16 : current & 0xffff;

        next_ushort = ____half_as_ushort(__ushort_as___half(current_ushort) + val);
        next        = is_32_align ? (current & 0xffff) | (next_ushort << 16)
                                  : (current & 0xffff0000) | next_ushort;

        current = atomicCAS(addr_as_uint32_t, expected, next);
    } while(current != expected);
}

__device__ inline void atomic_add_g(float* addr, const float val) { atomicAdd(addr, val); }

__device__ inline void atomic_add_g(_Float16* addr, const _Float16 val)
{
    __half val_half = static_cast<__half>(val);
    atomic_add_g(reinterpret_cast<__half*>(addr), val_half);
}

__device__ void GET_NCDHW(uint64_t& n,
                          uint64_t& c,
                          uint64_t& d,
                          uint64_t& h,
                          uint64_t& w,
                          uint64_t gid,
                          uint64_t output_dim0,
                          uint64_t output_dim1,
                          uint64_t output_dim2,
                          uint64_t output_dim3,
                          uint64_t output_dim4)
{
    uint64_t ncdh = (gid) / output_dim4;
    w             = (gid) % output_dim4;
    uint64_t ncd  = (ncdh) / output_dim3;
    h             = (ncdh) % output_dim3;
    uint64_t nc   = (ncd) / output_dim2;
    d             = (ncd) % output_dim2;
    n             = (nc) / output_dim1;
    c             = (nc) % output_dim1;
}

__device__ int GET_5D_INDEX(const uint64_t input_dimensions[5],
                             uint64_t n,
                             uint64_t c,
                             uint64_t d,
                             uint64_t h,
                             uint64_t w)
{
    return (((n * input_dimensions[1] + c) * input_dimensions[2] + d) * input_dimensions[3] + h) *
                 input_dimensions[4] + w;
}

extern "C" __global__ void RepeatForward(const FLOAT* __restrict__ x,
                                         FLOAT* __restrict__ y,
                                         uint64_t inout_size,
                                         uint64_t offset,
                                         uint64_t input_dim0,
                                         uint64_t input_dim1,
                                         uint64_t input_dim2,
                                         uint64_t input_dim3,
                                         uint64_t input_dim4,
                                         uint64_t output_dim0,
                                         uint64_t output_dim1,
                                         uint64_t output_dim2,
                                         uint64_t output_dim3,
                                         uint64_t output_dim4)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= inout_size)
        return;

    uint64_t input_dimensions[5] = {input_dim0, input_dim1, input_dim2, input_dim3, input_dim4};

    // get output index
    uint64_t o[5];
    GET_NCDHW(o[0],
              o[1],
              o[2],
              o[3],
              o[4],
              gid,
              output_dim0,
              output_dim1,
              output_dim2,
              output_dim3,
              output_dim4);

    // get input index
    uint64_t n[5] = {0, 0, 0, 0, 0};
    for(uint64_t i = offset; i < 5; i++)
    {
        n[i - offset] = o[i] % input_dimensions[i - offset];
    }

    uint64_t input_index = GET_5D_INDEX(input_dimensions, n[0], n[1], n[2], n[3], n[4]);
    y[gid] = x[input_index];
}

extern "C" __global__ void RepeatBackward(const FLOAT* __restrict__ dy,
                                          FLOAT* __restrict__ dx,
                                          uint64_t inout_size,
                                          uint64_t offset,
                                          uint64_t output_grad_dim0,
                                          uint64_t output_grad_dim1,
                                          uint64_t output_grad_dim2,
                                          uint64_t output_grad_dim3,
                                          uint64_t output_grad_dim4,
                                          uint64_t input_grad_dim0,
                                          uint64_t input_grad_dim1,
                                          uint64_t input_grad_dim2,
                                          uint64_t input_grad_dim3,
                                          uint64_t input_grad_dim4)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= inout_size)
        return;

    uint64_t input_grad_dimensions[5] = {
        input_grad_dim0, input_grad_dim1, input_grad_dim2, input_grad_dim3, input_grad_dim4};

    // get output index
    uint64_t o[5];
    GET_NCDHW(o[0],
              o[1],
              o[2],
              o[3],
              o[4],
              gid,
              output_grad_dim0,
              output_grad_dim1,
              output_grad_dim2,
              output_grad_dim3,
              output_grad_dim4);

    // get input index
    uint64_t n[5] = {0, 0, 0, 0, 0};
    for(uint64_t i = offset; i < 5; i++)
    {
        n[i - offset] = o[i] % input_grad_dimensions[i - offset];
    }

    uint64_t input_grad_index = GET_5D_INDEX(input_grad_dimensions, n[0], n[1], n[2], n[3], n[4]);
    atomic_add_g(&dx[input_grad_index], dy[gid]);
}
