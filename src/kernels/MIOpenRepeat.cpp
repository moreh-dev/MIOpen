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
#include "tensor_view.hpp"

#define LOCAL_SIZE_128 128

template <typename T>
__device__ void RepeatForwardImpl(const T* __restrict__ x,
                                  T* __restrict__ y,
                                  uint64_t inout_size,
                                  uint64_t offset,
                                  tensor_view_t<5> x_tv,
                                  tensor_view_t<5> y_tv)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= inout_size)
        return;

    // get output index
    tensor_layout_t<5> output_ncdhw(y_tv, gid);

    // get input index
    tensor_layout_t<5> input_ncdhw(x_tv, 0);
    for(uint64_t i = offset; i < 5; i++)
    {
        input_ncdhw.layout[i - offset] = output_ncdhw.layout[i] % x_tv.size[i - offset];
    }

    y[y_tv.get_tensor_view_idx(output_ncdhw)] = x[x_tv.get_tensor_view_idx(input_ncdhw)];
}

template <typename T>
__device__ void RepeatLargeKBackwardImpl(const T* __restrict__ dy,
                                         T* __restrict__ dx,
                                         uint64_t N,
                                         uint64_t K,
                                         tensor_view_t<5> dy_tv,
                                         tensor_view_t<5> dx_tv)
{
    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;
    if(gid >= N)
        return;

    __shared__ FLOAT_ACCUM ltmp[LOCAL_SIZE_128];

    FLOAT_ACCUM sum = static_cast<FLOAT_ACCUM>(0.0);
    for(uint64_t k = lid; k < K; k += LOCAL_SIZE_128)
    {
        tensor_layout_t<5> dy_ncdhw(dy_tv, gid + k * N);
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(dy[dy_tv.get_tensor_view_idx(dy_ncdhw)]);
        sum += val;
    }

    ltmp[lid] = sum;
    __syncthreads();

    if(lid < 64)
    {
        ltmp[lid] += ltmp[lid + 64];
        __syncthreads();
        ltmp[lid] += ltmp[lid + 32];
        __syncthreads();
        ltmp[lid] += ltmp[lid + 16];
        __syncthreads();
        ltmp[lid] += ltmp[lid + 8];
        __syncthreads();
        ltmp[lid] += ltmp[lid + 4];
        __syncthreads();
        ltmp[lid] += ltmp[lid + 2];
        __syncthreads();
        ltmp[lid] += ltmp[lid + 1];
    }

    if(lid == 0)
    {
        tensor_layout_t<5> dx_ncdhw(dx_tv, gid);
        dx[dx_tv.get_tensor_view_idx(dx_ncdhw)] = CVT_ACCUM2FLOAT(ltmp[0]);
    }
}

template <typename T>
__device__ void RepeatSmallKBackwardImpl(const T* __restrict__ dy,
                                         T* __restrict__ dx,
                                         uint64_t N,
                                         uint64_t K,
                                         tensor_view_t<5> dy_tv,
                                         tensor_view_t<5> dx_tv)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= N)
        return;

    FLOAT_ACCUM sum = static_cast<FLOAT_ACCUM>(0.0);

    for(uint64_t k = 0; k < K; k++)
    {
        tensor_layout_t<5> dy_ncdhw(dy_tv, gid + k * N);
        sum += CVT_FLOAT2ACCUM(dy[dy_tv.get_tensor_view_idx(dy_ncdhw)]);
    }

    tensor_layout_t<5> dx_ncdhw(dx_tv, gid);
    dx[dx_tv.get_tensor_view_idx(dx_ncdhw)] = CVT_ACCUM2FLOAT(sum);
}

extern "C" __global__ void RepeatForward(const FLOAT* __restrict__ x,
                                         FLOAT* __restrict__ y,
                                         uint64_t inout_size,
                                         uint64_t offset,
                                         tensor_view_t<5> x_tv,
                                         tensor_view_t<5> y_tv)
{
    RepeatForwardImpl<FLOAT>(x, y, inout_size, offset, x_tv, y_tv);
}

extern "C" __global__ void RepeatLargeKBackward(const FLOAT* __restrict__ dy,
                                                FLOAT* __restrict__ dx,
                                                uint64_t N,
                                                uint64_t K,
                                                tensor_view_t<5> dy_tv,
                                                tensor_view_t<5> dx_tv)
{
    RepeatLargeKBackwardImpl<FLOAT>(dy, dx, N, K, dy_tv, dx_tv);
}

extern "C" __global__ void RepeatSmallKBackward(const FLOAT* __restrict__ dy,
                                                FLOAT* __restrict__ dx,
                                                uint64_t N,
                                                uint64_t K,
                                                tensor_view_t<5> dy_tv,
                                                tensor_view_t<5> dx_tv)
{
    RepeatSmallKBackwardImpl<FLOAT>(dy, dx, N, K, dy_tv, dx_tv);
}