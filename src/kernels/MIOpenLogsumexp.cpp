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
#include "miopen_limits.hpp"
#include "tensor_view.hpp"
#include "dims_utils.hpp"

#define LOCAL_SIZE_64 64
#define LIMIT_SMALL_K 16

template <typename T>
__device__ void LogsumexpLargeKForwardImpl(const T* __restrict__ input,
                                           T* __restrict__ output,
                                           int64_t N,
                                           int64_t K,
                                           tensor_view_t<5> input_tv,
                                           tensor_view_t<5> output_tv)
{
    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;
    if(gid >= N)
        return;

    __shared__ FLOAT_ACCUM ltmp[LOCAL_SIZE_64];
    FLOAT_ACCUM max_v = std::numeric_limits<FLOAT_ACCUM>::lowest();

    for(uint64_t k = lid; k < K; k += LOCAL_SIZE_64)
    {
        tensor_layout_t<5> input_ncdhw(input_tv, gid * K + k);
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_ncdhw)]);
        max_v           = max_v > val ? max_v : val;
    }

    // Max Reduction
    ltmp[lid] = max_v;
    __syncthreads();

    for(size_t i = LOCAL_SIZE_64 / 2; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp[lid] = ltmp[lid] > ltmp[lid + i] ? ltmp[lid] : ltmp[lid + i];
        }
        __syncthreads();
    }

    max_v = ltmp[0];
    __syncthreads();

    FLOAT_ACCUM logsum = static_cast<FLOAT_ACCUM>(0.0);
    for(uint64_t k = lid; k < K; k += LOCAL_SIZE_64)
    {
        tensor_layout_t<5> input_ncdhw(input_tv, gid * K + k);
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_ncdhw)]);
        logsum += expf(val - max_v);
    }

    // Logsum Reduction
    ltmp[lid] = logsum;
    __syncthreads();

    if(lid < 32)
    {
        ltmp[lid] = ltmp[lid] + ltmp[lid + 32];
        __syncthreads();
        ltmp[lid] = ltmp[lid] + ltmp[lid + 16];
        __syncthreads();
        ltmp[lid] = ltmp[lid] + ltmp[lid + 8];
        __syncthreads();
        ltmp[lid] = ltmp[lid] + ltmp[lid + 4];
        __syncthreads();
        ltmp[lid] = ltmp[lid] + ltmp[lid + 2];
        __syncthreads();
        ltmp[lid] = ltmp[lid] + ltmp[lid + 1];
    }

    if(lid == 0)
    {
        tensor_layout_t<5> output_ncdhw(output_tv, gid);
        if(ltmp[0] > static_cast<FLOAT_ACCUM>(0.0))
            output[output_tv.get_tensor_view_idx(output_ncdhw)] =
                CVT_ACCUM2FLOAT(max_v + logf(ltmp[0]));
        else
            output[output_tv.get_tensor_view_idx(output_ncdhw)] =
                CVT_ACCUM2FLOAT(max_v + std::numeric_limits<FLOAT_ACCUM>::lowest());
    }
}

template <typename T>
__device__ void LogsumexpSmallKForwardImpl(const T* __restrict__ input,
                                           T* __restrict__ output,
                                           int64_t N,
                                           int64_t K,
                                           tensor_view_t<5> input_tv,
                                           tensor_view_t<5> output_tv)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= N)
        return;

    FLOAT_ACCUM vals[LIMIT_SMALL_K];
    FLOAT_ACCUM max = std::numeric_limits<FLOAT_ACCUM>::lowest();

    for(int64_t k = 0; k < K; k++)
    {
        tensor_layout_t<5> input_ncdhw(input_tv, gid * K + k);
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_ncdhw)]);
        max             = max > val ? max : val;
        vals[k]         = val;
    }

    FLOAT_ACCUM logsum = static_cast<FLOAT_ACCUM>(0.0);
    for(int64_t k = 0; k < K; k++)
    {
        logsum += expf(vals[k] - max);
    }

    tensor_layout_t<5> output_ncdhw(output_tv, gid);
    if(logsum > static_cast<FLOAT_ACCUM>(0.0))
        output[output_tv.get_tensor_view_idx(output_ncdhw)] = CVT_ACCUM2FLOAT(max + logf(logsum));
    else
        output[output_tv.get_tensor_view_idx(output_ncdhw)] =
            CVT_ACCUM2FLOAT(max + std::numeric_limits<FLOAT_ACCUM>::lowest());
}

template <typename T>
__device__ void LogsumexpBackwardImpl(const T* __restrict__ input,
                                      T* __restrict__ input_grad,
                                      const T* __restrict__ output,
                                      const T* __restrict__ output_grad,
                                      dims_5d_t selection_info,
                                      int64_t N,
                                      tensor_view_t<5> input_tv,
                                      tensor_view_t<5> input_grad_tv,
                                      tensor_view_t<5> output_tv,
                                      tensor_view_t<5> output_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= N)
        return;

    tensor_layout_t<5> input_ncdhw(input_tv, gid);
    tensor_layout_t<5> output_ncdhw(input_tv, gid);
    if(selection_info.x[0] == 1)
        output_ncdhw.layout[0] = 0;
    if(selection_info.x[1] == 1)
        output_ncdhw.layout[1] = 0;
    if(selection_info.x[2] == 1)
        output_ncdhw.layout[2] = 0;
    if(selection_info.x[3] == 1)
        output_ncdhw.layout[3] = 0;
    if(selection_info.x[4] == 1)
        output_ncdhw.layout[4] = 0;

    FLOAT_ACCUM x  = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_ncdhw)]);
    FLOAT_ACCUM y  = CVT_FLOAT2ACCUM(output[output_tv.get_tensor_view_idx(output_ncdhw)]);
    FLOAT_ACCUM dy = CVT_FLOAT2ACCUM(output_grad[output_grad_tv.get_tensor_view_idx(output_ncdhw)]);

    input_grad[input_grad_tv.get_tensor_view_idx(input_ncdhw)] =
        CVT_ACCUM2FLOAT(dy * (expf(x - y)));
}

extern "C" __global__ void LogsumexpLargeKForward(const FLOAT* __restrict__ input,
                                                  FLOAT* __restrict__ output,
                                                  int64_t N,
                                                  int64_t K,
                                                  tensor_view_t<5> input_tv,
                                                  tensor_view_t<5> output_tv)
{
    LogsumexpLargeKForwardImpl<FLOAT>(input, output, N, K, input_tv, output_tv);
}

extern "C" __global__ void LogsumexpSmallKForward(const FLOAT* __restrict__ input,
                                                  FLOAT* __restrict__ output,
                                                  int64_t N,
                                                  int64_t K,
                                                  tensor_view_t<5> input_tv,
                                                  tensor_view_t<5> output_tv)
{
    LogsumexpSmallKForwardImpl<FLOAT>(input, output, N, K, input_tv, output_tv);
}

extern "C" __global__ void LogsumexpBackward(const FLOAT* __restrict__ input,
                                             FLOAT* __restrict__ input_grad,
                                             const FLOAT* __restrict__ output,
                                             const FLOAT* __restrict__ output_grad,
                                             dims_5d_t selection_info,
                                             int64_t N,
                                             tensor_view_t<5> input_tv,
                                             tensor_view_t<5> input_grad_tv,
                                             tensor_view_t<5> output_tv,
                                             tensor_view_t<5> output_grad_tv)
{
    LogsumexpBackwardImpl<FLOAT>(input,
                                 input_grad,
                                 output,
                                 output_grad,
                                 selection_info,
                                 N,
                                 input_tv,
                                 input_grad_tv,
                                 output_tv,
                                 output_grad_tv);
}