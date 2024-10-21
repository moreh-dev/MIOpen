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
#include <cstddef>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "hip_atomic.hpp"
#include "tensor_view.hpp"

template <typename TIO, typename TINDEX>
__device__ void BatchedGatherV2BackwardKernel(const TIO* outputGrad,
                                              const TINDEX* indices,
                                              TIO* paramGrad,
                                              tensor_view_t<4> outputGrad_tv,
                                              size_t outer_size,
                                              size_t gather_dim_size,
                                              size_t indices_numel,
                                              size_t inner_size,
                                              size_t out_grad_numel,
                                              bool is_batch_dim_zero)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid >= out_grad_numel)
        return;

    bool is_axis_zero = (outer_size == 1);

    size_t batch_i   = 0;
    size_t outer_i   = 0;
    size_t indices_i = 0;
    size_t inner_i   = 0;

    const size_t slices_count = gid / inner_size;
    inner_i                   = gid - slices_count * inner_size;
    if(is_batch_dim_zero)
    {
        if(is_axis_zero)
        {
            indices_i = slices_count;
        }
        else
        {
            outer_i   = slices_count / indices_numel;
            indices_i = slices_count - outer_i * indices_numel;
        }
    }
    else
    {
        const size_t entries_count = slices_count / indices_numel;
        indices_i                  = slices_count - entries_count * indices_numel;
        if(is_axis_zero)
        {
            batch_i = entries_count;
        }
        else
        {
            batch_i = entries_count / outer_size;
            outer_i = entries_count - batch_i * outer_size;
        }
    }

    size_t gather_i = static_cast<size_t>(indices[batch_i * indices_numel + indices_i]);

    if(gather_i < gather_dim_size)
    {
        // paramGrad[batch_i][outer_i][gather_i][inner_i] += outputGrad[gid];
        size_t param_i =
            ((batch_i * outer_size + outer_i) * gather_dim_size + gather_i) * inner_size + inner_i;
        FLOAT_ACCUM val =
            CVT_FLOAT2ACCUM(getNDVal(outputGrad, outputGrad_tv, static_cast<uint64_t>(gid)));
        atomic_add_g(paramGrad + param_i, val);
    }
}

extern "C" __global__ void BatchedGatherV2Backward(const IO_TYPE* outputGrad,
                                                   const INDEX_TYPE* indices,
                                                   IO_TYPE* paramGrad,
                                                   tensor_view_t<4> outputGrad_tv,
                                                   size_t outer_size,
                                                   size_t gather_dim_size,
                                                   size_t indices_numel,
                                                   size_t inner_size,
                                                   size_t out_grad_numel,
                                                   bool is_batch_dim_zero)
{
    BatchedGatherV2BackwardKernel<IO_TYPE, INDEX_TYPE>(outputGrad,
                                                       indices,
                                                       paramGrad,
                                                       outputGrad_tv,
                                                       outer_size,
                                                       gather_dim_size,
                                                       indices_numel,
                                                       inner_size,
                                                       out_grad_numel,
                                                       is_batch_dim_zero);
}

template <typename TIO, typename TINDEX>
__device__ void GatherV2BackwardKernel(const TIO* outputGrad,
                                       const TINDEX* indices,
                                       TIO* paramGrad,
                                       tensor_view_t<3> outputGrad_tv,
                                       size_t gather_dim_size,
                                       size_t indices_numel,
                                       size_t inner_size,
                                       size_t out_grad_size,
                                       bool is_axis_zero)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid >= out_grad_size)
        return;

    size_t outer_i   = 0;
    size_t indices_i = 0;
    size_t inner_i   = 0;

    if(is_axis_zero)
    {
        indices_i = gid / inner_size;
        inner_i   = gid - indices_i * inner_size;
    }
    else
    {
        size_t outer_indices_i = gid / inner_size;
        outer_i                = outer_indices_i / indices_numel;
        indices_i              = outer_indices_i - outer_i * indices_numel;
        inner_i                = gid - outer_indices_i * inner_size;
    }

    size_t gather_i = indices[indices_i];

    if(gather_i < gather_dim_size)
    {
        size_t param_i = (outer_i * gather_dim_size + gather_i) * inner_size + inner_i;
        FLOAT_ACCUM val =
            CVT_FLOAT2ACCUM(getNDVal(outputGrad, outputGrad_tv, static_cast<uint64_t>(gid)));
        atomic_add_g(paramGrad + param_i, val);
    }
}

extern "C" __global__ void GatherV2Backward(const IO_TYPE* outputGrad,
                                            const INDEX_TYPE* indices,
                                            IO_TYPE* paramGrad,
                                            tensor_view_t<3> outputGrad_tv,
                                            size_t gather_dim_size,
                                            size_t indices_numel,
                                            size_t inner_size,
                                            size_t out_grad_size,
                                            bool is_axis_zero)
{
    GatherV2BackwardKernel<IO_TYPE, INDEX_TYPE>(outputGrad,
                                                indices,
                                                paramGrad,
                                                outputGrad_tv,
                                                gather_dim_size,
                                                indices_numel,
                                                inner_size,
                                                out_grad_size,
                                                is_axis_zero);
}
