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
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view_5d.hpp"

#ifndef IN_OUT_TYPE
#define IN_OUT_TYPE float
#endif

#ifndef TARGET_TYPE
#define TARGET_TYPE int
#endif

#ifndef REDUCE_SIZE
#define REDUCE_SIZE 256
#endif

template <typename TIO, typename TT>
__device__ void hingeEmbeddingLossUnreducedFwd(
    const TIO* I, TT* T, TIO* O, float margin, tensor_view_5d_t I_tv, tensor_view_5d_t T_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t n[5];
    GET_NCDHW(n[0], n[1], n[2], n[3], n[4], gid, I_tv);

    if(n[0] >= I_tv.size[0])
        return;

    TIO i = TV_5D_AT(I, n[0], n[1], n[2], n[3], n[4]);
    TT t  = TV_5D_AT(T, n[0], n[1], n[2], n[3], n[4]);

    FLOAT_ACCUM loss;

    if(t == 1)
        loss = CVT_FLOAT2ACCUM(i);
    else
        loss = fmaxf(0.0f, margin - CVT_FLOAT2ACCUM(i));

    O[gid] = CVT_ACCUM2FLOAT(loss);
}

extern "C" __global__ void HingeEmbeddingLossUnreducedFwd(const IN_OUT_TYPE* I,
                                                          TARGET_TYPE* T,
                                                          IN_OUT_TYPE* O,
                                                          float margin,
                                                          tensor_view_5d_t I_tv,
                                                          tensor_view_5d_t T_tv)
{
    hingeEmbeddingLossUnreducedFwd<IN_OUT_TYPE, TARGET_TYPE>(I, T, O, margin, I_tv, T_tv);
}

template <typename TIO, typename TT>
__device__ void hingeEmbeddingLossUnreducedBwd(const TIO* I,
                                               const TT* T,
                                               const TIO* dO,
                                               TIO* dI,
                                               float margin,
                                               tensor_view_5d_t I_tv,
                                               tensor_view_5d_t T_tv,
                                               tensor_view_5d_t dO_tv,
                                               tensor_view_5d_t dI_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t n[5];
    GET_NCDHW(n[0], n[1], n[2], n[3], n[4], gid, I_tv);

    if(n[0] >= I_tv.size[0])
        return;

    TIO i = TV_5D_AT(I, n[0], n[1], n[2], n[3], n[4]);
    TT t  = TV_5D_AT(T, n[0], n[1], n[2], n[3], n[4]);

    if(t == 1)
    {
        dI[gid] = TV_5D_AT(dO, n[0], n[1], n[2], n[3], n[4]);
    }
    else
    {
        if(margin - CVT_FLOAT2ACCUM(i) > 0)
            dI[gid] = CVT_ACCUM2FLOAT(-CVT_FLOAT2ACCUM(TV_5D_AT(dO, n[0], n[1], n[2], n[3], n[4])));
        else
            dI[gid] = TIO(0);
    }
}

extern "C" __global__ void HingeEmbeddingLossUnreducedBwd(const IN_OUT_TYPE* I,
                                                          TARGET_TYPE* T,
                                                          IN_OUT_TYPE* dO,
                                                          IN_OUT_TYPE* dI,
                                                          float margin,
                                                          tensor_view_5d_t I_tv,
                                                          tensor_view_5d_t T_tv,
                                                          tensor_view_5d_t dO_tv,
                                                          tensor_view_5d_t dI_tv)
{
    hingeEmbeddingLossUnreducedBwd<IN_OUT_TYPE, TARGET_TYPE>(
        I, T, dO, dI, margin, I_tv, T_tv, dO_tv, dI_tv);
}
