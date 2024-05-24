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

template <typename TIO, typename TT>
__device__ void hingeEmbeddingLossFwd(const TIO* input,
                                      TT* target,
                                      TIO* output,
                                      float margin,
                                      float divisor,
                                      tensor_view_5d_t input_tv,
                                      tensor_view_5d_t target_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t n[5];
    GET_NCDHW(n[0], n[1], n[2], n[3], n[4], gid, input_tv);

    if(n[0] >= input_tv.size[0])
        return;

    TIO i = TV_5D_AT(input, n[0], n[1], n[2], n[3], n[4]);
    TT t  = TV_5D_AT(target, n[0], n[1], n[2], n[3], n[4]);

    FLOAT_ACCUM loss;

    if(t == 1)
        loss = CVT_FLOAT2ACCUM(i);
    else
        loss = fmaxf(0.0f, margin - CVT_FLOAT2ACCUM(i));

    output[gid] = CVT_ACCUM2FLOAT(loss / divisor);
}

extern "C" __global__ void HingeEmbeddingLossFwd(const IN_OUT_TYPE* input,
                                                 TARGET_TYPE* target,
                                                 IN_OUT_TYPE* output,
                                                 float margin,
                                                 float divisor,
                                                 tensor_view_5d_t input_tv,
                                                 tensor_view_5d_t target_tv)
{
    hingeEmbeddingLossFwd<IN_OUT_TYPE, TARGET_TYPE>(
        input, target, output, margin, divisor, input_tv, target_tv);
}

template <typename TIO, typename TT>
__device__ void hingeEmbeddingLossBwd(const TIO* input,
                                      const TT* target,
                                      const TIO* doutput,
                                      TIO* dinput,
                                      float margin,
                                      float divisor,
                                      tensor_view_5d_t input_tv,
                                      tensor_view_5d_t target_tv,
                                      tensor_view_5d_t doutput_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t n[5];
    GET_NCDHW(n[0], n[1], n[2], n[3], n[4], gid, input_tv);

    if(n[0] >= input_tv.size[0])
        return;

    TIO i = TV_5D_AT(input, n[0], n[1], n[2], n[3], n[4]);
    TT t  = TV_5D_AT(target, n[0], n[1], n[2], n[3], n[4]);

    if(t == 1)
    {
        dinput[gid] = CVT_ACCUM2FLOAT(CVT_FLOAT2ACCUM(TV_5D_AT(doutput, 0, 0, 0, 0, 0)) / divisor);
    }
    else
    {
        if(margin - CVT_FLOAT2ACCUM(i) > 0)
            dinput[gid] =
                CVT_ACCUM2FLOAT(-CVT_FLOAT2ACCUM(TV_5D_AT(doutput, 0, 0, 0, 0, 0)) / divisor);
        else
            dinput[gid] = TIO(0);
    }
}

extern "C" __global__ void HingeEmbeddingLossBwd(const IN_OUT_TYPE* input,
                                                 TARGET_TYPE* target,
                                                 IN_OUT_TYPE* doutput,
                                                 IN_OUT_TYPE* dinput,
                                                 float margin,
                                                 float divisor,
                                                 tensor_view_5d_t input_tv,
                                                 tensor_view_5d_t target_tv,
                                                 tensor_view_5d_t doutput_tv)
{
    hingeEmbeddingLossBwd<IN_OUT_TYPE, TARGET_TYPE>(
        input, target, doutput, dinput, margin, divisor, input_tv, target_tv, doutput_tv);
}

template <typename TIO, typename TT>
__device__ void hingeEmbeddingLossUnreducedFwd(const TIO* input,
                                               TT* target,
                                               TIO* output,
                                               float margin,
                                               tensor_view_5d_t input_tv,
                                               tensor_view_5d_t target_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t n[5];
    GET_NCDHW(n[0], n[1], n[2], n[3], n[4], gid, input_tv);

    if(n[0] >= input_tv.size[0])
        return;

    TIO i = TV_5D_AT(input, n[0], n[1], n[2], n[3], n[4]);
    TT t  = TV_5D_AT(target, n[0], n[1], n[2], n[3], n[4]);

    FLOAT_ACCUM loss;

    if(t == 1)
        loss = CVT_FLOAT2ACCUM(i);
    else
        loss = fmaxf(0.0f, margin - CVT_FLOAT2ACCUM(i));

    output[gid] = CVT_ACCUM2FLOAT(loss);
}

extern "C" __global__ void HingeEmbeddingLossUnreducedFwd(const IN_OUT_TYPE* input,
                                                          TARGET_TYPE* target,
                                                          IN_OUT_TYPE* output,
                                                          float margin,
                                                          tensor_view_5d_t input_tv,
                                                          tensor_view_5d_t target_tv)
{
    hingeEmbeddingLossUnreducedFwd<IN_OUT_TYPE, TARGET_TYPE>(
        input, target, output, margin, input_tv, target_tv);
}

template <typename TIO, typename TT>
__device__ void hingeEmbeddingLossUnreducedBwd(const TIO* input,
                                               const TT* target,
                                               const TIO* doutput,
                                               TIO* dinput,
                                               float margin,
                                               tensor_view_5d_t input_tv,
                                               tensor_view_5d_t target_tv,
                                               tensor_view_5d_t doutput_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t n[5];
    GET_NCDHW(n[0], n[1], n[2], n[3], n[4], gid, input_tv);

    if(n[0] >= input_tv.size[0])
        return;

    TIO i = TV_5D_AT(input, n[0], n[1], n[2], n[3], n[4]);
    TT t  = TV_5D_AT(target, n[0], n[1], n[2], n[3], n[4]);

    if(t == 1)
    {
        dinput[gid] = TV_5D_AT(doutput, n[0], n[1], n[2], n[3], n[4]);
    }
    else
    {
        if(margin - CVT_FLOAT2ACCUM(i) > 0)
            dinput[gid] =
                CVT_ACCUM2FLOAT(-CVT_FLOAT2ACCUM(TV_5D_AT(doutput, n[0], n[1], n[2], n[3], n[4])));
        else
            dinput[gid] = TIO(0);
    }
}

extern "C" __global__ void HingeEmbeddingLossUnreducedBwd(const IN_OUT_TYPE* input,
                                                          TARGET_TYPE* target,
                                                          IN_OUT_TYPE* doutput,
                                                          IN_OUT_TYPE* dinput,
                                                          float margin,
                                                          tensor_view_5d_t input_tv,
                                                          tensor_view_5d_t target_tv,
                                                          tensor_view_5d_t doutput_tv)
{
    hingeEmbeddingLossUnreducedBwd<IN_OUT_TYPE, TARGET_TYPE>(
        input, target, doutput, dinput, margin, input_tv, target_tv, doutput_tv);
}
