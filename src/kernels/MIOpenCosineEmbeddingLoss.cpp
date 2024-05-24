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
#include "tensor_view.hpp"

#ifndef INPUT_TYPE
#define INPUT_TYPE float
#endif

#ifndef OUTPUT_TYPE
#define OUTPUT_TYPE float
#endif

#ifndef D_TYPE
#define D_TYPE float
#endif

#ifndef REDUCE_SIZE
#define REDUCE_SIZE 256
#endif

__device__ FLOAT_ACCUM warpReduceSum(FLOAT_ACCUM val)
{
    if(warpSize >= 64)
        val += __shfl_down(val, 32);
    if(warpSize >= 32)
        val += __shfl_down(val, 16);
    if(warpSize >= 16)
        val += __shfl_down(val, 8);
    if(warpSize >= 8)
        val += __shfl_down(val, 4);
    if(warpSize >= 4)
        val += __shfl_down(val, 2);
    if(warpSize >= 2)
        val += __shfl_down(val, 1);
    return val;
}

__device__ FLOAT_ACCUM blockReduceSum(FLOAT_ACCUM val)
{
    static __shared__ FLOAT_ACCUM shared[REDUCE_SIZE / warpSize];
    auto lane = threadIdx.x % warpSize;
    auto wid  = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if(lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = threadIdx.x < REDUCE_SIZE / warpSize ? shared[lane] : CVT_FP32_2ACCUM(0.0f);
    if(wid == 0)
        val = warpReduceSum(val);

    return val;
}

template <typename DTYPE>
__device__ void lossSum(const DTYPE* input, DTYPE* output, size_t N)
{
    auto gid = blockIdx.x * blockDim.x + threadIdx.x;

    FLOAT_ACCUM val = gid < N ? CVT_FLOAT2ACCUM(input[gid]) : CVT_FP32_2ACCUM(0.0f);
    val             = blockReduceSum(val);

    if(threadIdx.x == 0)
        output[blockIdx.x] = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void
LossSum(const D_TYPE* __restrict__ input, D_TYPE* __restrict__ output, size_t N)
{
    lossSum<D_TYPE>(input, output, N);
}

template <typename TI, typename TO>
__device__ void cosineembeddinglossUnreducedForward2d(const TI* __restrict__ input1,
                                                      const TI* __restrict__ input2,
                                                      const int32_t* __restrict__ target,
                                                      TO* __restrict__ output,
                                                      float margin,
                                                      tensor_view_2d_t input1_tv,
                                                      tensor_view_2d_t input2_tv,
                                                      tensor_view_1d_t target_tv,
                                                      tensor_view_1d_t output_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t N = input1_tv.size[0], D = input1_tv.size[1];
    size_t n = gid;
    if(!(n < input1_tv.size[0]))
        return;

    FLOAT_ACCUM loss     = 0.0f;
    size_t Tidx          = TV1D_IDX(target_tv, n);
    int32_t t            = target[Tidx];
    FLOAT_ACCUM cos_term = 0.0f;
    FLOAT_ACCUM norm1 = 0.0f, norm2 = 0.0f;
    for(size_t d = 0; d < D; d++)
    {
        size_t I1idx = TV2D_IDX(input1_tv, n, d);
        size_t I2idx = TV2D_IDX(input2_tv, n, d);
        cos_term += CVT_FLOAT2ACCUM(input1[I1idx]) * CVT_FLOAT2ACCUM(input2[I2idx]);
        norm1 += CVT_FLOAT2ACCUM(input1[I1idx]) * CVT_FLOAT2ACCUM(input1[I1idx]);
        norm2 += CVT_FLOAT2ACCUM(input2[I2idx]) * CVT_FLOAT2ACCUM(input2[I2idx]);
    }
    norm1 = sqrt(norm1);
    norm2 = sqrt(norm2);
    cos_term /= norm1 * norm2;

    if(t == 1)
        loss = 1.0f - cos_term;
    else
        loss = max(0.0f, cos_term - margin);

    size_t Oidx  = TV1D_IDX(output_tv, n);
    output[Oidx] = CVT_ACCUM2FLOAT(loss);
}

extern "C" __global__ void
CosineEmbeddingLossUnreducedForward2d(const INPUT_TYPE* __restrict__ input1,
                                      const INPUT_TYPE* __restrict__ input2,
                                      const int32_t* __restrict__ target,
                                      OUTPUT_TYPE* __restrict__ output,
                                      float margin,
                                      tensor_view_2d_t input1_tv,
                                      tensor_view_2d_t input2_tv,
                                      tensor_view_1d_t target_tv,
                                      tensor_view_1d_t output_tv)
{
    cosineembeddinglossUnreducedForward2d<INPUT_TYPE, OUTPUT_TYPE>(
        input1, input2, target, output, margin, input1_tv, input2_tv, target_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void cosineembeddinglossReducedForward2d(const TI* __restrict__ input1,
                                                    const TI* __restrict__ input2,
                                                    const int32_t* __restrict__ target,
                                                    TO* __restrict__ loss_sum,
                                                    float margin,
                                                    float divisor,
                                                    tensor_view_2d_t input1_tv,
                                                    tensor_view_2d_t input2_tv,
                                                    tensor_view_1d_t target_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t N = input1_tv.size[0], D = input1_tv.size[1];
    size_t n = gid;
    if(!(n < input1_tv.size[0]))
        return;

    FLOAT_ACCUM loss     = 0.0f;
    size_t Tidx          = TV1D_IDX(target_tv, n);
    int32_t t            = target[Tidx];
    FLOAT_ACCUM cos_term = 0.0f;
    FLOAT_ACCUM norm1 = 0.0f, norm2 = 0.0f;
    for(size_t d = 0; d < D; d++)
    {
        size_t I1idx = TV2D_IDX(input1_tv, n, d);
        size_t I2idx = TV2D_IDX(input2_tv, n, d);
        cos_term += CVT_FLOAT2ACCUM(input1[I1idx]) * CVT_FLOAT2ACCUM(input2[I2idx]);
        norm1 += CVT_FLOAT2ACCUM(input1[I1idx]) * CVT_FLOAT2ACCUM(input1[I1idx]);
        norm2 += CVT_FLOAT2ACCUM(input2[I2idx]) * CVT_FLOAT2ACCUM(input2[I2idx]);
    }
    norm1 = sqrt(norm1);
    norm2 = sqrt(norm2);
    cos_term /= norm1 * norm2;

    if(t == 1)
        loss = (1.0f - cos_term) / divisor;
    else
        loss = max(0.0f, cos_term - margin) / divisor;

    loss_sum[gid] = CVT_ACCUM2FLOAT(loss);
}

extern "C" __global__ void
CosineEmbeddingLossReducedForward2d(const INPUT_TYPE* __restrict__ input1,
                                    const INPUT_TYPE* __restrict__ input2,
                                    const int32_t* __restrict__ target,
                                    OUTPUT_TYPE* __restrict__ loss_sum,
                                    float margin,
                                    float divisor,
                                    tensor_view_2d_t input1_tv,
                                    tensor_view_2d_t input2_tv,
                                    tensor_view_1d_t target_tv)
{
    cosineembeddinglossReducedForward2d<INPUT_TYPE, OUTPUT_TYPE>(
        input1, input2, target, loss_sum, margin, divisor, input1_tv, input2_tv, target_tv);
}

template <typename TI, typename TO>
__device__ void cosineembeddinglossUnreducedBackward2d(const TI* __restrict__ input1,
                                                       const TI* __restrict__ input2,
                                                       const int32_t* __restrict__ target,
                                                       const TI* __restrict__ output_grad,
                                                       TO* __restrict__ input1_grad,
                                                       TO* __restrict__ input2_grad,
                                                       float margin,
                                                       tensor_view_2d_t input1_tv,
                                                       tensor_view_2d_t input2_tv,
                                                       tensor_view_1d_t target_tv,
                                                       tensor_view_1d_t output_grad_tv,
                                                       tensor_view_2d_t input1_grad_tv,
                                                       tensor_view_2d_t input2_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t N = input1_tv.size[0], D = input1_tv.size[1];
    size_t n = gid;

    if(!(n < input1_tv.size[0]))
        return;

    for(size_t d = 0; d < D; d++)
    {
        if(input1_grad)
            TV_2D_AT(input1_grad, n, d) = CVT_FP32_2FLOAT(0.0f);
        if(input2_grad)
            TV_2D_AT(input2_grad, n, d) = CVT_FP32_2FLOAT(0.0f);
    }

    size_t Tidx          = TV1D_IDX(target_tv, n);
    int32_t t            = target[Tidx];
    FLOAT_ACCUM cos_term = 0.0f;
    FLOAT_ACCUM norm1 = 0.0f, norm2 = 0.0f;

    for(size_t d = 0; d < D; d++)
    {
        size_t I1idx = TV2D_IDX(input1_tv, n, d);
        size_t I2idx = TV2D_IDX(input2_tv, n, d);
        cos_term += CVT_FLOAT2ACCUM(input1[I1idx]) * CVT_FLOAT2ACCUM(input2[I2idx]);
        norm1 += CVT_FLOAT2ACCUM(input1[I1idx]) * CVT_FLOAT2ACCUM(input1[I1idx]);
        norm2 += CVT_FLOAT2ACCUM(input2[I2idx]) * CVT_FLOAT2ACCUM(input2[I2idx]);
    }
    norm1 = sqrt(norm1);
    norm2 = sqrt(norm2);
    cos_term /= norm1 * norm2;

    for(size_t d = 0; d < D; d++)
    {
        size_t I1idx = TV2D_IDX(input1_tv, n, d);
        size_t I2idx = TV2D_IDX(input2_tv, n, d);

        FLOAT_ACCUM i1 = CVT_FLOAT2ACCUM(input1[I1idx]);
        FLOAT_ACCUM i2 = CVT_FLOAT2ACCUM(input2[I2idx]);

        if(t == 1)
        {
            if(input1_grad)
            {
                size_t IG1idx = TV2D_IDX(input1_grad_tv, n, d);
                input1_grad[IG1idx] +=
                    CVT_ACCUM2FLOAT(-(i2 / (norm1 * norm2) - cos_term * i1 / (norm1 * norm1)));
            }
            if(input2_grad)
            {
                size_t IG2idx = TV2D_IDX(input2_grad_tv, n, d);
                input2_grad[IG2idx] +=
                    CVT_ACCUM2FLOAT(-(i1 / (norm1 * norm2) - cos_term * i2 / (norm2 * norm2)));
            }
        }
        else
        {
            if(cos_term - margin < 0.0f)
                continue;
            if(input1_grad)
            {
                size_t IG1idx = TV2D_IDX(input1_grad_tv, n, d);
                input1_grad[IG1idx] +=
                    CVT_ACCUM2FLOAT(i2 / (norm1 * norm2) - cos_term * i1 / (norm1 * norm1));
            }
            if(input2_grad)
            {
                size_t IG2idx = TV2D_IDX(input2_grad_tv, n, d);
                input2_grad[IG2idx] +=
                    CVT_ACCUM2FLOAT(i1 / (norm1 * norm2) - cos_term * i2 / (norm2 * norm2));
            }
        }
    }

    FLOAT_ACCUM og = TV_1D_AT(dO, n);
    for(size_t d = 0; d < D; d++)
    {
        if(input1_grad)
        {
            size_t IG1idx = TV2D_IDX(input1_grad_tv, n, d);
            input1_grad[IG1idx] *= og;
        }
        if(input2_grad)
        {
            size_t IG2idx = TV2D_IDX(input2_grad_tv, n, d);
            input2_grad[IG2idx] *= og;
        }
    }
}

extern "C" __global__ void
CosineEmbeddingLossUnreducedBackward2d(const INPUT_TYPE* __restrict__ input1,
                                       const INPUT_TYPE* __restrict__ input2,
                                       const int32_t* __restrict__ target,
                                       const INPUT_TYPE* __restrict__ output_grad,
                                       OUTPUT_TYPE* __restrict__ input1_grad,
                                       OUTPUT_TYPE* __restrict__ input2_grad,
                                       float margin,
                                       tensor_view_2d_t input1_tv,
                                       tensor_view_2d_t input2_tv,
                                       tensor_view_1d_t target_tv,
                                       tensor_view_1d_t output_grad_tv,
                                       tensor_view_2d_t input1_grad_tv,
                                       tensor_view_2d_t input2_grad_tv)
{
    cosineembeddinglossUnreducedBackward2d<INPUT_TYPE, OUTPUT_TYPE>(input1,
                                                                    input2,
                                                                    target,
                                                                    output_grad,
                                                                    input1_grad,
                                                                    input2_grad,
                                                                    margin,
                                                                    input1_tv,
                                                                    input2_tv,
                                                                    target_tv,
                                                                    output_grad_tv,
                                                                    input1_grad_tv,
                                                                    input2_grad_tv);
}

template <typename TI, typename TO>
__device__ void cosineembeddinglossReducedBackward2d(const TI* __restrict__ input1,
                                                     const TI* __restrict__ input2,
                                                     const int32_t* __restrict__ target,
                                                     const TI* __restrict__ output_grad,
                                                     TO* __restrict__ input1_grad,
                                                     TO* __restrict__ input2_grad,
                                                     float margin,
                                                     float divisor,
                                                     tensor_view_2d_t input1_tv,
                                                     tensor_view_2d_t input2_tv,
                                                     tensor_view_1d_t target_tv,
                                                     tensor_view_1d_t output_grad_tv,
                                                     tensor_view_2d_t input1_grad_tv,
                                                     tensor_view_2d_t input2_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
}

extern "C" __global__ void
CosineEmbeddingLossReducedBackward2d(const INPUT_TYPE* __restrict__ input1,
                                     const INPUT_TYPE* __restrict__ input2,
                                     const int32_t* __restrict__ target,
                                     const INPUT_TYPE* __restrict__ output_grad,
                                     OUTPUT_TYPE* __restrict__ input1_grad,
                                     OUTPUT_TYPE* __restrict__ input2_grad,
                                     float margin,
                                     float divisor,
                                     tensor_view_2d_t input1_tv,
                                     tensor_view_2d_t input2_tv,
                                     tensor_view_1d_t target_tv,
                                     tensor_view_1d_t output_grad_tv,
                                     tensor_view_2d_t input1_grad_tv,
                                     tensor_view_2d_t input2_grad_tv)
{
    cosineembeddinglossReducedBackward2d<INPUT_TYPE, OUTPUT_TYPE>(input1,
                                                                  input2,
                                                                  target,
                                                                  output_grad,
                                                                  input1_grad,
                                                                  input2_grad,
                                                                  margin,
                                                                  divisor,
                                                                  input1_tv,
                                                                  input2_tv,
                                                                  target_tv,
                                                                  output_grad_tv,
                                                                  input1_grad_tv,
                                                                  input2_grad_tv);
}
