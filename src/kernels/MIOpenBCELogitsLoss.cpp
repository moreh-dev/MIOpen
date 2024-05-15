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

template <typename TI, typename TO>
__device__ void BCELogitsLossReducedForward5d(const TI* I,
                                              const TI* T,
                                              const TI* W,
                                              const TI* PW,
                                              TO* lsum,
                                              const float divisor,
                                              tensor_view_5d_t I_tv,
                                              tensor_view_5d_t T_tv,
                                              tensor_view_5d_t W_tv,
                                              tensor_view_5d_t PW_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t n[5];
    GET_NCDHW(n[0], n[1], n[2], n[3], n[4], gid, I_tv);

    if(n[0] >= I_tv.size[0])
        return;

    size_t c = gid % PW_tv.size[0];

    FLOAT_ACCUM w = (W == nullptr) ? static_cast<FLOAT_ACCUM>(1.0f)
                                   : CVT_FLOAT2ACCUM(TV_5D_AT(W, n[0], n[1], n[2], n[3], n[4]));
    FLOAT_ACCUM pw =
        (PW == nullptr) ? static_cast<FLOAT_ACCUM>(1.0f) : CVT_FLOAT2ACCUM(TV_1D_AT(PW, c));

    FLOAT_ACCUM x = CVT_FLOAT2ACCUM(TV_5D_AT(I, n[0], n[1], n[2], n[3], n[4]));
    FLOAT_ACCUM y = CVT_FLOAT2ACCUM(TV_5D_AT(T, n[0], n[1], n[2], n[3], n[4]));

    FLOAT_ACCUM max_val;
    max_val = (x < 0) ? -x : static_cast<FLOAT_ACCUM>(0.0f);

    FLOAT_ACCUM loss =
        w * (((1.0f - y) * x) +
             (1 + (pw - 1) * y) * (log(exp(-max_val) + exp(-x - max_val)) + max_val));
    lsum[gid] = CVT_ACCUM2FLOAT(loss / divisor);
}

extern "C" __global__ void BCELogitsLossReducedForward5d(const INPUT_TYPE* __restrict__ I,
                                                         const INPUT_TYPE* __restrict__ T,
                                                         const INPUT_TYPE* __restrict__ W,
                                                         const INPUT_TYPE* __restrict__ PW,
                                                         OUTPUT_TYPE* __restrict__ lsum,
                                                         const float divisor,
                                                         tensor_view_5d_t I_tv,
                                                         tensor_view_5d_t T_tv,
                                                         tensor_view_5d_t W_tv,
                                                         tensor_view_5d_t PW_tv)
{
    // instantiate the kernel
    BCELogitsLossReducedForward5d<INPUT_TYPE, OUTPUT_TYPE>(
        I, T, W, PW, lsum, divisor, I_tv, T_tv, W_tv, PW_tv);
}

__device__ FLOAT_ACCUM warp_reduce_sum(FLOAT_ACCUM val)
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

__device__ FLOAT_ACCUM block_reduce_sum(FLOAT_ACCUM val)
{
    static __shared__ FLOAT_ACCUM shared[REDUCE_SIZE / warpSize];
    auto lane = threadIdx.x % warpSize;
    auto wid  = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);

    if(lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = threadIdx.x < REDUCE_SIZE / warpSize ? shared[lane] : 0;
    if(wid == 0)
        val = warp_reduce_sum(val);

    return val;
}

template <typename DTYPE>
__device__ void losssum(const DTYPE* input, DTYPE* output, size_t N)
{
    auto gid = blockIdx.x * blockDim.x + threadIdx.x;

    FLOAT_ACCUM val = gid < N ? CVT_FLOAT2ACCUM(input[gid]) : static_cast<FLOAT_ACCUM>(0.0f);
    val             = block_reduce_sum(val);

    if(threadIdx.x == 0)
        output[blockIdx.x] = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void
LossSum(const D_TYPE* __restrict__ input, D_TYPE* __restrict__ output, size_t N)
{
    // instantiate the kernel
    losssum<D_TYPE>(input, output, N);
}

__device__ FLOAT_ACCUM sigmoid(FLOAT_ACCUM x) { return 1.0f / (1.0f + exp(-x)); }

template <typename TI, typename TO>
__device__ void BCELogitsLossReducedBackward5d(const TI* I,
                                               const TI* T,
                                               const TI* W,
                                               const TI* PW,
                                               const TO* dO,
                                               TI* dI,
                                               TI* dT,
                                               float divisor,
                                               tensor_view_5d_t I_tv,
                                               tensor_view_5d_t T_tv,
                                               tensor_view_5d_t W_tv,
                                               tensor_view_5d_t PW_tv,
                                               tensor_view_5d_t dI_tv,
                                               tensor_view_5d_t dT_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t n[5];
    GET_NCDHW(n[0], n[1], n[2], n[3], n[4], gid, I_tv);

    if(n[0] >= I_tv.size[0])
        return;

    size_t c = gid % PW_tv.size[0];

    FLOAT_ACCUM w = (W == nullptr) ? static_cast<FLOAT_ACCUM>(1.0f)
                                   : CVT_FLOAT2ACCUM(TV_5D_AT(W, n[0], n[1], n[2], n[3], n[4]));
    FLOAT_ACCUM pw =
        (PW == nullptr) ? static_cast<FLOAT_ACCUM>(1.0f) : CVT_FLOAT2ACCUM(TV_1D_AT(PW, c));

    FLOAT_ACCUM x = CVT_FLOAT2ACCUM(TV_5D_AT(I, n[0], n[1], n[2], n[3], n[4]));
    FLOAT_ACCUM y = CVT_FLOAT2ACCUM(TV_5D_AT(T, n[0], n[1], n[2], n[3], n[4]));

    if(dI)
    {
        size_t dIidx       = TV5D_IDX(dI_tv, n[0], n[1], n[2], n[3], n[4]);
        FLOAT_ACCUM result = -w * (pw * y * (1.0f - sigmoid(x)) + (y - 1.0f) * sigmoid(x));
        result *= CVT_FLOAT2ACCUM(dO[0]) / divisor;
        dI[dIidx] = CVT_ACCUM2FLOAT(result);
    }

    if(dT)
    {
        size_t dTidx       = TV5D_IDX(dT_tv, n[0], n[1], n[2], n[3], n[4]);
        FLOAT_ACCUM result = w * (log(1.0f - sigmoid(x)) - pw * log(sigmoid(x)));
        result *= CVT_FLOAT2ACCUM(dO[0]) / divisor;
        dT[dTidx] = CVT_ACCUM2FLOAT(result);
    }
}

extern "C" __global__ void BCELogitsLossReducedBackward5d(const INPUT_TYPE* I,
                                                          const INPUT_TYPE* T,
                                                          const INPUT_TYPE* W,
                                                          const INPUT_TYPE* PW,
                                                          const OUTPUT_TYPE* dO,
                                                          INPUT_TYPE* dI,
                                                          INPUT_TYPE* dT,
                                                          float divisor,
                                                          tensor_view_5d_t I_tv,
                                                          tensor_view_5d_t T_tv,
                                                          tensor_view_5d_t W_tv,
                                                          tensor_view_5d_t PW_tv,
                                                          tensor_view_5d_t dI_tv,
                                                          tensor_view_5d_t dT_tv)
{
    BCELogitsLossReducedBackward5d<INPUT_TYPE, OUTPUT_TYPE>(
        I, T, W, PW, dO, dI, dT, divisor, I_tv, T_tv, W_tv, PW_tv, dI_tv, dT_tv);
}
