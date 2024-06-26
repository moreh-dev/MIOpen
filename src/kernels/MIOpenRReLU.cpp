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

using prngStates = struct prngStates
{
    // Xorshift values (160 bits)
    uint x;
    uint y;
    uint z;
    uint w;
    uint v;

    // Weyl sequence value
    uint d;
};

__device__ uint xorwow_lite_next(prngStates* cur_state)
{
    const uint t = cur_state->x ^ (cur_state->x >> 2);
    cur_state->x = cur_state->y;
    cur_state->y = cur_state->z;
    cur_state->z = cur_state->w;
    cur_state->w = cur_state->v;
    cur_state->v = (cur_state->v ^ (cur_state->v << 4)) ^ (t ^ (t << 1));

    cur_state->d += 362437;

    return cur_state->d + cur_state->v;
}

#define ROCRAND_2POW32_INV (2.3283064e-10f)

__device__ float uniform_distribution(uint v)
{
    return ROCRAND_2POW32_INV + (v * ROCRAND_2POW32_INV);
}

template <typename TI, typename TO>
__device__ void RReLUForward(const prngStates* __restrict__ states,
                             const size_t num_states,
                             const TI* __restrict__ input,
                             TO* __restrict__ output,
                             float* __restrict__ noise,
                             const float lower,
                             const float upper,
                             const size_t N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    prngStates curState = states[gid % num_states];

    for(int i = gid; i < N; i += gridDim.x * blockDim.x)
    {
        FLOAT_ACCUM x = CVT_FLOAT2ACCUM(input[i]);
        float alpha   = 1.0f;
        if(x < 0.0f)
            // This part is copied from Dropout operation
            alpha = uniform_distribution(xorwow_lite_next(&curState)) * (upper - lower) + lower;

        output[i] = CVT_ACCUM2FLOAT(alpha * x);
        if(noise)
            noise[i] = alpha;
    }
}

extern "C" __global__ void RReLUForward(const prngStates* __restrict__ states,
                                        const size_t num_states,
                                        const INPUT_TYPE* __restrict__ input,
                                        OUTPUT_TYPE* __restrict__ output,
                                        float* __restrict__ noise,
                                        const float lower,
                                        const float upper,
                                        const size_t N)
{
    // instantiate the kernel
    RReLUForward<INPUT_TYPE, OUTPUT_TYPE>(
        states, num_states, input, output, noise, lower, upper, N);
}

template <typename TI, typename TO>
__device__ void
RReLUBackwardContiguous(const float* noise, const TO* doutput, TI* dinput, const size_t N)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;
    dinput[gid] = CVT_ACCUM2FLOAT(CVT_FLOAT2ACCUM(doutput[gid]) / noise[gid]);
}

extern "C" __global__ void RReLUBackwardContiguous(const float* __restrict__ noise,
                                                   const OUTPUT_TYPE* __restrict__ doutput,
                                                   INPUT_TYPE* __restrict__ dinput,
                                                   const size_t N)
{
    // instantiate the kernel
    RReLUBackwardContiguous<INPUT_TYPE, OUTPUT_TYPE>(noise, doutput, dinput, N);
}

template <typename TI, typename TO, unsigned NDIMS>
__device__ void RReLUBackwardNd(const float* __restrict__ noise,
                                const TO* __restrict__ doutput,
                                TI* __restrict__ dinput,
                                const tensor_view_t<NDIMS> doutput_tv,
                                const tensor_view_t<NDIMS> dinput_tv)
{
    int gid     = blockIdx.x * blockDim.x + threadIdx.x;
    auto layout = tensor_layout_t<NDIMS>(dinput_tv, gid);
    if(layout.layout[0] >= dinput_tv.size[0])
        return;

    auto dOidx    = doutput_tv.get_tensor_view_idx(layout);
    auto dIidx    = dinput_tv.get_tensor_view_idx(layout);
    dinput[dIidx] = CVT_ACCUM2FLOAT(CVT_FLOAT2ACCUM(doutput[dOidx]) / noise[gid]);
}

extern "C" __global__ void RReLUBackwardNd(const float* __restrict__ noise,
                                           const OUTPUT_TYPE* __restrict__ doutput,
                                           INPUT_TYPE* __restrict__ dinput,
                                           const tensor_view_t<VIEW_DIMS> doutput_tv,
                                           const tensor_view_t<VIEW_DIMS> dinput_tv)
{
    // instantiate the kernel
    RReLUBackwardNd<INPUT_TYPE, OUTPUT_TYPE, VIEW_DIMS>(
        noise, doutput, dinput, doutput_tv, dinput_tv);
}
