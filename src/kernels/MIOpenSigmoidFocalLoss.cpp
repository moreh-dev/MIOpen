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

#ifndef CVT_ACCUM2FLOAT
#define CVT_ACCUM2FLOAT(x) (float_to_bfloat16(x))
#endif

#ifndef CVT_FLOAT2ACCUM
#define CVT_FLOAT2ACCUM(x) (bfloat16_to_float(x))
#endif

template <typename TIO>
__device__ void sigmoidFocalLossUnreducedFwd(const TIO* input,
                                             TIO* target,
                                             TIO* output,
                                             float alpha,
                                             float gamma,
                                             tensor_view_5d_t input_tv,
                                             tensor_view_5d_t target_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t n[5];
    GET_NCDHW(n[0], n[1], n[2], n[3], n[4], gid, input_tv);

    if(n[0] >= input_tv.size[0])
        return;

    FLOAT_ACCUM i = CVT_FLOAT2ACCUM(TV_5D_AT(input, n[0], n[1], n[2], n[3], n[4]));
    FLOAT_ACCUM t = CVT_FLOAT2ACCUM(TV_5D_AT(target, n[0], n[1], n[2], n[3], n[4]));

    FLOAT_ACCUM sig    = 1 / (1 + exp(-i));
    FLOAT_ACCUM ceLoss = -(t * log(sig) + (1 - t) * log(1 - sig));
    FLOAT_ACCUM sig_t  = sig * t + (1 - sig) * (1 - t);
    FLOAT_ACCUM loss   = ceLoss * pow(1 - sig_t, gamma);

    if(alpha >= 0)
    {
        FLOAT_ACCUM alpha_t = alpha * t + (1 - alpha) * (1 - t);
        loss                = alpha_t * loss;
        // printf("%f %f %f %f %f debug \n", sig, ceLoss, sig_t, loss, alpha_t);
    }

    output[gid] = CVT_ACCUM2FLOAT(loss);
}

extern "C" __global__ void SigmoidFocalLossUnreducedFwd(const IN_OUT_TYPE* input,
                                                        IN_OUT_TYPE* target,
                                                        IN_OUT_TYPE* output,
                                                        float alpha,
                                                        float gamma,
                                                        tensor_view_5d_t input_tv,
                                                        tensor_view_5d_t target_tv)
{
    sigmoidFocalLossUnreducedFwd<IN_OUT_TYPE>(
        input, target, output, alpha, gamma, input_tv, target_tv);
}
