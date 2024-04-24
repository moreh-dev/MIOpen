/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#if MIOPEN_USE_BFP16 == 1
#define CVT_FLOAT2ACCUM(x) (bfloat16_to_float(x))
#define CVT_ACCUM2FLOAT(x) (float_to_bfloat16(x))
#define CVT_INTEGRAL2ACCUM(x) ((_FLOAT_ACCUM)(x))
#define CVT_FP32_2FLOAT(x) (CVT_ACCUM2FLOAT(x))
#define CVT_FP32_2ACCUM(x) (x)
#endif

extern "C" __global__ void SGDFwdContiguous(const FLOAT* __restrict__ param_in,
                                            FLOAT* __restrict__ param_out,
                                            const FLOAT* __restrict__ grad,
                                            const FLOAT* __restrict__ momentum_buffer_in,
                                            FLOAT* __restrict__ momentum_buffer_out,
                                            double lr,
                                            double momentum,
                                            double dampening,
                                            double weight_decay,
                                            char nesterov,
                                            char momentum_initialized,
                                            size_t param_size)
{
    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= param_size)
        return;

    FLOAT_ACCUM param = CVT_FLOAT2ACCUM(param_in[gid]);
    FLOAT_ACCUM d_p   = CVT_FLOAT2ACCUM(grad[gid]);

    if(weight_decay != 0)
    {
        d_p += param * CVT_FLOAT2ACCUM(weight_decay);
    }

    if(momentum != 0)
    {
        FLOAT_ACCUM momentum_v;
        if(momentum_initialized)
        {
            momentum_v = CVT_FLOAT2ACCUM(momentum_buffer_in[gid]);
            momentum_v =
                momentum_v * CVT_FLOAT2ACCUM(momentum) + d_p * CVT_FLOAT2ACCUM(1 - dampening);
        }
        else
        {
            momentum_v = d_p;
        }
        momentum_buffer_out[gid] = CVT_FLOAT2ACCUM(momentum_v);

        if(nesterov)
        {
            d_p = d_p + momentum_v * CVT_FLOAT2ACCUM(momentum);
        }
        else
        {
            d_p = momentum_v;
        }
    }

    param_out[gid] = CVT_ACCUM2FLOAT(param - CVT_FLOAT2ACCUM(lr) * d_p);
}
