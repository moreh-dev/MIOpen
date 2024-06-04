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

#ifndef IN_OUT_TYPE
#define IN_OUT_TYPE float
#endif

#ifndef TARGET_TYPE
#define TARGET_TYPE int
#endif

template <typename TIO, typename TT>
__device__ void multilabelMarginLossForward2d(const TIO* __restrict__ I,
                                                const TT* __restrict__ T,
                                                TIO* __restrict__ lsum,
                                                char * ws,
                                                long ws_offset,
                                                const float divisor,
                                                const size_t I_size_0,
                                                const size_t I_size_1,
                                                const size_t T_size_0,
                                                const size_t T_size_1,
                                                const size_t I_stride_0,
                                                const size_t I_stride_1,
                                                const size_t T_stride_0,
                                                const size_t T_stride_1)
{
    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t N = I_size_0, C = I_size_1;
    size_t n = gid;

    if (!(n < I_size_0)) return;

    ws = ws + ws_offset;
    for (size_t c = 0; c < C; c++) 
    {
        ws[n * C + c] = 0;
    }

    for (size_t c = 0; c < C; c++) 
    {
        int is_target_idx = 0;
        for (size_t i = 0; i < C; i++)
        {
            size_t T_at_n_i = T[I_stride_1 * i + T_stride_0 * n];
            if (T_at_n_i == -1) break;
            if (T_at_n_i == c) 
            {
                is_target_idx = 1;
                break;
            }
        }
        if (is_target_idx)
        {
            ws[n * C + c] = 1;
        }
    }

    FLOAT_ACCUM loss = CVT_FLOAT2ACCUM(0.0f);

    for (size_t ct = 0; ct < C; ct++)
    {
        size_t T_at_n_ct = T[T_stride_1 * ct + T_stride_0 * n];
        if (T_at_n_ct == -1) break;
        for (size_t ci = 0; ci < C; ci++)
        {
            if (ws[n * C + ci] == 0)
            {
                FLOAT_ACCUM t = CVT_FLOAT2ACCUM(1.0f) - CVT_FLOAT2ACCUM(I[I_stride_1 * T_at_n_ct + I_stride_0 * n]) - CVT_FLOAT2ACCUM(I[I_stride_1 * ci + I_stride_0 * n]);
                t /= C;
                loss += t >= 0 ? t : CVT_FLOAT2ACCUM(0.0f);
            }
        }
    }

    lsum[n] = CVT_ACCUM2FLOAT(loss / divisor);
}

extern "C" __global__ void MultilabelMarginLossForward2d(const IN_OUT_TYPE* __restrict__ I,
                                                        const TARGET_TYPE* __restrict__ T,
                                                        IN_OUT_TYPE* __restrict__ lsum,
                                                        char * ws,
                                                        long ws_offset,
                                                        const float divisor,
                                                        const size_t I_size_0,
                                                        const size_t I_size_1,
                                                        const size_t T_size_0,
                                                        const size_t T_size_1,
                                                        const size_t I_stride_0,
                                                        const size_t I_stride_1,
                                                        const size_t T_stride_0,
                                                        const size_t T_stride_1)
{
    multilabelMarginLossForward2d<IN_OUT_TYPE, TARGET_TYPE>(I,
                                                T,
                                                lsum,
                                                ws,
                                                ws_offset,
                                                divisor,
                                                I_size_0,
                                                I_size_1,
                                                T_size_0,
                                                T_size_1,
                                                I_stride_0,
                                                I_stride_1,
                                                T_stride_0,
                                                T_stride_1);
}
