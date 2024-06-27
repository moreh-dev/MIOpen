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

extern "C" __global__ void MarginRankingLossReducedForward5d(const FLOAT* __restrict__ I1,
                                                             const FLOAT* __restrict__ I2,
                                                             const FLOAT* __restrict__ T,
                                                             FLOAT* __restrict__ O,
                                                             float margin,
                                                             float divisor,
                                                             tensor_view_5d_t I1_tv,
                                                             tensor_view_5d_t I2_tv,
                                                             tensor_view_5d_t T_tv,
                                                             tensor_view_5d_t O_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t n0123 = gid / I1_tv.size[4], n4 = gid % I1_tv.size[4];
    size_t n012 = n0123 / I1_tv.size[3], n3 = n0123 % I1_tv.size[3];
    size_t n01 = n012 / I1_tv.size[2], n2 = n012 % I1_tv.size[2];
    size_t n0 = n01 / I1_tv.size[1], n1 = n01 % I1_tv.size[1];

    if(!(n0 < I1_tv.size[0]))
        return;

    size_t I1idx = TV5D_IDX(I1_tv, n0, n1, n2, n3, n4);
    size_t I2idx = TV5D_IDX(I2_tv, n0, n1, n2, n3, n4);
    size_t Tidx  = TV5D_IDX(T_tv, n0, n1, n2, n3, n4);
    size_t Oidx  = TV5D_IDX(O_tv, n0, n1, n2, n3, n4);

    FLOAT_ACCUM output_accum =
        -static_cast<FLOAT_ACCUM>(T[Tidx]) *
            (static_cast<FLOAT_ACCUM>(I1[I1idx]) - static_cast<FLOAT_ACCUM>(I2[I2idx])) +
        FLOAT_ACCUM(margin);
    if(output_accum < 0)
        output_accum = 0;
    O[Oidx] = CVT_ACCUM2FLOAT(output_accum / FLOAT_ACCUM(divisor));
}

extern "C" __global__ void MarginRankingLossReducedBackward5d(const FLOAT* __restrict__ I1,
                                                              const FLOAT* __restrict__ I2,
                                                              const FLOAT* __restrict__ T,
                                                              const FLOAT* __restrict__ dO,
                                                              FLOAT* __restrict__ dI1,
                                                              FLOAT* __restrict__ dI2,
                                                              float margin,
                                                              float divisor,
                                                              tensor_view_5d_t I1_tv,
                                                              tensor_view_5d_t I2_tv,
                                                              tensor_view_5d_t T_tv,
                                                              tensor_view_5d_t dO_tv,
                                                              tensor_view_5d_t dI1_tv,
                                                              tensor_view_5d_t dI2_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t n0123 = gid / I1_tv.size[4], n4 = gid % I1_tv.size[4];
    size_t n012 = n0123 / I1_tv.size[3], n3 = n0123 % I1_tv.size[3];
    size_t n01 = n012 / I1_tv.size[2], n2 = n012 % I1_tv.size[2];
    size_t n0 = n01 / I1_tv.size[1], n1 = n01 % I1_tv.size[1];

    if(!(n0 < I1_tv.size[0]))
        return;

    size_t I1idx  = TV5D_IDX(I1_tv, n0, n1, n2, n3, n4);
    size_t I2idx  = TV5D_IDX(I2_tv, n0, n1, n2, n3, n4);
    size_t dI1idx = TV5D_IDX(dI1_tv, n0, n1, n2, n3, n4);
    size_t dI2idx = TV5D_IDX(dI2_tv, n0, n1, n2, n3, n4);
    size_t Tidx   = TV5D_IDX(T_tv, n0, n1, n2, n3, n4);
    size_t dOidx  = TV5D_IDX(dO_tv, n0, n1, n2, n3, n4);

    FLOAT_ACCUM t = -static_cast<FLOAT_ACCUM>(T[Tidx]) * (static_cast<FLOAT_ACCUM>(I1[I1idx]) -
                                                          static_cast<FLOAT_ACCUM>(I2[I2idx])) +
                    FLOAT_ACCUM(margin);

    if(t < 0)
    {
        dI1[dI1idx] = 0.0f;
        dI2[dI2idx] = 0.0f;
    }
    else
    {
        FLOAT_ACCUM d_accum = static_cast<FLOAT_ACCUM>(T[Tidx]) *
                              static_cast<FLOAT_ACCUM>(dO[dOidx]) / FLOAT_ACCUM(divisor);
        dI1[dI1idx] = CVT_ACCUM2FLOAT(-d_accum);
        dI2[dI2idx] = CVT_ACCUM2FLOAT(d_accum);
    }
}

extern "C" __global__ void MarginRankingLossUnreducedForward5d(const FLOAT* __restrict__ I1,
                                                               const FLOAT* __restrict__ I2,
                                                               const FLOAT* __restrict__ T,
                                                               FLOAT* __restrict__ O,
                                                               float margin,
                                                               tensor_view_5d_t I1_tv,
                                                               tensor_view_5d_t I2_tv,
                                                               tensor_view_5d_t T_tv,
                                                               tensor_view_5d_t O_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t n0123 = gid / I1_tv.size[4], n4 = gid % I1_tv.size[4];
    size_t n012 = n0123 / I1_tv.size[3], n3 = n0123 % I1_tv.size[3];
    size_t n01 = n012 / I1_tv.size[2], n2 = n012 % I1_tv.size[2];
    size_t n0 = n01 / I1_tv.size[1], n1 = n01 % I1_tv.size[1];

    if(!(n0 < I1_tv.size[0]))
        return;

    size_t I1idx = TV5D_IDX(I1_tv, n0, n1, n2, n3, n4);
    size_t I2idx = TV5D_IDX(I2_tv, n0, n1, n2, n3, n4);
    size_t Tidx  = TV5D_IDX(T_tv, n0, n1, n2, n3, n4);
    size_t Oidx  = TV5D_IDX(O_tv, n0, n1, n2, n3, n4);

    FLOAT_ACCUM output_accum =
        -static_cast<FLOAT_ACCUM>(T[Tidx]) *
            (static_cast<FLOAT_ACCUM>(I1[I1idx]) - static_cast<FLOAT_ACCUM>(I2[I2idx])) +
        FLOAT_ACCUM(margin);
    if(output_accum < 0)
        output_accum = 0;
    O[Oidx] = CVT_ACCUM2FLOAT(output_accum);
}

extern "C" __global__ void MarginRankingLossUnreducedBackward5d(const FLOAT* __restrict__ I1,
                                                                const FLOAT* __restrict__ I2,
                                                                const FLOAT* __restrict__ T,
                                                                const FLOAT* __restrict__ dO,
                                                                FLOAT* __restrict__ dI1,
                                                                FLOAT* __restrict__ dI2,
                                                                float margin,
                                                                tensor_view_5d_t I1_tv,
                                                                tensor_view_5d_t I2_tv,
                                                                tensor_view_5d_t T_tv,
                                                                tensor_view_5d_t dO_tv,
                                                                tensor_view_5d_t dI1_tv,
                                                                tensor_view_5d_t dI2_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t n0123 = gid / I1_tv.size[4], n4 = gid % I1_tv.size[4];
    size_t n012 = n0123 / I1_tv.size[3], n3 = n0123 % I1_tv.size[3];
    size_t n01 = n012 / I1_tv.size[2], n2 = n012 % I1_tv.size[2];
    size_t n0 = n01 / I1_tv.size[1], n1 = n01 % I1_tv.size[1];

    if(!(n0 < I1_tv.size[0]))
        return;

    size_t I1idx  = TV5D_IDX(I1_tv, n0, n1, n2, n3, n4);
    size_t I2idx  = TV5D_IDX(I2_tv, n0, n1, n2, n3, n4);
    size_t dI1idx = TV5D_IDX(dI1_tv, n0, n1, n2, n3, n4);
    size_t dI2idx = TV5D_IDX(dI2_tv, n0, n1, n2, n3, n4);
    size_t Tidx   = TV5D_IDX(T_tv, n0, n1, n2, n3, n4);
    size_t dOidx  = TV5D_IDX(dO_tv, n0, n1, n2, n3, n4);

    FLOAT_ACCUM t = -static_cast<FLOAT_ACCUM>(T[Tidx]) * (static_cast<FLOAT_ACCUM>(I1[I1idx]) -
                                                          static_cast<FLOAT_ACCUM>(I2[I2idx])) +
                    FLOAT_ACCUM(margin);

    if(t < 0)
    {
        dI1[dI1idx] = 0.0f;
        dI2[dI2idx] = 0.0f;
    }
    else
    {
        FLOAT_ACCUM d_accum =
            static_cast<FLOAT_ACCUM>(T[Tidx]) * static_cast<FLOAT_ACCUM>(dO[dOidx]);
        dI1[dI1idx] = CVT_ACCUM2FLOAT(-d_accum);
        dI2[dI2idx] = CVT_ACCUM2FLOAT(d_accum);
    }
}