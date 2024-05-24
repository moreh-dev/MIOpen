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

template <typename TI, typename TO>
inline __device__ void dist(const TI* I1,
                            const TI* I2,
                            TO* Dis,
                            const int p,
                            const float eps,
                            const tensor_view_t<2> I1_tv,
                            const tensor_view_t<2> I2_tv,
                            const int n,
                            const int c)
{
    int gid        = blockIdx.x * blockDim.x + threadIdx.x;
    FLOAT_ACCUM i1 = CVT_FLOAT2ACCUM(I1[I1_tv.stride[0] * n + I1_tv.stride[1] * c]);
    FLOAT_ACCUM i2 = CVT_FLOAT2ACCUM(I2[I2_tv.stride[0] * n + I2_tv.stride[1] * c]);
    Dis[gid]       = CVT_ACCUM2FLOAT(pow(fabs(i1 - i2 + eps), p));
}

template <typename TI, typename TO>
__device__ void TripletMarginLossForward2d_1(const TI* A,
                                             const TI* P,
                                             const TI* N,
                                             TO* ldist,
                                             const int p,
                                             const float eps,
                                             const tensor_view_t<2> A_tv,
                                             const tensor_view_t<2> P_tv,
                                             const tensor_view_t<2> N_tv)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int c = gid % A_tv.size[1];
    int n = gid / A_tv.size[1];

    if(n >= A_tv.size[0])
        return;

    int size = A_tv.size[0] * A_tv.size[1];
    dist<TI, TO>(A, P, ldist + 0 * size, p, eps, A_tv, P_tv, n, c);
    dist<TI, TO>(A, N, ldist + 1 * size, p, eps, A_tv, N_tv, n, c);
    dist<TI, TO>(P, N, ldist + 2 * size, p, eps, P_tv, N_tv, n, c);
}

extern "C" __global__ void TripletMarginLossForward2d_1(const INPUT_TYPE* A,
                                                        const INPUT_TYPE* P,
                                                        const INPUT_TYPE* N,
                                                        OUTPUT_TYPE* ldist,
                                                        const int p,
                                                        const float eps,
                                                        const tensor_view_t<2> A_tv,
                                                        const tensor_view_t<2> P_tv,
                                                        const tensor_view_t<2> N_tv)
{
    // instantiate the kernel
    TripletMarginLossForward2d_1<INPUT_TYPE, OUTPUT_TYPE>(A, P, N, ldist, p, eps, A_tv, P_tv, N_tv);
}

template <typename T>
__device__ void TripletMarginLossUnreducedForward2d_2(const T* ldist,
                                                      T* O,
                                                      const float margin,
                                                      const int p,
                                                      const bool swap,
                                                      const tensor_view_t<1> O_tv)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= O_tv.size[0])
        return;

    FLOAT_ACCUM ap = pow(CVT_FLOAT2ACCUM(ldist[0 * O_tv.size[0] + gid]), 1.0f / p);
    FLOAT_ACCUM an = pow(CVT_FLOAT2ACCUM(ldist[1 * O_tv.size[0] + gid]), 1.0f / p);
    FLOAT_ACCUM pn = pow(CVT_FLOAT2ACCUM(ldist[2 * O_tv.size[0] + gid]), 1.0f / p);

    if(swap && pn < an)
    {
        an = pn;
    }

    O[O_tv.stride[0] * gid] = CVT_ACCUM2FLOAT(max(ap - an + margin, 0.0f));
}

extern "C" __global__ void TripletMarginLossUnreducedForward2d_2(const D_TYPE* ldist,
                                                                 D_TYPE* O,
                                                                 const float margin,
                                                                 const int p,
                                                                 const bool swap,
                                                                 const tensor_view_t<1> O_tv)
{
    // instantiate the kernel
    TripletMarginLossUnreducedForward2d_2<D_TYPE>(ldist, O, margin, p, swap, O_tv);
}

template <typename T>
__device__ void TripletMarginLossForward2d_2(const T* ldist,
                                             T* lsum,
                                             const float margin,
                                             const int p,
                                             const bool swap,
                                             const float divisor,
                                             const size_t size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
        return;

    FLOAT_ACCUM ap = pow(CVT_FLOAT2ACCUM(ldist[0 * size + gid]), 1.0f / p);
    FLOAT_ACCUM an = pow(CVT_FLOAT2ACCUM(ldist[1 * size + gid]), 1.0f / p);
    FLOAT_ACCUM pn = pow(CVT_FLOAT2ACCUM(ldist[2 * size + gid]), 1.0f / p);

    if(swap && pn < an)
    {
        an = pn;
    }

    lsum[gid] = CVT_ACCUM2FLOAT(max(ap - an + margin, 0.0f) / divisor);
}

extern "C" __global__ void TripletMarginLossForward2d_2(const D_TYPE* ldist,
                                                        D_TYPE* lsum,
                                                        const float margin,
                                                        const int p,
                                                        const bool swap,
                                                        const float divisor,
                                                        const size_t size)
{
    // instantiate the kernel
    TripletMarginLossForward2d_2<D_TYPE>(ldist, lsum, margin, p, swap, divisor, size);
}
