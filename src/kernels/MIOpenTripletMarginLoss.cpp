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

template <typename TI, typename TO>
inline __device__ void dist(const TI* I1,
                            const TI* I2,
                            TO* Dis,
                            const int p,
                            const float eps,
                            const tensor_view_t<2> I1_tv,
                            const tensor_view_t<2> I2_tv,
                            const int b,
                            const int c)
{
    int gid        = blockIdx.x * blockDim.x + threadIdx.x;
    FLOAT_ACCUM i1 = CVT_FLOAT2ACCUM(I1[I1_tv.stride[0] * b + I1_tv.stride[1] * c]);
    FLOAT_ACCUM i2 = CVT_FLOAT2ACCUM(I2[I2_tv.stride[0] * b + I2_tv.stride[1] * c]);
    Dis[gid]       = CVT_ACCUM2FLOAT(pow(fabs(i1 - i2) + eps, p));
}

template <typename TI, typename TO>
__device__ void TripletMarginLossDist2d(const TI* A,
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
    int b = gid / A_tv.size[1];

    if(b >= A_tv.size[0])
        return;

    int size = A_tv.size[0] * A_tv.size[1];
    dist<TI, TO>(A, P, ldist + 0 * size, p, eps, A_tv, P_tv, b, c);
    dist<TI, TO>(A, N, ldist + 1 * size, p, eps, A_tv, N_tv, b, c);
    dist<TI, TO>(P, N, ldist + 2 * size, p, eps, P_tv, N_tv, b, c);
}

extern "C" __global__ void TripletMarginLossDist2d(const INPUT_TYPE* A,
                                                   const INPUT_TYPE* P,
                                                   const INPUT_TYPE* N,
                                                   D_TYPE* ldist,
                                                   const int p,
                                                   const float eps,
                                                   const tensor_view_t<2> A_tv,
                                                   const tensor_view_t<2> P_tv,
                                                   const tensor_view_t<2> N_tv)
{
    // instantiate the kernel
    TripletMarginLossDist2d<INPUT_TYPE, D_TYPE>(A, P, N, ldist, p, eps, A_tv, P_tv, N_tv);
}

template <typename T>
__device__ void TripletMarginLossDistSumPow2d(const T* ldist_a,
                                              T* ldist_b,
                                              const size_t size,
                                              const size_t reduce_size,
                                              const int p,
                                              const float eps)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= size)
        return;
    FLOAT_ACCUM dist = 0.0;
    for(size_t i = 0; i < reduce_size; ++i)
        dist += CVT_FLOAT2ACCUM(ldist_a[gid * reduce_size + i]);
    ldist_b[gid] = CVT_ACCUM2FLOAT(pow(dist + eps, 1.0f / p));
}

extern "C" __global__ void TripletMarginLossDistSumPow2d(const D_TYPE* ldist_a,
                                                         D_TYPE* ldist_b,
                                                         const size_t size,
                                                         const size_t reduce_size,
                                                         const int p,
                                                         const float eps)
{
    // instantiate the kernel
    TripletMarginLossDistSumPow2d<D_TYPE>(ldist_a, ldist_b, size, reduce_size, p, eps);
}

template <typename TI, typename TO>
__device__ void TripletMarginLossUnreducedForward2d(const TI* ldist,
                                                    TO* O,
                                                    const float margin,
                                                    const float eps,
                                                    const bool swap,
                                                    const tensor_view_t<1> O_tv)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= O_tv.size[0])
        return;

    int b          = gid;
    FLOAT_ACCUM ap = CVT_FLOAT2ACCUM(ldist[0 * O_tv.size[0] + b]) + eps;
    FLOAT_ACCUM an = CVT_FLOAT2ACCUM(ldist[1 * O_tv.size[0] + b]) + eps;
    FLOAT_ACCUM pn = CVT_FLOAT2ACCUM(ldist[2 * O_tv.size[0] + b]) + eps;

    if(swap && pn < an)
    {
        an = pn;
    }

    O[O_tv.stride[0] * b] = CVT_ACCUM2FLOAT(max(ap - an + margin, 0.0f));
}

extern "C" __global__ void TripletMarginLossUnreducedForward2d(const D_TYPE* ldist,
                                                               OUTPUT_TYPE* O,
                                                               const float margin,
                                                               const float eps,
                                                               const bool swap,
                                                               const tensor_view_t<1> O_tv)
{
    // instantiate the kernel
    TripletMarginLossUnreducedForward2d<D_TYPE, OUTPUT_TYPE>(ldist, O, margin, eps, swap, O_tv);
}

template <typename TI, typename TO>
__device__ void TripletMarginLossForward2d(const TI* ldist,
                                           TO* lsum,
                                           const float margin,
                                           const float eps,
                                           const bool swap,
                                           const float divisor,
                                           const size_t size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
        return;

    int b          = gid;
    FLOAT_ACCUM ap = CVT_FLOAT2ACCUM(ldist[0 * size + b]) + eps;
    FLOAT_ACCUM an = CVT_FLOAT2ACCUM(ldist[1 * size + b]) + eps;
    FLOAT_ACCUM pn = CVT_FLOAT2ACCUM(ldist[2 * size + b]) + eps;

    if(swap && pn < an)
    {
        an = pn;
    }

    lsum[b] = CVT_ACCUM2FLOAT(max(ap - an + margin, 0.0f) / divisor);
}

extern "C" __global__ void TripletMarginLossForward2d(const D_TYPE* ldist,
                                                      OUTPUT_TYPE* lsum,
                                                      const float margin,
                                                      const float eps,
                                                      const bool swap,
                                                      const float divisor,
                                                      const size_t size)
{
    // instantiate the kernel
    TripletMarginLossForward2d<D_TYPE, OUTPUT_TYPE>(ldist, lsum, margin, eps, swap, divisor, size);
}

template <typename TI, typename TO, typename T>
__device__ void TripletMarginLossUnreducedBackward2d(const T* ldist,
                                                     const TI* A,
                                                     const TI* P,
                                                     const TI* N,
                                                     const TO* dO,
                                                     TI* dA,
                                                     TI* dP,
                                                     TI* dN,
                                                     const float margin,
                                                     const int p,
                                                     const float eps,
                                                     const bool swap,
                                                     const tensor_view_t<2> A_tv,
                                                     const tensor_view_t<2> P_tv,
                                                     const tensor_view_t<2> N_tv,
                                                     const tensor_view_t<1> dO_tv,
                                                     const tensor_view_t<2> dA_tv,
                                                     const tensor_view_t<2> dP_tv,
                                                     const tensor_view_t<2> dN_tv)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int c = gid % dA_tv.size[1];
    int b = gid / dA_tv.size[1];

    if(b >= dA_tv.size[0])
        return;

    FLOAT_ACCUM ap = CVT_FLOAT2ACCUM(ldist[0 * dA_tv.size[0] + b]);
    FLOAT_ACCUM an = CVT_FLOAT2ACCUM(ldist[1 * dA_tv.size[0] + b]);
    FLOAT_ACCUM pn = CVT_FLOAT2ACCUM(ldist[2 * dA_tv.size[0] + b]);

    bool swapped = true;
    if(swap && pn < an)
    {
        an      = pn;
        swapped = true;
    }

    FLOAT_ACCUM grad_output = CVT_FLOAT2ACCUM(dO[dO_tv.stride[0] * b]);

    if(ap - an + margin > 0)
    {
        FLOAT_ACCUM a   = CVT_FLOAT2ACCUM(A[A_tv.stride[0] * b + A_tv.stride[1] * c]);
        FLOAT_ACCUM pos = CVT_FLOAT2ACCUM(P[P_tv.stride[0] * b + P_tv.stride[1] * c]);
        FLOAT_ACCUM neg = CVT_FLOAT2ACCUM(N[N_tv.stride[0] * b + N_tv.stride[1] * c]);
        FLOAT_ACCUM l, grad;
        if(dA)
        {
            grad = CVT_FP32_2ACCUM(0.0f);
            l    = pow(fabs(a - pos) + eps, (p - 1)) * pow(ap, (1 - p));
            if(a < pos)
                l = -l;
            grad += l * grad_output;
            if(!swapped)
            {
                l = -pow(fabs(a - neg) + eps, (p - 1)) * pow(an, (1 - p));
                if(a < neg)
                    l = -l;
                grad += l * grad_output;
            }
            dA[dA_tv.stride[0] * b + dA_tv.stride[1] * c] = CVT_ACCUM2FLOAT(grad);
        }
        if(dP)
        {
            grad = CVT_FP32_2ACCUM(0.0f);
            l    = -pow(fabs(a - pos) + eps, (p - 1)) * pow(ap, (1 - p));
            if(a < pos)
                l = -l;
            grad += l * grad_output;
            if(swapped)
            {
                l = -pow(fabs(pos - neg) + eps, (p - 1)) * pow(pn, (1 - p));
                if(pos < neg)
                    l = -l;
                grad += l * grad_output;
            }
            dP[dP_tv.stride[0] * b + dP_tv.stride[1] * c] = CVT_ACCUM2FLOAT(grad);
        }
        if(dN)
        {
            if(swapped)
            {
                l = pow(fabs(pos - neg) + eps, (p - 1)) * pow(pn, (1 - p));
                if(pos < neg)
                    l = -l;
            }
            else
            {
                l = pow(fabs(a - neg) + eps, (p - 1)) * pow(an, (1 - p));
                if(a < neg)
                    l = -l;
            }
            dN[dN_tv.stride[0] * b + dN_tv.stride[1] * c] = CVT_ACCUM2FLOAT(l * grad_output);
        }
    }
    else
    {
        if(dA)
            dA[dA_tv.stride[0] * b + dA_tv.stride[1] * c] = CVT_FP32_2FLOAT(0.0f);
        if(dP)
            dP[dP_tv.stride[0] * b + dP_tv.stride[1] * c] = CVT_FP32_2FLOAT(0.0f);
        if(dN)
            dN[dN_tv.stride[0] * b + dN_tv.stride[1] * c] = CVT_FP32_2FLOAT(0.0f);
    }
}

extern "C" __global__ void TripletMarginLossUnreducedBackward2d(const D_TYPE* ldist,
                                                                const INPUT_TYPE* A,
                                                                const INPUT_TYPE* P,
                                                                const INPUT_TYPE* N,
                                                                const OUTPUT_TYPE* dO,
                                                                INPUT_TYPE* dA,
                                                                INPUT_TYPE* dP,
                                                                INPUT_TYPE* dN,
                                                                const float margin,
                                                                const int p,
                                                                const float eps,
                                                                const bool swap,
                                                                const tensor_view_t<2> A_tv,
                                                                const tensor_view_t<2> P_tv,
                                                                const tensor_view_t<2> N_tv,
                                                                const tensor_view_t<1> dO_tv,
                                                                const tensor_view_t<2> dA_tv,
                                                                const tensor_view_t<2> dP_tv,
                                                                const tensor_view_t<2> dN_tv)
{
    // instantiate the kernel
    TripletMarginLossUnreducedBackward2d<INPUT_TYPE, OUTPUT_TYPE, D_TYPE>(ldist,
                                                                          A,
                                                                          P,
                                                                          N,
                                                                          dO,
                                                                          dA,
                                                                          dP,
                                                                          dN,
                                                                          margin,
                                                                          p,
                                                                          eps,
                                                                          swap,
                                                                          A_tv,
                                                                          P_tv,
                                                                          N_tv,
                                                                          dO_tv,
                                                                          dA_tv,
                                                                          dP_tv,
                                                                          dN_tv);
}

template <typename TI, typename TO, typename T>
__device__ void TripletMarginLossBackward2d(const T* ldist,
                                            const TI* A,
                                            const TI* P,
                                            const TI* N,
                                            const TO* dO,
                                            TI* dA,
                                            TI* dP,
                                            TI* dN,
                                            const float margin,
                                            const int p,
                                            const float eps,
                                            const bool swap,
                                            const float divisor,
                                            const tensor_view_t<2> A_tv,
                                            const tensor_view_t<2> P_tv,
                                            const tensor_view_t<2> N_tv,
                                            const tensor_view_t<2> dA_tv,
                                            const tensor_view_t<2> dP_tv,
                                            const tensor_view_t<2> dN_tv)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int c = gid % dA_tv.size[1];
    int b = gid / dA_tv.size[1];

    if(b >= dA_tv.size[0])
        return;

    FLOAT_ACCUM ap = CVT_FLOAT2ACCUM(ldist[0 * dA_tv.size[0] + b]);
    FLOAT_ACCUM an = CVT_FLOAT2ACCUM(ldist[1 * dA_tv.size[0] + b]);
    FLOAT_ACCUM pn = CVT_FLOAT2ACCUM(ldist[2 * dA_tv.size[0] + b]);

    bool swapped = true;
    if(swap && pn < an)
    {
        an      = pn;
        swapped = true;
    }

    FLOAT_ACCUM grad_output = CVT_FLOAT2ACCUM(dO[0]);

    if(ap - an + margin > 0)
    {
        FLOAT_ACCUM a   = CVT_FLOAT2ACCUM(A[A_tv.stride[0] * b + A_tv.stride[1] * c]);
        FLOAT_ACCUM pos = CVT_FLOAT2ACCUM(P[P_tv.stride[0] * b + P_tv.stride[1] * c]);
        FLOAT_ACCUM neg = CVT_FLOAT2ACCUM(N[N_tv.stride[0] * b + N_tv.stride[1] * c]);
        FLOAT_ACCUM l, grad;
        if(dA)
        {
            grad = CVT_FP32_2ACCUM(0);
            l    = pow(fabs(a - pos) + eps, (p - 1)) * pow(ap, (1 - p));
            if(a < pos)
                l = -l;
            grad += l * grad_output;
            if(!swapped)
            {
                l = -pow(fabs(a - neg) + eps, (p - 1)) * pow(an, (1 - p));
                if(a < neg)
                    l = -l;
                grad += l * grad_output;
            }
            dA[dA_tv.stride[0] * b + dA_tv.stride[1] * c] = CVT_ACCUM2FLOAT(grad / divisor);
        }
        if(dP)
        {
            grad = CVT_FP32_2ACCUM(0);
            l    = -pow(fabs(a - pos) + eps, (p - 1)) * pow(ap, (1 - p));
            if(a < pos)
                l = -l;
            grad += l * grad_output;
            if(swapped)
            {
                l = -pow(fabs(pos - neg) + eps, (p - 1)) * pow(pn, (1 - p));
                if(pos < neg)
                    l = -l;
                grad += l * grad_output;
            }
            dP[dP_tv.stride[0] * b + dP_tv.stride[1] * c] = CVT_ACCUM2FLOAT(grad / divisor);
        }
        if(dN)
        {
            if(swapped)
            {
                l = pow(fabs(pos - neg) + eps, (p - 1)) * pow(pn, (1 - p));
                if(pos < neg)
                    l = -l;
            }
            else
            {
                l = pow(fabs(a - neg) + eps, (p - 1)) * pow(an, (1 - p));
                if(a < neg)
                    l = -l;
            }
            dN[dN_tv.stride[0] * b + dN_tv.stride[1] * c] =
                CVT_ACCUM2FLOAT(l * grad_output / divisor);
        }
    }
    else
    {
        if(dA)
            dA[dA_tv.stride[0] * b + dA_tv.stride[1] * c] = CVT_FP32_2FLOAT(0.0f);
        if(dP)
            dP[dP_tv.stride[0] * b + dP_tv.stride[1] * c] = CVT_FP32_2FLOAT(0.0f);
        if(dN)
            dN[dN_tv.stride[0] * b + dN_tv.stride[1] * c] = CVT_FP32_2FLOAT(0.0f);
    }
}

extern "C" __global__ void TripletMarginLossBackward2d(const D_TYPE* ldist,
                                                       const INPUT_TYPE* A,
                                                       const INPUT_TYPE* P,
                                                       const INPUT_TYPE* N,
                                                       const OUTPUT_TYPE* dO,
                                                       INPUT_TYPE* dA,
                                                       INPUT_TYPE* dP,
                                                       INPUT_TYPE* dN,
                                                       const float margin,
                                                       const int p,
                                                       const float eps,
                                                       const bool swap,
                                                       const float divisor,
                                                       const tensor_view_t<2> A_tv,
                                                       const tensor_view_t<2> P_tv,
                                                       const tensor_view_t<2> N_tv,
                                                       const tensor_view_t<2> dA_tv,
                                                       const tensor_view_t<2> dP_tv,
                                                       const tensor_view_t<2> dN_tv)
{
    // instantiate the kernel
    TripletMarginLossBackward2d<INPUT_TYPE, OUTPUT_TYPE, D_TYPE>(ldist,
                                                                 A,
                                                                 P,
                                                                 N,
                                                                 dO,
                                                                 dA,
                                                                 dP,
                                                                 dN,
                                                                 margin,
                                                                 p,
                                                                 eps,
                                                                 swap,
                                                                 divisor,
                                                                 A_tv,
                                                                 P_tv,
                                                                 N_tv,
                                                                 dA_tv,
                                                                 dP_tv,
                                                                 dN_tv);
}
