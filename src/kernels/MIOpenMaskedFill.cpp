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
#include <hip/hipf16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"

template <typename T>
__device__ void
MaskedFillForwardContiguousImpl(T const* const __restrict__ input,
                                T* const __restrict__ output,
                                __hip_internal::int8_t const* const __restrict__ mask,
                                T const value,
                                unsigned long const numel)
{
    uint64_t const gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= numel)
        return;
    output[gid] = mask[gid] ? value : input[gid];
}
extern "C" __global__ void
MaskedFillForwardContiguous(FLOAT const* const __restrict__ input,
                            FLOAT* const __restrict__ output,
                            __hip_internal::int8_t const* const __restrict__ mask,
                            float const value,
                            unsigned long const numel)
{
    MaskedFillForwardContiguousImpl<FLOAT>(input,
                                           output,
                                           mask,
#if MIOPEN_USE_BFP16
                                           float_to_bfloat16(value),
#else
                                           value,
#endif
                                           numel);
}

template <typename T>
__device__ void
MaskedFillBackwardContiguousImpl(T const* const __restrict__ output_gradient,
                                 T* const __restrict__ input_gradient,
                                 __hip_internal::int8_t const* const __restrict__ mask,
                                 unsigned long const numel)
{
    const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= numel)
        return;
    input_gradient[gid] = mask[gid] ? static_cast<T>(0) : output_gradient[gid];
}
extern "C" __global__ void
MaskedFillBackwardContiguous(FLOAT const* const __restrict__ output_gradient,
                             FLOAT* const __restrict__ input_gradient,
                             __hip_internal::int8_t const* const __restrict__ mask,
                             float const value,
                             unsigned long const numel)
{
    MaskedFillBackwardContiguousImpl<FLOAT>(output_gradient, input_gradient, mask, numel);
}

template <typename T>
__device__ void MaskedFillForwardImpl(T const* const __restrict__ input,
                                      tensor_view_t<5> const input_tensor_view,
                                      T* const __restrict__ output,
                                      tensor_view_t<5> const output_tensor_view,
                                      __hip_internal::int8_t const* const __restrict__ mask,
                                      tensor_view_t<5> const mask_tensor_view,
                                      T const value,
                                      unsigned long const numel)
{
    uint64_t const gid = blockIdx.x * blockDim.x + threadIdx.x;
    tensor_layout_t output_tensor_layout{output_tensor_view, gid};
    if(output_tensor_layout.layout[0] >= output_tensor_view.size[0])
        return;
    tensor_layout_t input_tensor_layout{input_tensor_view, gid};
    tensor_layout_t mask_tensor_layout{mask_tensor_view, gid};
    output[output_tensor_view.get_tensor_view_idx(output_tensor_layout)] =
        mask[mask_tensor_view.get_tensor_view_idx(mask_tensor_layout)]
            ? value
            : input[input_tensor_view.get_tensor_view_idx(input_tensor_layout)];
}
extern "C" __global__ void MaskedFillForward(FLOAT const* const __restrict__ input,
                                             tensor_view_t<5> const input_tensor_view,
                                             FLOAT* const __restrict__ output,
                                             tensor_view_t<5> const output_tensor_view,
                                             __hip_internal::int8_t const* const __restrict__ mask,
                                             tensor_view_t<5> const mask_tensor_view,
                                             float const value,
                                             unsigned long const numel)
{
    MaskedFillForwardImpl<FLOAT>(input,
                                 input_tensor_view,
                                 output,
                                 output_tensor_view,
                                 mask,
                                 mask_tensor_view,
#if MIOPEN_USE_BFP16
                                 float_to_bfloat16(value),
#else
                                 value,
#endif
                                 numel);
}

template <typename T>
__device__ void MaskedFillBackwardImpl(T const* const __restrict__ output_gradient,
                                       tensor_view_t<5> const output_gradient_tensor_view,
                                       T* const __restrict__ input_gradient,
                                       tensor_view_t<5> const input_gradient_tensor_view,
                                       __hip_internal::int8_t const* const __restrict__ mask,
                                       tensor_view_t<5> const mask_tensor_view,
                                       unsigned long const numel)
{
    const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    tensor_layout_t input_gradient_tensor_layout{input_gradient_tensor_view, gid};
    if(input_gradient_tensor_layout.layout[0] >= input_gradient_tensor_view.size[0])
        return;
    tensor_layout_t output_gradient_tensor_layout{output_gradient_tensor_view, gid};
    tensor_layout_t mask_tensor_layout{mask_tensor_view, gid};
    input_gradient[input_gradient_tensor_view.get_tensor_view_idx(input_gradient_tensor_layout)] =
        mask[mask_tensor_view.get_tensor_view_idx(mask_tensor_layout)]
            ? static_cast<T>(0)
            : output_gradient[output_gradient_tensor_view.get_tensor_view_idx(
                  output_gradient_tensor_layout)];
}
extern "C" __global__ void MaskedFillBackward(FLOAT const* const __restrict__ output_gradient,
                                              tensor_view_t<5> const output_gradient_tensor_view,
                                              FLOAT* const __restrict__ input_gradient,
                                              tensor_view_t<5> const input_gradient_tensor_view,
                                              __hip_internal::int8_t const* const __restrict__ mask,
                                              tensor_view_t<5> const mask_tensor_view,
                                              float const value,
                                              unsigned long const numel)
{
    MaskedFillBackwardImpl<FLOAT>(output_gradient,
                                  output_gradient_tensor_view,
                                  input_gradient,
                                  input_gradient_tensor_view,
                                  mask,
                                  mask_tensor_view,
                                  numel);
}
