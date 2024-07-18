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
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "hip_atomic.hpp"
#include "tensor_utils.hpp"

template <typename T>
__device__ void RepeatForwardImpl(const T* __restrict__ x,
                                  T* __restrict__ y,
                                  uint64_t inout_size,
                                  uint64_t offset,
                                  const uint64_t input_dimensions[5],
                                  const uint64_t output_dimensions[5],
                                  const uint64_t input_strides[5],
                                  const uint64_t output_strides[5])
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= inout_size)
        return;

    // get output index
    uint64_t o[5];
    GET_NCDHW(o, gid, output_dimensions);

    // get input index
    uint64_t n[5] = {0, 0, 0, 0, 0};
    for(uint64_t i = offset; i < 5; i++)
    {
        n[i - offset] = o[i] % input_dimensions[i - offset];
    }

    uint64_t input_index  = GET_STRIDED_INDEX(n, input_strides);
    uint64_t output_index = GET_STRIDED_INDEX(o, output_strides);
    y[output_index]       = x[input_index];
}

template <typename T>
__device__ void RepeatBackwardImpl(const T* __restrict__ dy,
                                   T* __restrict__ dx,
                                   uint64_t inout_size,
                                   uint64_t offset,
                                   const uint64_t output_grad_dimensions[5],
                                   const uint64_t input_grad_dimensions[5],
                                   const uint64_t output_grad_strides[5],
                                   const uint64_t input_grad_strides[5])
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= inout_size)
        return;

    // get output index
    uint64_t o[5];
    GET_NCDHW(o, gid, output_grad_dimensions);

    // get input index
    uint64_t n[5] = {0, 0, 0, 0, 0};
    for(uint64_t i = offset; i < 5; i++)
    {
        n[i - offset] = o[i] % input_grad_dimensions[i - offset];
    }

    uint64_t input_grad_index  = GET_STRIDED_INDEX(n, input_grad_strides);
    uint64_t output_grad_index = GET_STRIDED_INDEX(o, output_grad_strides);
    atomic_add_g(&dx[input_grad_index], dy[output_grad_index]);
}

extern "C" __global__ void RepeatForward(const FLOAT* __restrict__ x,
                                         FLOAT* __restrict__ y,
                                         uint64_t inout_size,
                                         uint64_t offset,
                                         uint64_t input_dim0,
                                         uint64_t input_dim1,
                                         uint64_t input_dim2,
                                         uint64_t input_dim3,
                                         uint64_t input_dim4,
                                         uint64_t output_dim0,
                                         uint64_t output_dim1,
                                         uint64_t output_dim2,
                                         uint64_t output_dim3,
                                         uint64_t output_dim4,
                                         uint64_t input_stride0,
                                         uint64_t input_stride1,
                                         uint64_t input_stride2,
                                         uint64_t input_stride3,
                                         uint64_t input_stride4,
                                         uint64_t output_stride0,
                                         uint64_t output_stride1,
                                         uint64_t output_stride2,
                                         uint64_t output_stride3,
                                         uint64_t output_stride4)
{
    uint64_t input_dimensions[5]  = {input_dim0, input_dim1, input_dim2, input_dim3, input_dim4};
    uint64_t output_dimensions[5] = {
        output_dim0, output_dim1, output_dim2, output_dim3, output_dim4};
    uint64_t input_strides[5] = {
        input_stride0, input_stride1, input_stride2, input_stride3, input_stride4};
    uint64_t output_strides[5] = {
        output_stride0, output_stride1, output_stride2, output_stride3, output_stride4};
    RepeatForwardImpl<FLOAT>(x,
                             y,
                             inout_size,
                             offset,
                             input_dimensions,
                             output_dimensions,
                             input_strides,
                             output_strides);
}

extern "C" __global__ void RepeatBackward(const FLOAT* __restrict__ dy,
                                          FLOAT* __restrict__ dx,
                                          uint64_t inout_size,
                                          uint64_t offset,
                                          uint64_t output_grad_dim0,
                                          uint64_t output_grad_dim1,
                                          uint64_t output_grad_dim2,
                                          uint64_t output_grad_dim3,
                                          uint64_t output_grad_dim4,
                                          uint64_t input_grad_dim0,
                                          uint64_t input_grad_dim1,
                                          uint64_t input_grad_dim2,
                                          uint64_t input_grad_dim3,
                                          uint64_t input_grad_dim4,
                                          uint64_t output_grad_stride0,
                                          uint64_t output_grad_stride1,
                                          uint64_t output_grad_stride2,
                                          uint64_t output_grad_stride3,
                                          uint64_t output_grad_stride4,
                                          uint64_t input_grad_stride0,
                                          uint64_t input_grad_stride1,
                                          uint64_t input_grad_stride2,
                                          uint64_t input_grad_stride3,
                                          uint64_t input_grad_stride4)
{
    uint64_t output_grad_dimensions[5] = {
        output_grad_dim0, output_grad_dim1, output_grad_dim2, output_grad_dim3, output_grad_dim4};
    uint64_t input_grad_dimensions[5] = {
        input_grad_dim0, input_grad_dim1, input_grad_dim2, input_grad_dim3, input_grad_dim4};
    uint64_t output_grad_strides[5] = {output_grad_stride0,
                                       output_grad_stride1,
                                       output_grad_stride2,
                                       output_grad_stride3,
                                       output_grad_stride4};
    uint64_t input_grad_strides[5]  = {input_grad_stride0,
                                      input_grad_stride1,
                                      input_grad_stride2,
                                      input_grad_stride3,
                                      input_grad_stride4};
    RepeatBackwardImpl<FLOAT>(dy,
                              dx,
                              inout_size,
                              offset,
                              output_grad_dimensions,
                              input_grad_dimensions,
                              output_grad_strides,
                              input_grad_strides);
}
