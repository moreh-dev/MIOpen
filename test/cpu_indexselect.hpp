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
#ifndef GUARD_CPU_INDEXSELECT_HPP
#define GUARD_CPU_INDEXSELECT_HPP

#include "tensor_holder.hpp"

template <class T>
void cpu_indexselect_forward(
    tensor<T> input, tensor<int> indices, tensor<T> output, int dim, tensor<T>& outputhost)
{
    auto input_dims  = input.desc.GetLengths();
    auto output_dims = output.desc.GetLengths();

    auto input_strides  = input.desc.GetStrides();
    auto output_strides = output.desc.GetStrides();

    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1LL, std::multiplies<int64_t>());

    size_t n[4], n012, n01;

    for(size_t i = 0; i < output_numel; i++)
    {
        n[3] = i % output_dims[3];
        n012 = i / output_dims[3];
        n[2] = n012 % output_dims[2];
        n01  = n012 / output_dims[2];
        n[1] = n01 % output_dims[1];
        n[0] = n01 / output_dims[1];

        size_t output_idx = n[0] * output_strides[0] + n[1] * output_strides[1] +
                            n[2] * output_strides[2] + n[3] * output_strides[3];

        n[dim] = indices[n[dim]];

        size_t input_idx = n[0] * input_strides[0] + n[1] * input_strides[1] +
                           n[2] * input_strides[2] + n[3] * input_strides[3];

        outputhost[output_idx] = input[input_idx];
    }
}
#endif