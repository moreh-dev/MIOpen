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

#ifndef GUARD_CPU_MASKEDFILL_HPP
#define GUARD_CPU_MASKEDFILL_HPP

#include "tensor_holder.hpp"

template <class T, size_t dim>
void cpu_maskedfill_forward(tensor<T> const& input,
                            tensor<T>& output,
                            tensor<int8_t> const& mask,
                            float const value)
{
    auto const inputtensorview  = get_inner_expanded_tv<dim>(input.desc);
    auto const outputtensorview = get_inner_expanded_tv<dim>(output.desc);
    auto const masktensorview   = get_inner_expanded_tv<dim>(mask.desc);
    par_ford(output.desc.GetElementSize())([&](size_t const gid) {
        output[outputtensorview.get_tensor_view_idx({outputtensorview, gid})] =
            mask[masktensorview.get_tensor_view_idx({masktensorview, gid})]
                ? static_cast<T>(value)
                : input[inputtensorview.get_tensor_view_idx({inputtensorview, gid})];
    });
}

template <class T, size_t dim>
void cpu_maskedfill_backward(tensor<T> const& outputgradient,
                             tensor<T>& inputgradient,
                             tensor<int8_t> const& mask)
{
    auto const outputgradienttensorview = get_inner_expanded_tv<dim>(outputgradient.desc);
    auto const inputgradienttensorview  = get_inner_expanded_tv<dim>(inputgradient.desc);
    auto const masktensorview           = get_inner_expanded_tv<dim>(mask.desc);
    par_ford(inputgradient.desc.GetElementSize())([&](size_t const gid) {
        inputgradient[inputgradienttensorview.get_tensor_view_idx({inputgradienttensorview, gid})] =
            mask[masktensorview.get_tensor_view_idx({masktensorview, gid})]
                ? static_cast<T>(0)
                : outputgradient[outputgradienttensorview.get_tensor_view_idx(
                      {outputgradienttensorview, gid})];
    });
}

#endif
