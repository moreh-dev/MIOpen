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

#pragma once

#include "tensor_holder.hpp"
#include <miopen/rrelu/utils.hpp>

template <class T>
void cpu_rrelu_forward(const tensor<T> input, const tensor<float> noise, tensor<T>& ref_output)
{
    auto input_tv  = miopen::solver::rrelu::get_inner_expanded_tv<5>(input.desc);
    auto output_tv = miopen::solver::rrelu::get_inner_expanded_tv<5>(ref_output.desc);

    int size = input.desc.GetElementSize();
    par_ford(size)([&](int i) {
        auto layout = tensor_layout_t<5>(input_tv, i);
        auto Iidx   = input_tv.get_tensor_view_idx(layout);
        auto Oidx   = output_tv.get_tensor_view_idx(layout);

        ref_output[Oidx] = static_cast<T>(static_cast<float>(input[Iidx]) * noise[i]);
    });
}
