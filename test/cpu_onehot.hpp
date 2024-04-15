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
#ifndef GUARD_CPU_ONEHOT_HPP
#define GUARD_CPU_ONEHOT_HPP

#include "tensor_holder.hpp"

template <class T>
void cpu_onehot(tensor<T> input, tensor<T>& ref_output, long input_size, long num_classes = -1)
{
    if(num_classes == -1)
    {
        for(int i = 0; i < input_size; ++i)
        {
            num_classes = std::max(num_classes, static_cast<long>(input[i]) + 1);
        }
    }
    auto input_dims  = input.desc.GetLengths();
    auto output_dims = ref_output.desc.GetLengths();

    for(int i = 0; i < input_size; ++i)
    {
        ref_output[i * num_classes + static_cast<int>(input[i])] = 1;
    }
}
#endif
