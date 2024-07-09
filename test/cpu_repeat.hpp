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
#ifndef GUARD_CPU_REPEAT_HPP
#define GUARD_CPU_REPEAT_HPP

#include "tensor_holder.hpp"
#include <atomic>
#include <vector>

template <class T>
void cpu_repeat_forward(tensor<T> input,
                        tensor<T>& ref_output,
                        int32_t* sizes,
                        int32_t num_sizes)
{
    auto input_dims  = input.desc.GetLengths();
    auto output_dims = ref_output.desc.GetLengths();

    int32_t offset = num_sizes - static_cast<int32_t>(input_dims.size());
    if(offset < 0)
    {
        throw std::runtime_error("Number of dimensions of sizes cannot be smaller than number of dimensions of input tensor.");
    }

    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1LL, std::multiplies<int64_t>());

    par_ford(output_numel)([&](size_t gid) {
        std::vector<int64_t> output_idx(output_dims.size(), 0);
        int64_t tmp_gid = gid;

        for(int i = output_dims.size() - 1; i >= 0; --i)
        {
            output_idx[i] = tmp_gid % output_dims[i];
            tmp_gid /= output_dims[i];
        }

        std::vector<int64_t> input_idx(input_dims.size(), 0);
        for(int i = offset; i < output_idx.size(); ++i)
        {
            input_idx[i - offset] = output_idx[i] % input_dims[i - offset];
        }

        int64_t input_flat_idx = 0;
        int64_t stride = 1;
        for(int i = input_dims.size() - 1; i >= 0; --i)
        {
            input_flat_idx += input_idx[i] * stride;
            stride *= input_dims[i];
        }

        ref_output[gid] = input[input_flat_idx];
    });
}

template <class T>
void cpu_repeat_backward(tensor<T> doutput,
                         tensor<T>& dinput,
                         int32_t* sizes,
                         int32_t num_sizes)
{
    auto doutput_dims = doutput.desc.GetLengths();
    auto dinput_dims  = dinput.desc.GetLengths();

    int32_t offset = num_sizes - static_cast<int32_t>(dinput_dims.size());
    if(offset < 0)
    {
        throw std::runtime_error("Number of dimensions of sizes cannot be smaller than number of dimensions of input tensor.");
    }

    auto doutput_numel =
        std::accumulate(doutput_dims.begin(), doutput_dims.end(), 1LL, std::multiplies<int64_t>());

    std::fill(dinput.data.begin(), dinput.data.end(), 0);

    for(size_t gid = 0; gid < doutput_numel; ++gid)
    {
        std::vector<int64_t> doutput_idx(doutput_dims.size(), 0);
        int64_t tmp_gid = gid;

        for(int i = doutput_dims.size() - 1; i >= 0; --i)
        {
            doutput_idx[i] = tmp_gid % doutput_dims[i];
            tmp_gid /= doutput_dims[i];
        }

        std::vector<int64_t> dinput_idx(dinput_dims.size(), 0);
        for(int i = offset; i < doutput_idx.size(); ++i)
        {
            dinput_idx[i - offset] = doutput_idx[i] % dinput_dims[i - offset];
        }

        int64_t dinput_flat_idx = 0;
        int64_t stride = 1;
        for(int i = dinput_dims.size() - 1; i >= 0; --i)
        {
            dinput_flat_idx += dinput_idx[i] * stride;
            stride *= dinput_dims[i];
        }

        dinput[dinput_flat_idx] += doutput[gid];
    }
}


#endif
