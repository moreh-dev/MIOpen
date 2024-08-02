/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#ifndef GUARD_CPU_LOGSUMEXP_HPP
#define GUARD_CPU_LOGSUMEXP_HPP

#include "tensor_holder.hpp"
#include <vector>

template <class T>
void cpu_logsumexp_forward(tensor<T> input,
                           tensor<T>& output)
{
    auto input_dims    = input.desc.GetLengths();
    auto input_strides = input.desc.GetStrides();
    auto output_dims   = output.desc.GetLengths();
    auto output_strides = output.desc.GetStrides();
    auto K = input_dims.size() / output_dims.size();

    auto output_numel = std::accumulate(output_dims.begin(), output_dims.end(), 1LL, std::multiplies<int64_t>());

    par_ford(output_numel)([&](size_t gid) {
        T vals[K];
        T max = std::numeric_limits<T>::lowest();

        for(int64_t k = 0; k < K; ++k)
        {
            std::vector<int64_t> input_idx(input_dims.size(), 0);
            int64_t tmp_gid = gid * K + k;
            for(int i = input_dims.size() - 1; i >= 0; --i)
            {
                input_idx[i] = tmp_gid % input_dims[i];
                tmp_gid /= input_dims[i];
            }
            T val = input[std::inner_product(input_idx.begin(), input_idx.end(), input_strides.begin(), static_cast<size_t>(0))];
            max = max > val ? max : val;
            vals[k] = val;
        }

        T logsum = static_cast<T>(0,0);
        for(int64_t k = 0; k < K; ++k)
        {
            logsum += std::exp(vals[k] - max);
        }

        int64_t tmp_gid = gid;
        for(int i = output_dims.size() - 1; i >= 0; --i)
        {
            output.data[tmp_gid % output_dims[i]] = max + std::log(logsum);
            tmp_gid /= output_dims[i];
        }
        
        output[std::inner_product(output_idx.begin(), output_idx.end(), output_strides.begin(), static_cast<size_t>(0))] = max + std::log(logsum);
    });
}

template <class T>
void cpu_logsumexp_backward(tensor<T> input,
                            tensor<T>& input_grad,
                            tensor<T> output,
                            tensor<T> output_grad,
                            int32_t* dims,
                            int32_t num_dims)
{
    auto input_dims    = input.desc.GetLengths();
    auto input_strides = input.desc.GetStrides();
    auto output_dims   = output.desc.GetLengths();
    auto output_strides = output.desc.GetStrides();

    auto input_grad_numel = std::accumulate(input_grad.desc.GetLengths().begin(),
                                            input_grad.desc.GetLengths().end(),
                                            1LL,
                                            std::multiplies<int64_t>());

    std::fill(input_grad.data.begin(), input_grad.data.end(), 0);

    par_ford(input_grad_numel)([&](size_t gid){
        std::vector<int64_t> input_idx(input_dims.size(), 0);
        int64_t tmp_gid = gid;

        for(int i = input_dims.size() - 1; i >= 0; --i)
        {
            input_idx[i] = tmp_gid % input_dims[i];
            tmp_gid /= input_dims[i];
        }

        std::vector<int64_t> reduced_idx(input_dims.size(), 0);
        for(int i = 0; i < input_dims.size(); ++i)
        {
            if(std::find(dims, dims + num_dims, i) == dims + num_dims)
            {
                reduced_idx[i] = input_idx[i];
            }
        }

        int64_t input_index = std::inner_product(
            input_idx.begin(), input_idx.end(), input_strides.begin(), static_cast<size_t>(0));
        int64_t output_index = std::inner_product(
            reduced_idx.begin(), reduced_idx.end(), output_strides.begin(), static_cast<size_t>(0));

        T x = input[input_index];
        T y = output[output_index];
        T dy = output_grad[output_index];

        input_grad[input_index] = dy * std::exp(x - y);
    });
}
