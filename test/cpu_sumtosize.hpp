/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <miopen/sumtosize.hpp>
#include <miopen/tensor_view_utils.hpp>

template <class T>
void cpu_sumtosize_forward(const tensor<T> input, tensor<T>& output)
{
    std::vector<int> reduce_dims =
        miopen::sumtosize::GetReduceDim(input.desc.GetLengths(), output.desc.GetLengths());
    size_t reduce_size = 1;
    for(auto reduce_dim : reduce_dims)
        reduce_size *= input.desc.GetLengths()[reduce_dim];
    auto input_tv  = miopen::get_inner_expanded_tv<5>(input.desc);
    auto output_tv = miopen::get_inner_expanded_tv<5>(output.desc);

    par_ford(output.desc.GetElementSize())([&](size_t gid) {
        tensor_layout_t<5> idx_output(output_tv, gid);
        double res = 0;
        for(size_t i = 0; i < reduce_size; i++)
        {
            // construct idx_input
            tensor_layout_t<5> idx_input;
            size_t reduce_idx = i;

            int cur_reduce = 0;

            for(int dim = input.desc.GetNumDims() - 1; dim >= 0; dim--)
            {
                if(cur_reduce < reduce_dims.size() && dim == reduce_dims[cur_reduce])
                {
                    // This is reduce dim
                    cur_reduce++;
                    idx_input.layout[dim] = reduce_idx % input_tv.size[dim];
                    reduce_idx /= input_tv.size[dim];
                }
                else
                {
                    // This is NOT reduce dim
                    idx_input.layout[dim] = idx_output.layout[dim];
                }
            }

            res += static_cast<double>(input[input_tv.get_tensor_view_idx(idx_input)]);
        }
        output[gid] = static_cast<T>(res);
    });
}
