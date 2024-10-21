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

#include <miopen/gather/problem_description.hpp>

template <typename Tgpu, typename Tcheck, typename Tindex>
int mloGatherV2BackwardRunHost(miopenTensorDescriptor_t outputGradDesc,
                               const Tgpu* outputGrad,
                               miopenTensorDescriptor_t indicesDesc,
                               const Tindex* indices,
                               miopenTensorDescriptor_t paramGradDesc,
                               Tcheck* paramGrad,
                               uint32_t dim,
                               uint32_t batch_dims)
{
    size_t batch_size = 1;
    size_t outer_size = 1;
    size_t inner_size = 1;

    auto paramGrad_num_dim = miopen::deref(paramGradDesc).GetNumDims();
    auto paramGrad_lens    = miopen::deref(paramGradDesc).GetLengths();
    auto outGrad_numel     = miopen::deref(outputGradDesc).GetElementSize();

    for(uint32_t i = 0; i < batch_dims; i++)
    {
        batch_size *= paramGrad_lens[i];
    }
    for(uint32_t i = batch_dims; i < dim; i++)
    {
        outer_size *= paramGrad_lens[i];
    }
    for(uint32_t i = dim + 1; i < paramGrad_num_dim; i++)
    {
        inner_size *= paramGrad_lens[i];
    }

    auto indices_numel   = miopen::deref(indicesDesc).GetElementSize() / batch_size;
    auto gather_dim_size = paramGrad_lens[dim];

    const bool is_batch_dims_zero = (batch_size == 1);
    const bool is_axis_zero       = (outer_size == 1);

    if(batch_dims > 0)
    {
        auto outGrad_tv = miopen::gather::reshape<4>(
            miopen::deref(outputGradDesc), {batch_size, outer_size, indices_numel, inner_size});

        for(size_t i = 0; i < outGrad_numel; i++)
        {
            size_t batch_i   = 0;
            size_t outer_i   = 0;
            size_t indices_i = 0;
            size_t inner_i   = 0;

            const size_t slices_count = i / inner_size;
            inner_i                   = i - slices_count * inner_size;
            if(is_batch_dims_zero)
            {
                if(is_axis_zero)
                {
                    indices_i = slices_count;
                }
                else
                {
                    outer_i   = slices_count / indices_numel;
                    indices_i = slices_count - outer_i * indices_numel;
                }
            }
            else
            {
                const size_t entries_count = slices_count / indices_numel;
                indices_i                  = slices_count - entries_count * indices_numel;
                if(is_axis_zero)
                {
                    batch_i = entries_count;
                }
                else
                {
                    batch_i = entries_count / outer_size;
                    outer_i = entries_count - batch_i * outer_size;
                }
            }

            size_t gather_i = indices[batch_i * indices_numel + indices_i];
            if(gather_i < gather_dim_size)
            {
                size_t param_i =
                    ((batch_i * outer_size + outer_i) * gather_dim_size + gather_i) * inner_size +
                    inner_i;
                paramGrad[param_i] += getNDVal(outputGrad, outGrad_tv, i);
            }
        }
    }
    else
    {
        auto outputGrad_tv = miopen::gather::reshape<3>(miopen::deref(outputGradDesc),
                                                        {outer_size, indices_numel, inner_size});
        for(size_t i = 0; i < outGrad_numel; i++)
        {
            size_t outer_i   = 0;
            size_t indices_i = 0;
            size_t inner_i   = 0;
            if(is_axis_zero)
            {
                indices_i = i / inner_size;
                inner_i   = i - indices_i * inner_size;
            }
            else
            {
                size_t batch_indices_i = i / inner_size;
                outer_i                = batch_indices_i / indices_numel;
                indices_i              = batch_indices_i - outer_i * indices_numel;
                inner_i                = i - batch_indices_i * inner_size;
            }

            size_t gather_i = indices[indices_i];

            if(gather_i < gather_dim_size)
            {
                size_t param_i = (outer_i * gather_dim_size + gather_i) * inner_size + inner_i;
                paramGrad[param_i] += getNDVal(outputGrad, outputGrad_tv, i);
                // paramGrad[outer_i][gather_i][inner_i] += outputGrad[i];
            }
        }
    }

    return 0;
}
