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
#ifndef GUARD_CPU_HINGE_EMBEDDING_LOSS_HPP
#define GUARD_CPU_HINGE_EMBEDDING_LOSS_HPP

#include "miopen/tensor.hpp"
#include "tensor_holder.hpp"
#include "tensor_view_5d.hpp"
#include <algorithm>
#include <cstddef>

inline tensor_view_5d_t get_inner_expanded_tv(const miopen::TensorDescriptor Desc)
{
    auto dims    = Desc.GetLengths();
    auto strides = Desc.GetStrides();

    tensor_view_5d_t tv_5d;
    for(size_t i = 0; i < strides.size(); ++i)
    {
        tv_5d.stride[i] = strides[i];
        tv_5d.size[i]   = dims[i];
    }
    auto rest = strides.size();
    for(size_t j = rest; j < 5; ++j)
    {
        tv_5d.stride[j] = (rest == 0 ? 1 : strides[rest - 1]);
        tv_5d.size[j]   = 1;
    }
    return tv_5d;
}

template <class TIO, class TT>
void cpu_hinge_embedding_loss_unreduced_forward(tensor<TIO> I,
                                                tensor<TT> T,
                                                tensor<TIO>& ref_output,
                                                float margin = 1)
{
    tensor_view_5d_t I_tv = get_inner_expanded_tv(I.desc);
    tensor_view_5d_t T_tv = get_inner_expanded_tv(T.desc);
    size_t inputSize      = I.desc.GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < inputSize; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, I_tv);

        TIO i = TV_5D_AT(I, n[0], n[1], n[2], n[3], n[4]);
        TT t  = TV_5D_AT(T, n[0], n[1], n[2], n[3], n[4]);

        if(t == 1)
            ref_output[idx] = i;
        else
            ref_output[idx] = std::max(0.0f, margin - i);
    }
}

template <class TIO, class TT>
void cpu_hinge_embedding_loss_unreduced_backward(
    tensor<TIO> I, tensor<TT> T, tensor<TIO> dO, tensor<TIO>& dI, float margin = 1)
{
    tensor_view_5d_t I_tv  = get_inner_expanded_tv(I.desc);
    tensor_view_5d_t T_tv  = get_inner_expanded_tv(T.desc);
    tensor_view_5d_t dO_tv = get_inner_expanded_tv(dO.desc);
    tensor_view_5d_t dI_tv = get_inner_expanded_tv(dI.desc);
    size_t inputSize       = I.desc.GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < inputSize; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, I_tv);

        TIO i = TV_5D_AT(I, n[0], n[1], n[2], n[3], n[4]);
        TT t  = TV_5D_AT(T, n[0], n[1], n[2], n[3], n[4]);

        if(t == 1)
        {
            TV_5D_AT(dI, n[0], n[1], n[2], n[3], n[4]) = TV_5D_AT(dO, n[0], n[1], n[2], n[3], n[4]);
        }
        else
        {
            if(margin - i > 0)
                TV_5D_AT(dI, n[0], n[1], n[2], n[3], n[4]) =
                    -TV_5D_AT(dO, n[0], n[1], n[2], n[3], n[4]);
            else
                TV_5D_AT(dI, n[0], n[1], n[2], n[3], n[4]) = 0.0f;
        }
    }
}
#endif
