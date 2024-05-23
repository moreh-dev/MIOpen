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
    size_t inputSize       = I.desc.GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < inputSize; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, I_tv);

        TIO i = TV_5D_AT(I, n[0], n[1], n[2], n[3], n[4]);
        TT t  = TV_5D_AT(T, n[0], n[1], n[2], n[3], n[4]);

        if(t == 1)
        {
            dI[idx] = TV_5D_AT(dO, n[0], n[1], n[2], n[3], n[4]);
        }
        else
        {
            if(margin - i > 0)
                dI[idx] = -TV_5D_AT(dO, n[0], n[1], n[2], n[3], n[4]);
            else
                dI[idx] = 0.0f;
        }
    }
}

template <class TIO, class TT>
void cpu_hinge_embedding_loss_forward(tensor<TIO> I,
                                      tensor<TT> T,
                                      tensor<TIO>& workspace,
                                      tensor<TIO>& ref_output,
                                      float margin  = 1,
                                      float divisor = 1)
{
    tensor_view_5d_t I_tv = get_inner_expanded_tv(I.desc);
    tensor_view_5d_t T_tv = get_inner_expanded_tv(T.desc);
    size_t size           = I.desc.GetElementSize();
    size_t n[5];

    // Compute loss in each elem
    for(size_t idx = 0; idx < size; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, I_tv);

        TIO i = TV_5D_AT(I, n[0], n[1], n[2], n[3], n[4]);
        TT t  = TV_5D_AT(T, n[0], n[1], n[2], n[3], n[4]);

        if(t == 1)
            workspace[idx] = i / divisor;
        else
            workspace[idx] = std::max(0.0f, margin - i) / divisor;
    }

    // Reduce loss
    const int local_size = 256;
    int offset_a         = 0;
    int offset_b         = size;
    size_t _size         = size;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            TIO shared[local_size];
            for(int j = 0; j < local_size; ++j)
                shared[j] = i + j < _size ? workspace[offset_a + i + j] : 0.0f;
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                for(int j = 0; j < offset; ++j)
                    shared[j] += shared[j + offset];
            if(_size <= local_size)
                ref_output[0] = shared[0];
            else
                workspace[offset_b + i / local_size] = shared[0];
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);
}

template <class TIO, class TT>
void cpu_hinge_embedding_loss_backward(tensor<TIO> I,
                                       tensor<TT> T,
                                       tensor<TIO> dO,
                                       tensor<TIO>& dI,
                                       float margin  = 1,
                                       float divisor = 1)
{
    tensor_view_5d_t I_tv  = get_inner_expanded_tv(I.desc);
    tensor_view_5d_t T_tv  = get_inner_expanded_tv(T.desc);
    tensor_view_5d_t dO_tv = get_inner_expanded_tv(dO.desc);
    size_t inputSize       = I.desc.GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < inputSize; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, I_tv);

        TIO i = TV_5D_AT(I, n[0], n[1], n[2], n[3], n[4]);
        TT t  = TV_5D_AT(T, n[0], n[1], n[2], n[3], n[4]);
        TIO o = TV_5D_AT(dO, 0, 0, 0, 0, 0);

        if(t == 1)
        {
            dI[idx] = o / divisor;
        }
        else
        {
            if(margin - i > 0)
                dI[idx] = -o / divisor;
            else
                dI[idx] = 0.0f;
        }
    }
}
#endif
