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
#ifndef GUARD_CPU_BCELOSS_HPP
#define GUARD_CPU_BCELOSS_HPP

#include "tensor_holder.hpp"
#include <miopen/tensor_view_5d.hpp>

template <class T>
void cpu_bcelogitsloss_reduced_forward(tensor<T> input,
                                       tensor<T> target,
                                       tensor<T> weight,
                                       tensor<T> pos_weight,
                                       tensor<T>& ref_output,
                                       tensor<T>& ref_workspace,
                                       float divisor,
                                       bool hasWeight,
                                       bool hasPosWeight)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto input_tv      = get_inner_expanded_tv(input.desc);
    auto target_tv     = get_inner_expanded_tv(target.desc);
    auto weight_tv     = get_inner_expanded_tv(weight.desc);
    auto pos_weight_tv = get_inner_expanded_tv(pos_weight.desc);

    auto size = input.desc.GetElementSize();

    /* Phase 1: Calc loss for each element. */
    par_ford(size)([&](size_t i) {
        size_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, input_tv);

        if(n[0] >= input_tv.size[0])
            return;

        size_t c = i % pos_weight_tv.size[0];

        double w =
            (!hasWeight ? 1.0
                        : static_cast<double>(TV_5D_AT(weight, n[0], n[1], n[2], n[3], n[4])));
        double pw = (!hasPosWeight ? 1.0 : static_cast<double>(TV_1D_AT(pos_weight, c)));

        double x = static_cast<double>(TV_5D_AT(input, n[0], n[1], n[2], n[3], n[4]));
        double y = static_cast<double>(TV_5D_AT(target, n[0], n[1], n[2], n[3], n[4]));

        double max_val;
        max_val = (x < 0) ? -x : 0.0f;

        double loss      = w * (((1.0f - y) * x) +
                           (1 + (pw - 1) * y) * (log(exp(-max_val) + exp(-x - max_val)) + max_val));
        ref_workspace[i] = static_cast<T>(loss / divisor);
    });

    /* Phase 2: Reduce */
    const int local_size = 256;
    int offset_a         = 0;
    int offset_b         = size;
    size_t _size         = size;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            double shared[local_size];
            for(int j = 0; j < local_size; ++j)
                shared[j] =
                    static_cast<double>(i + j < _size ? ref_workspace[offset_a + i + j] : 0.0f);
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                for(int j = 0; j < offset; ++j)
                    shared[j] += shared[j + offset];
            if(_size <= local_size)
                ref_output[0] = static_cast<T>(shared[0]);
            else
                ref_workspace[offset_b + i / local_size] = static_cast<T>(shared[0]);
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);
}

inline double sigmoid(double x) { return 1.0f / (1.0f + exp(-x)); }

template <class T>
void cpu_bcelogitsloss_reduced_backward(tensor<T> input,
                                        tensor<T> target,
                                        tensor<T> weight,
                                        tensor<T> pos_weight,
                                        tensor<T> dO,
                                        tensor<T>& ref_dI,
                                        tensor<T>& ref_dT,
                                        float divisor,
                                        bool hasWeight,
                                        bool hasPosWeight)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto input_tv      = get_inner_expanded_tv(input.desc);
    auto target_tv     = get_inner_expanded_tv(target.desc);
    auto weight_tv     = get_inner_expanded_tv(weight.desc);
    auto pos_weight_tv = get_inner_expanded_tv(pos_weight.desc);
    auto ref_dI_tv     = get_inner_expanded_tv(ref_dI.desc);
    auto ref_dT_tv     = get_inner_expanded_tv(ref_dT.desc);

    auto size = input.desc.GetElementSize();

    par_ford(size)([&](size_t i) {
        size_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, input_tv);

        if(n[0] >= input_tv.size[0])
            return;

        size_t c = i % pos_weight_tv.size[0];

        double w =
            (!hasWeight ? 1.0
                        : static_cast<double>(TV_5D_AT(weight, n[0], n[1], n[2], n[3], n[4])));
        double pw = (!hasPosWeight ? 1.0 : static_cast<double>(TV_1D_AT(pos_weight, c)));

        double x = static_cast<double>(TV_5D_AT(input, n[0], n[1], n[2], n[3], n[4]));
        double y = static_cast<double>(TV_5D_AT(target, n[0], n[1], n[2], n[3], n[4]));

        {
            size_t dIidx  = TV5D_IDX(ref_dI_tv, n[0], n[1], n[2], n[3], n[4]);
            double result = -w * (pw * y * (1.0f - sigmoid(x)) + (y - 1.0f) * sigmoid(x));
            result *= static_cast<double>(dO[0]) / divisor;
            ref_dI[dIidx] = static_cast<T>(result);
        }

        {
            size_t dTidx  = TV5D_IDX(ref_dT_tv, n[0], n[1], n[2], n[3], n[4]);
            double result = w * (log(1.0f - sigmoid(x)) - pw * log(sigmoid(x)));
            result *= static_cast<double>(dO[0]) / divisor;
            ref_dT[dTidx] = static_cast<T>(result);
        }
    });
}

#endif // GUARD_CPU_BCELOSS_HPP
