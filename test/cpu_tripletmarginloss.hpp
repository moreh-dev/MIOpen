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
#include "tensor_view.hpp"
#include <miopen/tripletmarginloss/utils.hpp>

template <class T>
float dist(const tensor<T> I1,
           const tensor<T> I2,
           const int p,
           const float eps,
           const tensor_view_t<2> I1_tv,
           const tensor_view_t<2> I2_tv,
           const size_t n,
           const size_t C)
{
    // eps defined in
    // https://pytorch.org/docs/stable/generated/torch.nn.PairwiseDistance.html
    float d = 0.0f;
    for(size_t c = 0; c < C; c++)
    {
        d += std::pow(std::fabs(static_cast<float>(I1[I1_tv.stride[0] * n + I1_tv.stride[1] * c]) -
                                static_cast<float>(I2[I2_tv.stride[0] * n + I2_tv.stride[1] * c]) +
                                eps),
                      p);
    }
    d = std::pow(d, (1.0f / p));
    return d;
}

template <class T>
void cpu_tripletmarginloss_unreduced_forward(tensor<T> anchor,
                                             tensor<T> positive,
                                             tensor<T> negative,
                                             tensor<T>& ref_output,
                                             float margin,
                                             int p,
                                             int eps,
                                             bool swap)
{
    auto A_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(anchor.desc);
    auto P_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(positive.desc);
    auto N_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(negative.desc);
    auto O_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<1>(ref_output.desc);

    par_ford(ref_output.desc.GetElementSize())([&](int gid) {
        size_t C = A_tv.size[1];
        size_t b = gid;

        if(b >= A_tv.size[0])
            return;

        float ap = dist(anchor, positive, p, eps, A_tv, P_tv, b, C);
        float an = dist(anchor, negative, p, eps, A_tv, N_tv, b, C);
        float pn = dist(positive, negative, p, eps, P_tv, N_tv, b, C);

        if(swap && pn < an)
        {
            an = pn;
        }

        auto loss = std::max(ap - an + margin, 0.0f);

        ref_output[O_tv.stride[0] * b] = static_cast<T>(loss);
    });
}
