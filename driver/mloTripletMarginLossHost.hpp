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

#include <../test/ford.hpp>

#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tripletmarginloss/utils.hpp>

template <class T>
float dist(const T* I1,
           const T* I2,
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

template <typename Tgpu, typename Tcheck>
int32_t mloTripletMarginLossUnreducedForwardRunHost(const miopenTensorDescriptor_t aDesc,
                                                    const miopenTensorDescriptor_t pDesc,
                                                    const miopenTensorDescriptor_t nDesc,
                                                    const miopenTensorDescriptor_t oDesc,
                                                    const Tgpu* anchor,
                                                    const Tgpu* positive,
                                                    const Tgpu* negative,
                                                    Tcheck* outputhost,
                                                    const float margin,
                                                    const int p,
                                                    const float eps,
                                                    const bool swap)
{
    auto A_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(aDesc));
    auto P_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(pDesc));
    auto N_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(nDesc));
    auto O_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<1>(miopen::deref(oDesc));

    par_ford(miopen::deref(oDesc).GetElementSize())([&](int gid) {
        size_t C = A_tv.size[1];
        size_t b = gid;

        if(b >= A_tv.size[0])
            return;

        float ap = dist<Tgpu>(anchor, positive, p, eps, A_tv, P_tv, b, C);
        float an = dist<Tgpu>(anchor, negative, p, eps, A_tv, N_tv, b, C);
        float pn = dist<Tgpu>(positive, negative, p, eps, P_tv, N_tv, b, C);

        if(swap && pn < an)
        {
            an = pn;
        }

        auto loss = std::max(ap - an + margin, 0.0f);

        outputhost[O_tv.stride[0] * b] = static_cast<Tcheck>(loss);
    });

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloTripletMarginLossReducedForwardRunHost(const miopenTensorDescriptor_t aDesc,
                                                  const miopenTensorDescriptor_t pDesc,
                                                  const miopenTensorDescriptor_t nDesc,
                                                  const Tgpu* anchor,
                                                  const Tgpu* positive,
                                                  const Tgpu* negative,
                                                  Tcheck* outputhost,
                                                  const float margin,
                                                  const int p,
                                                  const float eps,
                                                  const bool swap,
                                                  const float divisor)
{
    auto A_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(aDesc));
    auto P_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(pDesc));
    auto N_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(nDesc));

    std::vector<float> buffer(A_tv.size[0]);

    par_ford(A_tv.size[0])([&](int gid) {
        size_t C = A_tv.size[1];
        size_t b = gid;

        if(b >= A_tv.size[0])
            return;

        float ap = dist<Tgpu>(anchor, positive, p, eps, A_tv, P_tv, b, C);
        float an = dist<Tgpu>(anchor, negative, p, eps, A_tv, N_tv, b, C);
        float pn = dist<Tgpu>(positive, negative, p, eps, P_tv, N_tv, b, C);

        if(swap && pn < an)
        {
            an = pn;
        }

        auto loss = std::max(ap - an + margin, 0.0f);

        buffer[gid] = loss;
    });

    double loss_sum = 0.0;
    for(auto loss : buffer)
        loss_sum += loss;

    outputhost[0] = static_cast<Tcheck>(loss_sum / divisor);

    return miopenStatusSuccess;
}
