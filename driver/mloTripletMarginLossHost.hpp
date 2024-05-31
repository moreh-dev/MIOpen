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
    float d = 0.0f;
    for(size_t c = 0; c < C; c++)
    {
        d += std::pow(std::fabs(static_cast<float>(I1[I1_tv.stride[0] * n + I1_tv.stride[1] * c]) -
                                static_cast<float>(I2[I2_tv.stride[0] * n + I2_tv.stride[1] * c])) +
                          eps,
                      p);
    }
    d = std::pow(d + eps, (1.0f / p));
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

        outputhost[O_tv.stride[0] * b] = static_cast<Tcheck>(loss);
    });

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloTripletMarginLossForwardRunHost(const miopenTensorDescriptor_t aDesc,
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

template <typename Tgpu, typename Tcheck>
int32_t mloTripletMarginLossUnreducedBackwardRunHost(const miopenTensorDescriptor_t aDesc,
                                                     const miopenTensorDescriptor_t pDesc,
                                                     const miopenTensorDescriptor_t nDesc,
                                                     const miopenTensorDescriptor_t dODesc,
                                                     const miopenTensorDescriptor_t dADesc,
                                                     const miopenTensorDescriptor_t dPDesc,
                                                     const miopenTensorDescriptor_t dNDesc,
                                                     const Tgpu* anchor,
                                                     const Tgpu* positive,
                                                     const Tgpu* negative,
                                                     const Tgpu* dO,
                                                     Tcheck* dAhost,
                                                     Tcheck* dPhost,
                                                     Tcheck* dNhost,
                                                     const float margin,
                                                     const int p,
                                                     const float eps,
                                                     const bool swap)
{
    auto A_tv  = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(aDesc));
    auto P_tv  = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(pDesc));
    auto N_tv  = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(nDesc));
    auto dO_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<1>(miopen::deref(dODesc));
    auto dA_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(dADesc));
    auto dP_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(dPDesc));
    auto dN_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(dNDesc));

    par_ford(A_tv.size[0])([&](int gid) {
        size_t C = A_tv.size[1];
        size_t b = gid;

        if(b >= A_tv.size[0])
            return;

        float ap = dist<Tgpu>(anchor, positive, p, eps, A_tv, P_tv, b, C);
        float an = dist<Tgpu>(anchor, negative, p, eps, A_tv, N_tv, b, C);
        float pn = dist<Tgpu>(positive, negative, p, eps, P_tv, N_tv, b, C);

        bool swapped = true;
        if(swap && pn < an)
        {
            an      = pn;
            swapped = true;
        }

        float grad_output = static_cast<float>(dO[dO_tv.stride[0] * b]);

        par_ford(C)([&](int c) {
            if(ap - an + margin > 0)
            {
                float a   = static_cast<float>(anchor[A_tv.stride[0] * b + A_tv.stride[1] * c]);
                float pos = static_cast<float>(positive[P_tv.stride[0] * b + P_tv.stride[1] * c]);
                float neg = static_cast<float>(negative[N_tv.stride[0] * b + N_tv.stride[1] * c]);
                float l = 0.0f, grad = 0.0f;
                if(dAhost)
                {
                    grad = 0.0f;
                    l    = std::pow(std::fabs(a - pos) + eps, (p - 1)) * std::pow(ap, (1 - p));
                    if(a < pos)
                        l = -l;
                    grad += l * grad_output;
                    if(!swapped)
                    {
                        l = -std::pow(std::fabs(a - neg) + eps, (p - 1)) * std::pow(an, (1 - p));
                        if(a < neg)
                            l = -l;
                        grad += l * grad_output;
                    }
                    dAhost[dA_tv.stride[0] * b + dA_tv.stride[1] * c] = static_cast<Tcheck>(grad);
                }
                if(dPhost)
                {
                    grad = 0.0f;
                    l    = -std::pow(std::fabs(a - pos) + eps, (p - 1)) * std::pow(ap, (1 - p));
                    if(a < pos)
                        l = -l;
                    grad += l * grad_output;
                    if(swapped)
                    {
                        l = -std::pow(std::fabs(pos - neg) + eps, (p - 1)) * std::pow(pn, (1 - p));
                        if(pos < neg)
                            l = -l;
                        grad += l * grad_output;
                    }
                    dPhost[dP_tv.stride[0] * b + dP_tv.stride[1] * c] = static_cast<Tcheck>(grad);
                }
                if(dNhost)
                {
                    if(swapped)
                    {
                        l = std::pow(std::fabs(pos - neg) + eps, (p - 1)) * std::pow(pn, (1 - p));
                        if(pos < neg)
                            l = -l;
                    }
                    else
                    {
                        l = std::pow(std::fabs(a - neg) + eps, (p - 1)) * std::pow(an, (1 - p));
                        if(a < neg)
                            l = -l;
                    }
                    dNhost[dN_tv.stride[0] * b + dN_tv.stride[1] * c] =
                        static_cast<Tcheck>(l * grad_output);
                }
            }
            else
            {
                if(dAhost)
                    dAhost[dA_tv.stride[0] * b + dA_tv.stride[1] * c] = static_cast<Tcheck>(0.0f);
                if(dPhost)
                    dPhost[dP_tv.stride[0] * b + dP_tv.stride[1] * c] = static_cast<Tcheck>(0.0f);
                if(dNhost)
                    dNhost[dN_tv.stride[0] * b + dN_tv.stride[1] * c] = static_cast<Tcheck>(0.0f);
            }
        });
    });

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloTripletMarginLossBackwardRunHost(const miopenTensorDescriptor_t aDesc,
                                            const miopenTensorDescriptor_t pDesc,
                                            const miopenTensorDescriptor_t nDesc,
                                            const miopenTensorDescriptor_t dADesc,
                                            const miopenTensorDescriptor_t dPDesc,
                                            const miopenTensorDescriptor_t dNDesc,
                                            const Tgpu* anchor,
                                            const Tgpu* positive,
                                            const Tgpu* negative,
                                            const Tgpu* dO,
                                            Tcheck* dAhost,
                                            Tcheck* dPhost,
                                            Tcheck* dNhost,
                                            const float margin,
                                            const int p,
                                            const float eps,
                                            const bool swap,
                                            const float divisor)
{
    auto A_tv  = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(aDesc));
    auto P_tv  = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(pDesc));
    auto N_tv  = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(nDesc));
    auto dA_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(dADesc));
    auto dP_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(dPDesc));
    auto dN_tv = miopen::solver::tripletmarginloss::get_inner_expanded_tv<2>(miopen::deref(dNDesc));

    par_ford(A_tv.size[0])([&](int gid) {
        size_t C = A_tv.size[1];
        size_t b = gid;

        if(b >= A_tv.size[0])
            return;

        float ap = dist<Tgpu>(anchor, positive, p, eps, A_tv, P_tv, b, C);
        float an = dist<Tgpu>(anchor, negative, p, eps, A_tv, N_tv, b, C);
        float pn = dist<Tgpu>(positive, negative, p, eps, P_tv, N_tv, b, C);

        bool swapped = true;
        if(swap && pn < an)
        {
            an      = pn;
            swapped = true;
        }

        float grad_output = static_cast<float>(dO[0]);

        par_ford(C)([&](int c) {
            if(ap - an + margin > 0)
            {
                float a   = static_cast<float>(anchor[A_tv.stride[0] * b + A_tv.stride[1] * c]);
                float pos = static_cast<float>(positive[P_tv.stride[0] * b + P_tv.stride[1] * c]);
                float neg = static_cast<float>(negative[N_tv.stride[0] * b + N_tv.stride[1] * c]);
                float l = 0.0f, grad = 0.0f;
                if(dAhost)
                {
                    grad = 0.0f;
                    l    = std::pow(std::fabs(a - pos) + eps, (p - 1)) * std::pow(ap, (1 - p));
                    if(a < pos)
                        l = -l;
                    grad += l * grad_output;
                    if(!swapped)
                    {
                        l = -std::pow(std::fabs(a - neg) + eps, (p - 1)) * std::pow(an, (1 - p));
                        if(a < neg)
                            l = -l;
                        grad += l * grad_output;
                    }
                    dAhost[dA_tv.stride[0] * b + dA_tv.stride[1] * c] =
                        static_cast<Tcheck>(grad / divisor);
                }
                if(dPhost)
                {
                    grad = 0.0f;
                    l    = -std::pow(std::fabs(a - pos) + eps, (p - 1)) * std::pow(ap, (1 - p));
                    if(a < pos)
                        l = -l;
                    grad += l * grad_output;
                    if(swapped)
                    {
                        l = -std::pow(std::fabs(pos - neg) + eps, (p - 1)) * std::pow(pn, (1 - p));
                        if(pos < neg)
                            l = -l;
                        grad += l * grad_output;
                    }
                    dPhost[dP_tv.stride[0] * b + dP_tv.stride[1] * c] =
                        static_cast<Tcheck>(grad / divisor);
                }
                if(dNhost)
                {
                    if(swapped)
                    {
                        l = std::pow(std::fabs(pos - neg) + eps, (p - 1)) * std::pow(pn, (1 - p));
                        if(pos < neg)
                            l = -l;
                    }
                    else
                    {
                        l = std::pow(std::fabs(a - neg) + eps, (p - 1)) * std::pow(an, (1 - p));
                        if(a < neg)
                            l = -l;
                    }
                    dNhost[dN_tv.stride[0] * b + dN_tv.stride[1] * c] =
                        static_cast<Tcheck>(l * grad_output / divisor);
                }
            }
            else
            {
                if(dAhost)
                    dAhost[dA_tv.stride[0] * b + dA_tv.stride[1] * c] = static_cast<Tcheck>(0.0f);
                if(dPhost)
                    dPhost[dP_tv.stride[0] * b + dP_tv.stride[1] * c] = static_cast<Tcheck>(0.0f);
                if(dNhost)
                    dNhost[dN_tv.stride[0] * b + dN_tv.stride[1] * c] = static_cast<Tcheck>(0.0f);
            }
        });
    });

    return miopenStatusSuccess;
}
