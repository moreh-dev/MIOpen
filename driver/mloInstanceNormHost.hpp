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
#include "tensor_view.hpp"
#include "miopen/tensor_view_utils.hpp"
#include <cmath>
#include <vector>
#include <miopen/tensor.hpp>

template <typename T>
int32_t mloInstanceNormTrainRunHost(T* input,
                               const miopenTensorDescriptor_t inputDesc,
                               T* ref_output,
                               const miopenTensorDescriptor_t ref_outputDesc,
                               T* scale,
                               const miopenTensorDescriptor_t scaleDesc,
                               T* bias,
                               const miopenTensorDescriptor_t biasDesc,
                               T* ref_meanIn,
                               const miopenTensorDescriptor_t ref_meanInDesc,
                               T* ref_varIn,
                               const miopenTensorDescriptor_t ref_varInDesc,
                               T* ref_meanOut,
                               const miopenTensorDescriptor_t ref_meanOutDesc,
                               T* ref_varOut,
                               const miopenTensorDescriptor_t ref_varOutDesc,
                               T* ref_meanVar,
                               const miopenTensorDescriptor_t ref_meanVarDesc,
                               const float eps          = 1e-05f,
                               const float momentum     = 0.1,
                               const bool useInputStats = true)
{
    auto x_tv                = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto y_tv                = miopen::get_inner_expanded_tv<5>(miopen::deref(ref_outputDesc));
    auto scale_tv            = miopen::get_inner_expanded_tv<1>(miopen::deref(scaleDesc));
    auto bias_tv             = miopen::get_inner_expanded_tv<1>(miopen::deref(biasDesc));
    auto running_mean_in_tv  = miopen::get_inner_expanded_tv<1>(miopen::deref(ref_meanInDesc));
    auto running_var_in_tv   = miopen::get_inner_expanded_tv<1>(miopen::deref(ref_varInDesc));
    auto running_mean_out_tv = miopen::get_inner_expanded_tv<1>(miopen::deref(ref_meanOutDesc));
    auto running_var_out_tv  = miopen::get_inner_expanded_tv<1>(miopen::deref(ref_varOutDesc));
    auto mean_var_tv         = miopen::get_inner_expanded_tv<2>(miopen::deref(ref_meanVarDesc));

    auto input_dims     = miopen::deref(inputDesc).GetLengths();
    auto outer_size     = input_dims[0];
    auto target_size    = input_dims[1];
    uint64_t inner_size = std::accumulate(
        input_dims.begin() + 2, input_dims.end(), 1ULL, std::multiplies<uint64_t>());
    uint64_t gws         = target_size;
    const int LOCAL_SIZE = 256;

    par_ford(gws)([&](uint64_t gid) {
        float ltmp1[LOCAL_SIZE];
        float ltmp2[LOCAL_SIZE];
        float smean = 0, svar = 0;
        ford(outer_size)([&](uint64_t i) {
            float pmean[LOCAL_SIZE] = {0.0f};
            float pvar[LOCAL_SIZE]  = {0.0f};
            par_ford(LOCAL_SIZE)([&](uint64_t lid) {
                for(uint64_t j = lid; j < inner_size; j += LOCAL_SIZE)
                {
                    uint64_t xidx23 = j / x_tv.size[4], xidx4 = j % x_tv.size[4];
                    uint64_t xidx2 = xidx23 / x_tv.size[3], xidx3 = xidx23 % x_tv.size[3];
                    uint64_t xidx0 = i, xidx1 = gid;
                    uint64_t xidx = x_tv.stride[4] * xidx4 + x_tv.stride[3] * xidx3 +
                                    x_tv.stride[2] * xidx2 + x_tv.stride[1] * xidx1 +
                                    x_tv.stride[0] * xidx0;
                    float tmp = static_cast<float>(input[xidx]);
                    pmean[lid] += tmp;
                    pvar[lid] += tmp * tmp;
                }

                ltmp1[lid] = pmean[lid];
                ltmp2[lid] = pvar[lid];
            });

            for(uint64_t j = LOCAL_SIZE >> 1; j > 0; j >>= 1)
            {
                par_ford(LOCAL_SIZE)([&](uint64_t lid) {
                    if(lid < j)
                    {
                        ltmp1[lid] += ltmp1[lid + j];
                        ltmp2[lid] += ltmp2[lid + j];
                    }
                });
            }
            par_ford(LOCAL_SIZE)([&](uint64_t lid) {
                pmean[lid] = ltmp1[0] / inner_size;
                pvar[lid]  = ltmp2[0] / inner_size - pmean[lid] * pmean[lid];
            });

            ref_meanVar[mean_var_tv.stride[1] * (gid * 2) + mean_var_tv.stride[0] * i] = pmean[0];
            ref_meanVar[mean_var_tv.stride[1] * (gid * 2 + 1) + mean_var_tv.stride[0] * i] =
                1 / sqrt(pvar[0] + eps);
            smean += ltmp1[0];
            svar += ltmp2[0];

            float pscale[LOCAL_SIZE];
            float pbias[LOCAL_SIZE];
            par_ford(LOCAL_SIZE)([&](uint64_t lid) {
                pscale[lid] = static_cast<float>(scale[scale_tv.stride[0] * gid]);
                pbias[lid]  = static_cast<float>(bias[bias_tv.stride[0] * gid]);
            });
            par_ford(LOCAL_SIZE)([&](uint64_t lid) {
                for(uint64_t j = lid; j < inner_size; j += LOCAL_SIZE)
                {
                    uint64_t xidx23 = j / x_tv.size[4], xidx4 = j % x_tv.size[4];
                    uint64_t xidx2 = xidx23 / x_tv.size[3], xidx3 = xidx23 % x_tv.size[3];
                    uint64_t xidx0 = i, xidx1 = gid;
                    uint64_t xidx = x_tv.stride[4] * xidx4 + x_tv.stride[3] * xidx3 +
                                    x_tv.stride[2] * xidx2 + x_tv.stride[1] * xidx1 +
                                    x_tv.stride[0] * xidx0;

                    uint64_t yidx23 = j / y_tv.size[4], yidx4 = j % y_tv.size[4];
                    uint64_t yidx2 = yidx23 / y_tv.size[3], yidx3 = yidx23 % y_tv.size[3];
                    uint64_t yidx0 = i, yidx1 = gid;
                    uint64_t yidx = y_tv.stride[4] * yidx4 + y_tv.stride[3] * yidx3 +
                                    y_tv.stride[2] * yidx2 + y_tv.stride[1] * yidx1 +
                                    y_tv.stride[0] * yidx0;

                    ref_output[yidx] =
                        static_cast<T>((static_cast<float>(input[xidx]) - pmean[lid]) *
                                           (1 / sqrt(pvar[lid] + eps)) * pscale[lid] +
                                       pbias[lid]);
                }
            });
        });

        smean = smean / (outer_size * inner_size);
        svar  = svar / (outer_size * inner_size) - smean * smean;
        ref_meanOut[running_mean_out_tv.stride[0] * gid] = static_cast<T>(
            (1 - momentum) * static_cast<float>(ref_meanIn[running_mean_in_tv.stride[0] * gid]) +
            momentum * smean);
        ref_varOut[running_var_out_tv.stride[0] * gid] = static_cast<T>(
            (1 - momentum) * static_cast<float>(ref_varIn[running_var_in_tv.stride[0] * gid]) +
            momentum * svar);
    });

    return miopenStatusSuccess;
}

template <typename T>
int32_t mloInstanceNormTestRunHost(T* input,
                               const miopenTensorDescriptor_t inputDesc,
                               T* ref_output,
                               const miopenTensorDescriptor_t ref_outputDesc,
                               T* scale,
                               const miopenTensorDescriptor_t scaleDesc,
                               T* bias,
                               const miopenTensorDescriptor_t biasDesc,
                               T* ref_meanIn,
                               const miopenTensorDescriptor_t ref_meanInDesc,
                               T* ref_varIn,
                               const miopenTensorDescriptor_t ref_varInDesc,
                               const float eps          = 1e-05f,
                               const bool useInputStats = true)
{
    auto x_tv                = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto y_tv                = miopen::get_inner_expanded_tv<5>(miopen::deref(ref_outputDesc));
    auto scale_tv            = miopen::get_inner_expanded_tv<1>(miopen::deref(scaleDesc));
    auto bias_tv             = miopen::get_inner_expanded_tv<1>(miopen::deref(biasDesc));
    auto running_mean_in_tv  = miopen::get_inner_expanded_tv<1>(miopen::deref(ref_meanInDesc));
    auto running_var_in_tv   = miopen::get_inner_expanded_tv<1>(miopen::deref(ref_varInDesc));

    auto input_dims     = miopen::deref(inputDesc).GetLengths();
    auto outer_size     = input_dims[0];
    auto target_size    = input_dims[1];
    uint64_t inner_size = std::accumulate(
        input_dims.begin() + 2, input_dims.end(), 1ULL, std::multiplies<uint64_t>());
    uint64_t gws         = target_size;
    const int LOCAL_SIZE = 256;

    par_ford(gws)([&](uint64_t gid) {
        float pmean = static_cast<float>(ref_meanIn[running_mean_in_tv.stride[0] * gid]);
        float pvar = static_cast<float>(ref_varIn[running_var_in_tv.stride[0] * gid]);
        float pscale = static_cast<float>(scale[scale_tv.stride[0] * gid]);
        float pbias = static_cast<float>(bias[bias_tv.stride[0] * gid]);
        par_ford(LOCAL_SIZE)([&](uint64_t lid) {
            ford(outer_size)([&](uint64_t i) {
                for(uint64_t j = lid; j < inner_size; j += LOCAL_SIZE)
                {
                    uint64_t xidx23 = j / x_tv.size[4], xidx4 = j % x_tv.size[4];
                    uint64_t xidx2 = xidx23 / x_tv.size[3], xidx3 = xidx23 % x_tv.size[3];
                    uint64_t xidx0 = i, xidx1 = gid;
                    uint64_t xidx = x_tv.stride[4] * xidx4 + x_tv.stride[3] * xidx3 +
                                    x_tv.stride[2] * xidx2 + x_tv.stride[1] * xidx1 +
                                    x_tv.stride[0] * xidx0;

                    uint64_t yidx23 = j / y_tv.size[4], yidx4 = j % y_tv.size[4];
                    uint64_t yidx2 = yidx23 / y_tv.size[3], yidx3 = yidx23 % y_tv.size[3];
                    uint64_t yidx0 = i, yidx1 = gid;
                    uint64_t yidx = y_tv.stride[4] * yidx4 + y_tv.stride[3] * yidx3 +
                                    y_tv.stride[2] * yidx2 + y_tv.stride[1] * yidx1 +
                                    y_tv.stride[0] * yidx0;

                    ref_output[yidx] =
                        static_cast<T>((static_cast<float>(input[xidx]) - pmean) *
                                           (1 / sqrt(pvar + eps)) * pscale +
                                       pbias);
                }
            });
        });
    });

    return miopenStatusSuccess;
}

template <typename T>
int32_t mloInstanceNormBackwardRunHost(T* input,
                            const miopenTensorDescriptor_t inputDesc,
                            T* scale,
                            const miopenTensorDescriptor_t scaleDesc,
                            T* ref_dinput,
                            const miopenTensorDescriptor_t ref_dinputDesc,
                            T* doutput,
                            const miopenTensorDescriptor_t doutputDesc,
                            T* ref_dscale,
                            const miopenTensorDescriptor_t ref_dscaleDesc,
                            T* ref_dbias,
                            const miopenTensorDescriptor_t ref_dbiasDesc,
                            T* meanVar,
                            const miopenTensorDescriptor_t meanVarDesc)
{
    auto x_tv                = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto dy_tv               = miopen::get_inner_expanded_tv<5>(miopen::deref(doutputDesc));
    auto scale_tv            = miopen::get_inner_expanded_tv<1>(miopen::deref(scaleDesc));
    auto mean_var_tv         = miopen::get_inner_expanded_tv<2>(miopen::deref(meanVarDesc));
    auto dx_tv               = miopen::get_inner_expanded_tv<5>(miopen::deref(ref_dinputDesc));
    auto scale_grad_tv       = miopen::get_inner_expanded_tv<1>(miopen::deref(ref_dscaleDesc));
    auto bias_grad_tv        = miopen::get_inner_expanded_tv<1>(miopen::deref(ref_dbiasDesc));

    auto input_dims     = miopen::deref(inputDesc).GetLengths();
    auto outer_size     = input_dims[0];
    auto target_size    = input_dims[1];
    uint64_t inner_size = std::accumulate(
        input_dims.begin() + 2, input_dims.end(), 1ULL, std::multiplies<uint64_t>());
    uint64_t gws         = target_size;
    const int LOCAL_SIZE = 256;

    par_ford(gws)([&](uint64_t gid) {
        float pscale = static_cast<float>(scale[scale_tv.stride[0] * gid]);
        float ltmp1[LOCAL_SIZE];
        float ltmp2[LOCAL_SIZE];
        float sbias_grad = 0.0f;
        float sscale_grad = 0.0f;

        ford(outer_size)([&](uint64_t i) {
            float pmean[LOCAL_SIZE] = {0.0f};
            float pvar[LOCAL_SIZE]  = {0.0f};
            float pbias_grad[LOCAL_SIZE] = {0.0f};
            float pscale_grad[LOCAL_SIZE]  = {0.0f};
            par_ford(LOCAL_SIZE)([&](uint64_t lid) {
                pmean[lid] = static_cast<float>(meanVar[mean_var_tv.stride[1] * (gid * 2) + mean_var_tv.stride[0] * i]);
                pvar[lid] = static_cast<float>(meanVar[mean_var_tv.stride[1] * (gid * 2 + 1) + mean_var_tv.stride[0] * i]);
            });

            par_ford(LOCAL_SIZE)([&](uint64_t lid) {
                for(uint64_t j = lid; j < inner_size; j += LOCAL_SIZE)
                {
                    uint64_t xidx23 = j / x_tv.size[4], xidx4 = j % x_tv.size[4];
                    uint64_t xidx2 = xidx23 / x_tv.size[3], xidx3 = xidx23 % x_tv.size[3];
                    uint64_t xidx0 = i, xidx1 = gid;
                    uint64_t xidx = x_tv.stride[4] * xidx4 + x_tv.stride[3] * xidx3 +
                                    x_tv.stride[2] * xidx2 + x_tv.stride[1] * xidx1 +
                                    x_tv.stride[0] * xidx0;

                    uint64_t dyidx23 = j / dy_tv.size[4], dyidx4 = j % dy_tv.size[4];
                    uint64_t dyidx2 = dyidx23 / dy_tv.size[3], dyidx3 = dyidx23 % dy_tv.size[3];
                    uint64_t dyidx0 = i, dyidx1 = gid;
                    uint64_t dyidx = dy_tv.stride[4] * dyidx4 + dy_tv.stride[3] * dyidx3 +
                                    dy_tv.stride[2] * dyidx2 + dy_tv.stride[1] * dyidx1 +
                                    dy_tv.stride[0] * dyidx0;

                    float xi = static_cast<float>(input[xidx]);
                    float dyi = static_cast<float>(doutput[dyidx]);
                    pbias_grad[lid] += dyi;
                    pscale_grad[lid] += dyi * (xi - pmean[lid]) * pvar[lid];
                }

                ltmp1[lid] = pbias_grad[lid];
                ltmp2[lid] = pscale_grad[lid];
            });

            for(uint64_t j = LOCAL_SIZE >> 1; j > 0; j >>= 1)
            {
                par_ford(LOCAL_SIZE)([&](uint64_t lid) {
                    if(lid < j)
                    {
                        ltmp1[lid] += ltmp1[lid + j];
                        ltmp2[lid] += ltmp2[lid + j];
                    }
                });
            }

            par_ford(LOCAL_SIZE)([&](uint64_t lid) {
                pbias_grad[lid] = ltmp1[0];
                pscale_grad[lid]  = ltmp2[0];
            });

            sbias_grad += pbias_grad[0];
            sscale_grad += pscale_grad[0];

            par_ford(LOCAL_SIZE)([&](uint64_t lid) {
                for(uint64_t j = lid; j < inner_size; j += LOCAL_SIZE)
                {
                    uint64_t xidx23 = j / x_tv.size[4], xidx4 = j % x_tv.size[4];
                    uint64_t xidx2 = xidx23 / x_tv.size[3], xidx3 = xidx23 % x_tv.size[3];
                    uint64_t xidx0 = i, xidx1 = gid;
                    uint64_t xidx = x_tv.stride[4] * xidx4 + x_tv.stride[3] * xidx3 + x_tv.stride[2] * xidx2 + x_tv.stride[1] * xidx1 + x_tv.stride[0] * xidx0;

                    uint64_t dyidx23 = j / dy_tv.size[4], dyidx4 = j % dy_tv.size[4];
                    uint64_t dyidx2 = dyidx23 / dy_tv.size[3], dyidx3 = dyidx23 % dy_tv.size[3];
                    uint64_t dyidx0 = i, dyidx1 = gid;
                    uint64_t dyidx = dy_tv.stride[4] * dyidx4 + dy_tv.stride[3] * dyidx3 + dy_tv.stride[2] * dyidx2 + dy_tv.stride[1] * dyidx1 + dy_tv.stride[0] * dyidx0;

                    uint64_t dxidx23 = j / dx_tv.size[4], dxidx4 = j % dx_tv.size[4];
                    uint64_t dxidx2 = dxidx23 / dx_tv.size[3], dxidx3 = dxidx23 % dx_tv.size[3];
                    uint64_t dxidx0 = i, dxidx1 = gid;
                    uint64_t dxidx = dx_tv.stride[4] * dxidx4 + dx_tv.stride[3] * dxidx3 + dx_tv.stride[2] * dxidx2 + dx_tv.stride[1] * dxidx1 + dx_tv.stride[0] * dxidx0;

                    uint64_t M = inner_size;
                    float dyi = static_cast<float>(doutput[dyidx]);
                    float nxi = (static_cast<float>(input[xidx]) - pmean[lid]) * pvar[lid];
                    ref_dinput[dxidx] =
                        static_cast<T>((M * dyi - pbias_grad[lid] - nxi * pscale_grad[lid]) * pscale * pvar[lid] / M);
                }
            });
        });

        ref_dbias[bias_grad_tv.stride[0] * gid] = static_cast<T>(sbias_grad);
        ref_dscale[scale_grad_tv.stride[0] * gid] = static_cast<T>(sscale_grad);
    });

    return miopenStatusSuccess;
}
