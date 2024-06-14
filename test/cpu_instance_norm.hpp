#pragma once

#include "miopen/tensor.hpp"
#include "tensor_holder.hpp"
#include "tensor_view.hpp"
#include "miopen/tensor_view_utils.hpp"
#include <cmath>
#include <vector>

template <class T>
void cpu_instance_norm_forward_train(const tensor<T> input,
                               tensor<T>& outputHost,
                               const tensor<T> scale,
                               const tensor<T> bias,
                               const tensor<T> meanInHost,
                               const tensor<T> varInHost,
                               tensor<T>& meanOutHost,
                               tensor<T>& varOutHost,
                               tensor<T>& meanVarHost,
                               const float eps          = 1e-05f,
                               const float momentum     = 0.1,
                               const bool useInputStats = true)
{
    auto x_tv                = miopen::get_inner_expanded_tv<5>(input.desc);
    auto y_tv                = miopen::get_inner_expanded_tv<5>(outputHost.desc);
    auto scale_tv            = miopen::get_inner_expanded_tv<1>(scale.desc);
    auto bias_tv             = miopen::get_inner_expanded_tv<1>(bias.desc);
    auto running_mean_in_tv  = miopen::get_inner_expanded_tv<1>(meanInHost.desc);
    auto running_var_in_tv   = miopen::get_inner_expanded_tv<1>(varInHost.desc);
    auto running_mean_out_tv = miopen::get_inner_expanded_tv<1>(meanOutHost.desc);
    auto running_var_out_tv  = miopen::get_inner_expanded_tv<1>(varOutHost.desc);
    auto mean_var_tv         = miopen::get_inner_expanded_tv<2>(meanVarHost.desc);

    auto input_dims     = input.desc.GetLengths();
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
            meanVarHost[mean_var_tv.stride[1] * (gid * 2) + mean_var_tv.stride[0] * i] = pmean[0];
            meanVarHost[mean_var_tv.stride[1] * (gid * 2 + 1) + mean_var_tv.stride[0] * i] =
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

                    outputHost[yidx] =
                        static_cast<T>((static_cast<float>(input[xidx]) - pmean[lid]) *
                                           (1 / sqrt(pvar[lid] + eps)) * pscale[lid] +
                                       pbias[lid]);
                }
            });
        });

        smean = smean / (outer_size * inner_size);
        svar  = svar / (outer_size * inner_size) - smean * smean;
        meanOutHost[running_mean_out_tv.stride[0] * gid] = static_cast<T>(
            (1 - momentum) * static_cast<float>(meanInHost[running_mean_in_tv.stride[0] * gid]) +
            momentum * smean);
        varOutHost[running_var_out_tv.stride[0] * gid] = static_cast<T>(
            (1 - momentum) * static_cast<float>(varInHost[running_var_in_tv.stride[0] * gid]) +
            momentum * svar);
    });
}

template <class T>
void cpu_instance_norm_forward_test(const tensor<T> input,
                               tensor<T>& outputHost,
                               const tensor<T> scale,
                               const tensor<T> bias,
                               const tensor<T> meanInHost,
                               const tensor<T> varInHost,
                               const float eps          = 1e-05f,
                               const bool useInputStats = true)
{
    auto x_tv                = miopen::get_inner_expanded_tv<5>(input.desc);
    auto y_tv                = miopen::get_inner_expanded_tv<5>(outputHost.desc);
    auto scale_tv            = miopen::get_inner_expanded_tv<1>(scale.desc);
    auto bias_tv             = miopen::get_inner_expanded_tv<1>(bias.desc);
    auto running_mean_in_tv  = miopen::get_inner_expanded_tv<1>(meanInHost.desc);
    auto running_var_in_tv   = miopen::get_inner_expanded_tv<1>(varInHost.desc);

    auto input_dims     = input.desc.GetLengths();
    auto outer_size     = input_dims[0];
    auto target_size    = input_dims[1];
    uint64_t inner_size = std::accumulate(
        input_dims.begin() + 2, input_dims.end(), 1ULL, std::multiplies<uint64_t>());
    uint64_t gws         = target_size;
    const int LOCAL_SIZE = 256;
    par_ford(gws)([&](uint64_t gid) {
        float pmean = static_cast<float>(meanInHost[running_mean_in_tv.stride[0] * gid]);
        float pvar = static_cast<float>(varInHost[running_var_in_tv.stride[0] * gid]);
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

                    outputHost[yidx] =
                        static_cast<T>((static_cast<float>(input[xidx]) - pmean) *
                                           (1 / sqrt(pvar + eps)) * pscale+
                                       pbias);
                }
            });
        });
    });
}

template <class T>
void cpu_instance_norm_backward(const tensor<T> input,
                               const tensor<T> scale,
                               tensor<T>& dinputHost,
                               const tensor<T> doutput,
                               tensor<T>& dscaleHost,
                               tensor<T>& dbiasHost,
                               const tensor<T> meanVar)
{
    auto x_tv                = miopen::get_inner_expanded_tv<5>(input.desc);
    auto dy_tv               = miopen::get_inner_expanded_tv<5>(doutput.desc);
    auto scale_tv            = miopen::get_inner_expanded_tv<1>(scale.desc);
    auto mean_var_tv         = miopen::get_inner_expanded_tv<2>(meanVar.desc);
    auto dx_tv               = miopen::get_inner_expanded_tv<5>(dinputHost.desc);
    auto scale_grad_tv       = miopen::get_inner_expanded_tv<1>(dscaleHost.desc);
    auto bias_grad_tv        = miopen::get_inner_expanded_tv<1>(dbiasHost.desc);

    auto input_dims     = input.desc.GetLengths();
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
                    dinputHost[dxidx] =
                        static_cast<T>((M * dyi - pbias_grad[lid] - nxi * pscale_grad[lid]) * pscale * pvar[lid] / M);
                }
            });
        });

        dbiasHost[bias_grad_tv.stride[0] * gid] = static_cast<T>(sbias_grad);
        dscaleHost[scale_grad_tv.stride[0] * gid] = static_cast<T>(sscale_grad);
    });
}
