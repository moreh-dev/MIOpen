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

#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <../test/ford.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloLPPoolForward1dRunHost(const miopenTensorDescriptor_t inputDesc,
                                  const miopenTensorDescriptor_t outputDesc,
                                  const Tgpu* input,
                                  Tcheck* output,
                                  int64_t N,
                                  int64_t C,
                                  int64_t D,
                                  int64_t OD,
                                  const int64_t* ksize,
                                  const int64_t* stride,
                                  float norm_type)
{
    auto dims  = miopen::deref(inputDesc).GetLengths();
    auto numel = miopen::deref(outputDesc).GetElementSize();

    auto input_tv  = miopen::get_inner_expanded_tv<3>(miopen::deref(inputDesc));
    auto output_tv = miopen::get_inner_expanded_tv<3>(miopen::deref(outputDesc));

    par_ford(numel)([&](int64_t gid) {
        int64_t nc = gid / OD, od = gid % OD;
        int64_t n = nc / C, c = nc % C;
        int64_t kd = ksize[0];
        int64_t sd = stride[0];

        if(n >= N)
            return;

        float sum  = 0;
        float invp = 1.0f / norm_type;
        for(int64_t r = 0; r < kd; ++r)
        {
            int64_t d = od * sd + r;
            if(d < 0 || d >= D)
                continue;

            sum += static_cast<float>(
                pow(static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, d})]), norm_type));
        }

        output[output_tv.get_tensor_view_idx({n, c, od})] = static_cast<Tcheck>(pow(sum, invp));
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloLPPoolForward2dRunHost(const miopenTensorDescriptor_t inputDesc,
                                  const miopenTensorDescriptor_t outputDesc,
                                  const Tgpu* input,
                                  Tcheck* output,
                                  int64_t N,
                                  int64_t C,
                                  int64_t H,
                                  int64_t W,
                                  int64_t OH,
                                  int64_t OW,
                                  const int64_t* ksize,
                                  const int64_t* stride,
                                  float norm_type)
{
    auto dims  = miopen::deref(inputDesc).GetLengths();
    auto numel = miopen::deref(outputDesc).GetElementSize();

    auto input_tv  = miopen::get_inner_expanded_tv<4>(miopen::deref(inputDesc));
    auto output_tv = miopen::get_inner_expanded_tv<4>(miopen::deref(outputDesc));

    par_ford(numel)([&](int64_t gid) {
        int64_t ncoh = gid / OW, ow = gid % OW;
        int64_t nc = ncoh / OH, oh = ncoh % OH;
        int64_t n = nc / C, c = nc % C;
        int64_t kh = ksize[0];
        int64_t kw = ksize[1];
        int64_t sh = stride[0];
        int64_t sw = stride[1];

        if(n >= N)
            return;

        float sum  = 0;
        float invp = 1.0f / norm_type;
        for(int64_t r = 0; r < kh; ++r)
        {
            for(int64_t s = 0; s < kw; ++s)
            {
                int64_t h = oh * sh + r;
                if(h < 0 || h >= H)
                    continue;
                int64_t w = ow * sw + s;
                if(w < 0 || w >= W)
                    continue;

                sum += static_cast<float>(
                    pow(static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, h, w})]),
                        norm_type));
            }
        }

        output[output_tv.get_tensor_view_idx({n, c, oh, ow})] = static_cast<Tcheck>(pow(sum, invp));
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloLPPoolBackward1dRunHost(const miopenTensorDescriptor_t inputDesc,
                                   const miopenTensorDescriptor_t outputDesc,
                                   const miopenTensorDescriptor_t outputGradDesc,
                                   const miopenTensorDescriptor_t inputGradDesc,
                                   const Tgpu* input,
                                   const Tgpu* output,
                                   const Tgpu* output_grad,
                                   Tcheck* input_grad,
                                   int64_t N,
                                   int64_t C,
                                   int64_t D,
                                   int64_t OD,
                                   const int64_t* ksize,
                                   const int64_t* stride,
                                   float norm_type)
{
    auto dims  = miopen::deref(inputGradDesc).GetLengths();
    auto numel = miopen::deref(inputGradDesc).GetElementSize();

    auto input_tv       = miopen::get_inner_expanded_tv<3>(miopen::deref(inputDesc));
    auto output_tv      = miopen::get_inner_expanded_tv<3>(miopen::deref(outputDesc));
    auto output_grad_tv = miopen::get_inner_expanded_tv<3>(miopen::deref(outputGradDesc));
    auto input_grad_tv  = miopen::get_inner_expanded_tv<3>(miopen::deref(inputGradDesc));

    par_ford(numel)([&](int64_t gid) {
        int64_t nc = gid / D, d = gid % D;
        int64_t n = nc / C, c = nc % C;
        int64_t kd = ksize[0];
        int64_t sd = stride[0];

        if(n >= N)
            return;

        float sum     = 0;
        float d_input = static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, d})]);
        for(int64_t r = 0; r < kd; ++r)
        {
            int64_t odsd = d - r;
            if(odsd % sd != 0)
                continue;
            int64_t od = odsd / sd;
            if(od < 0 || od >= OD)
                continue;

            // gradient of p-norm is x_j * |x_j|^{p-2} / |x|_p^{p-1}
            sum += static_cast<float>(output_grad[output_grad_tv.get_tensor_view_idx({n, c, od})]) *
                   d_input * static_cast<float>(pow(d_input, norm_type - 2.0f)) /
                   static_cast<float>(
                       pow(static_cast<float>(output[output_tv.get_tensor_view_idx({n, c, od})]),
                           norm_type - 1.0f));
        }
        input_grad[input_grad_tv.get_tensor_view_idx({n, c, d})] = static_cast<Tcheck>(sum);
    });
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloLPPoolBackward2dRunHost(const miopenTensorDescriptor_t inputDesc,
                                   const miopenTensorDescriptor_t outputDesc,
                                   const miopenTensorDescriptor_t outputGradDesc,
                                   const miopenTensorDescriptor_t inputGradDesc,
                                   const Tgpu* input,
                                   const Tgpu* output,
                                   const Tgpu* output_grad,
                                   Tcheck* input_grad,
                                   int64_t N,
                                   int64_t C,
                                   int64_t H,
                                   int64_t W,
                                   int64_t OH,
                                   int64_t OW,
                                   const int64_t* ksize,
                                   const int64_t* stride,
                                   float norm_type)
{
    auto dims  = miopen::deref(inputGradDesc).GetLengths();
    auto numel = miopen::deref(inputGradDesc).GetElementSize();

    auto input_tv       = miopen::get_inner_expanded_tv<4>(miopen::deref(inputDesc));
    auto output_tv      = miopen::get_inner_expanded_tv<4>(miopen::deref(outputDesc));
    auto output_grad_tv = miopen::get_inner_expanded_tv<4>(miopen::deref(outputGradDesc));
    auto input_grad_tv  = miopen::get_inner_expanded_tv<4>(miopen::deref(inputGradDesc));

    par_ford(numel)([&](int64_t gid) {
        int64_t nch = gid / W, w = gid % W;
        int64_t nc = nch / H, h = nch % H;
        int64_t n = nc / C, c = nc % C;
        int64_t kh = ksize[0];
        int64_t kw = ksize[1];
        int64_t sh = stride[0];
        int64_t sw = stride[1];

        if(n >= N)
            return;

        float sum     = 0;
        float d_input = static_cast<float>(input[input_tv.get_tensor_view_idx({n, c, h, w})]);
        for(int64_t r = 0; r < kh; ++r)
        {
            for(int64_t s = 0; s < kw; ++s)
            {
                int64_t ohsh = h - r;
                if(ohsh % sh != 0)
                    continue;
                int64_t oh = ohsh / sh;
                if(oh < 0 || oh >= OH)
                    continue;
                int64_t owsw = w - s;
                if(owsw % sw != 0)
                    continue;
                int64_t ow = owsw / sw;
                if(ow < 0 || ow >= OW)
                    continue;

                // gradient of p-norm is x_j * |x_j|^{p-2} / |x|_p^{p-1}
                sum +=
                    static_cast<float>(
                        output_grad[output_grad_tv.get_tensor_view_idx({n, c, oh, ow})]) *
                    d_input * static_cast<float>(pow(d_input, norm_type - 2.0f)) /
                    static_cast<float>(pow(
                        static_cast<float>(output[output_tv.get_tensor_view_idx({n, c, oh, ow})]),
                        norm_type - 1.0f));
            }
        }

        input_grad[input_grad_tv.get_tensor_view_idx({n, c, h, w})] = static_cast<Tcheck>(sum);
    });
    return miopenStatusSuccess;
}