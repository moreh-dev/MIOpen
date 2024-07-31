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
#ifndef MLO_VARHOST_H_
#define MLO_VARHOST_H_

#include <miopen/tensor.hpp>

////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////

template <typename Tgpu, typename Tcheck>
int32_t mloVarBackwardRunHost(miopenTensorDescriptor_t inputDesc,
                              miopenTensorDescriptor_t inputGradDesc,
                              miopenTensorDescriptor_t meanDesc,
                              miopenTensorDescriptor_t meanGradDesc,
                              miopenTensorDescriptor_t varGradDesc,
                              Tgpu* input,
                              Tcheck* input_grad,
                              Tgpu* mean,
                              Tgpu* mean_grad,
                              Tgpu* var_grad,
                              int32_t* dims,
                              int32_t num_dims,
                              bool unbiased,
                              int32_t divisor)
{
    auto input_dims      = miopen::deref(inputDesc).GetLengths();
    auto input_grad_dims = miopen::deref(inputGradDesc).GetLengths();
    auto mean_dims       = miopen::deref(meanDesc).GetLengths();
    auto mean_grad_dims  = miopen::deref(meanGradDesc).GetLengths();
    auto var_grad_dims   = miopen::deref(varGradDesc).GetLengths();

    auto input_grad_numel = std::accumulate(
        input_grad_dims.begin(), input_grad_dims.end(), 1LL, std::multiplies<int64_t>());

    std::fill(input_grad, input_grad + input_grad_numel, 0);

    for(size_t gid = 0; gid < input_grad_numel; ++gid)
    {
        std::vector<int64_t> input_idx(input_dims.size(), 0);
        int64_t tmp_gid = gid;

        for(int i = input_dims.size() - 1; i >= 0; --i)
        {
            input_idx[i] = tmp_gid % input_dims[i];
            tmp_gid /= input_dims[i];
        }

        std::vector<int64_t> reduced_idx(input_dims.size(), 0);
        for(int i = 0; i < input_dims.size(); ++i)
        {
            if(std::find(dims, dims + num_dims, i) == dims + num_dims)
            {
                reduced_idx[i] = input_idx[i];
            }
        }

        Tgpu input_v = input[gid];

        int64_t mean_idx    = 0;
        int64_t mean_stride = 1;

        for(int i = reduced_idx.size() - 1; i >= 0; --i)
        {
            mean_idx += reduced_idx[i] * mean_stride;
            mean_stride *= mean_dims[i];
        }

        Tgpu mean_v = static_cast<Tgpu>(0.0);

        if(mean != nullptr)
        {
            mean_v = mean[mean_idx];
        }

        Tgpu input_grad_v = static_cast<Tgpu>(0.0);

        int64_t var_grad_idx    = 0;
        int64_t var_grad_stride = 1;

        for(int i = var_grad_dims.size() - 1; i >= 0; --i)
        {
            var_grad_idx += reduced_idx[i] * var_grad_stride;
            var_grad_stride *= var_grad_dims[i];
        }

        if(var_grad != nullptr)
        {
            Tgpu var_grad_v = var_grad[var_grad_idx];
            Tgpu res        = static_cast<Tgpu>(
                var_grad_v * (static_cast<float>(input_v) - static_cast<float>(mean_v)) * 2.0f);
            input_grad_v +=
                unbiased ? res / static_cast<Tgpu>(divisor - 1) : res / static_cast<Tgpu>(divisor);
        }

        int64_t mean_grad_idx    = 0;
        int64_t mean_grad_stride = 1;

        for(int i = mean_grad_dims.size() - 1; i >= 0; --i)
        {
            mean_grad_idx += reduced_idx[i] * mean_grad_stride;
            mean_grad_stride *= mean_grad_dims[i];
        }

        if(mean_grad != nullptr)
        {
            Tgpu mean_grad_v = mean_grad[mean_grad_idx];
            input_grad_v += mean_grad_v / static_cast<Tgpu>(divisor);
        }

        input_grad[gid] = input_grad_v;
    }

    return 0;
}

template <typename Tgpu>
int32_t mloMeanForwardRunHost(miopenTensorDescriptor_t inputDesc,
                              miopenTensorDescriptor_t meanDesc,
                              Tgpu* input,
                              Tgpu* mean,
                              std::vector<int32_t>& dims_vector,
                              int32_t divisor)
{
    auto input_dims = miopen::deref(inputDesc).GetLengths();
    auto mean_dims  = miopen::deref(meanDesc).GetLengths();

    auto input_numel =
        std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>());

    for(size_t gid = 0; gid < input_numel; ++gid)
    {
        std::vector<int64_t> input_idx(input_dims.size(), 0);
        int64_t tmp_gid = gid;

        for(int i = input_dims.size() - 1; i >= 0; --i)
        {
            input_idx[i] = tmp_gid % input_dims[i];
            tmp_gid /= input_dims[i];
        }

        std::vector<int64_t> reduced_idx(input_dims.size(), 0);
        for(int i = 0; i < input_dims.size(); ++i)
        {
            if(std::find(dims_vector.begin(), dims_vector.end(), i) == dims_vector.end())
            {
                reduced_idx[i] = input_idx[i];
            }
        }

        int64_t mean_idx    = 0;
        int64_t mean_stride = 1;

        for(int i = mean_dims.size() - 1; i >= 0; --i)
        {
            mean_idx += reduced_idx[i] * mean_stride;
            mean_stride *= mean_dims[i];
        }

        mean[mean_idx] += input[gid];
    }

    auto mean_numel =
        std::accumulate(mean_dims.begin(), mean_dims.end(), 1LL, std::multiplies<int64_t>());
    for(size_t i = 0; i < mean_numel; ++i)
    {
        mean[i] /= static_cast<Tgpu>(divisor);
    }

    return 0;
}

#endif // MLO_VARHOST_H_