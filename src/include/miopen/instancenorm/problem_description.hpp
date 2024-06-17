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

#include "miopen/errors.hpp"
#include "miopen/miopen.h"
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace instancenorm {

bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y);

struct InstanceNormFwdProblemDescription : ProblemDescriptionBase
{
    InstanceNormFwdProblemDescription(const TensorDescriptor& inputDesc_,
                                      const TensorDescriptor& outputDesc_,
                                      const TensorDescriptor& weightDesc_,
                                      const TensorDescriptor& biasDesc_,
                                      const TensorDescriptor& meanInDesc_,
                                      const TensorDescriptor& varInDesc_,
                                      const TensorDescriptor& meanOutDesc_,
                                      const TensorDescriptor& varOutDesc_,
                                      const TensorDescriptor& meanVarDesc_,
                                      const bool useInputStats_)
        : inputDesc(inputDesc_),
          outputDesc(outputDesc_),
          weightDesc(weightDesc_),
          biasDesc(biasDesc_),
          meanInDesc(meanInDesc_),
          varInDesc(varInDesc_),
          meanOutDesc(meanOutDesc_),
          varOutDesc(varOutDesc_),
          meanVarDesc(meanVarDesc_),
          useInputStats(useInputStats_)
    {

        IsValidSize();
        IsValidLength();
    }

    bool IsValidSize() const
    {
        if(inputDesc.GetSize() < 2 || inputDesc.GetSize() > 5)
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm,
                         "Instance Norm: The input tensor dimension should be in range [2, 5].");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsValidLength() const
    {
        auto input_dims = inputDesc.GetLengths();
        if(weightDesc.GetSize() != 1 || biasDesc.GetSize() != 1 ||
           weightDesc.GetLengths()[0] != input_dims[1] || biasDesc.GetLengths()[0] != input_dims[1])
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(
                miopenStatusBadParm,
                "Instance Norm: The input tensor and weight, bias tensor size don't match.");
#else
            return false;
#endif
        }
        if(meanInDesc.GetSize() != 1 || varInDesc.GetSize() != 1 ||
           meanInDesc.GetLengths()[0] != input_dims[1] ||
           varInDesc.GetLengths()[0] != input_dims[1])
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm,
                         "Instance Norm: The input tensor and running_mean_in, running_var_in "
                         "tensor size don't match.");
#else
            return false;
#endif
        }
        if(meanOutDesc.GetSize() != 1 || varOutDesc.GetSize() != 1 ||
           meanOutDesc.GetLengths()[0] != input_dims[1] ||
           varOutDesc.GetLengths()[0] != input_dims[1])
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm,
                         "Instance Norm: The input tensor and running_mean_out, running_var_out "
                         "tensor size don't match.");
#else
            return false;
#endif
        }
        if(meanVarDesc.GetSize() != 2 || meanVarDesc.GetLengths()[0] != input_dims[0] ||
           meanVarDesc.GetLengths()[1] != (input_dims[1] * 2))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm,
                         "The input tensor and mean, var tensor size don't match.");
#else
            return false;
#endif
        }
        return true;
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetWeightDesc() const { return weightDesc; }
    const TensorDescriptor& GetBiasDesc() const { return biasDesc; }
    const TensorDescriptor& GetMeanInDesc() const { return meanInDesc; }
    const TensorDescriptor& GetVarInDesc() const { return varInDesc; }
    const TensorDescriptor& GetMeanOutDesc() const { return meanOutDesc; }
    const TensorDescriptor& GetVarOutDesc() const { return varOutDesc; }
    const TensorDescriptor& GetMeanVarDesc() const { return meanVarDesc; }
    bool IsUseInputStats() const { return useInputStats; }

    NetworkConfig MakeNetworkConfig() const override;

public:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor weightDesc;
    TensorDescriptor biasDesc;
    TensorDescriptor meanInDesc;
    TensorDescriptor varInDesc;
    TensorDescriptor meanOutDesc;
    TensorDescriptor varOutDesc;
    TensorDescriptor meanVarDesc;
    bool useInputStats;
};

struct InstanceNormBwdProblemDescription : ProblemDescriptionBase
{
    InstanceNormBwdProblemDescription(const TensorDescriptor& inputDesc_,
                                      const TensorDescriptor& doutputDesc_,
                                      const TensorDescriptor& weightDesc_,
                                      const TensorDescriptor& meanVarDesc_,
                                      const TensorDescriptor& dinputDesc_,
                                      const TensorDescriptor& dweightDesc_,
                                      const TensorDescriptor& biasGradDesc_)
        : inputDesc(inputDesc_),
          doutputDesc(doutputDesc_),
          weightDesc(weightDesc_),
          meanVarDesc(meanVarDesc_),
          dinputDesc(dinputDesc_),
          dweightDesc(dweightDesc_),
          dbiasDesc(biasGradDesc_)
    {
        IsValidSize();
        IsValidLength();
    }

    bool IsValidSize() const
    {
        if(inputDesc.GetSize() < 2 || inputDesc.GetSize() > 5)
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm,
                         "Instance Norm: The input tensor dimension should be in range [2, 5].");
#else
            return false;
#endif
        }
        if(!checkSameLength(inputDesc, doutputDesc))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(
                miopenStatusBadParm,
                "Instance Norm: The input tensor and output grad tensor size don't match.");
#else
            return false;
#endif
        }
        if(!checkSameLength(inputDesc, dinputDesc))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm,
                         "Instance Norm: The input tensor and input grad tensor size don't match.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsValidLength() const
    {
        auto input_dims = inputDesc.GetLengths();
        if(weightDesc.GetSize() != 1 || weightDesc.GetLengths()[0] != input_dims[1])
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Instance Norm: The weight_tensor has invalid size.");
#else
            return false;
#endif
        }
        if(dweightDesc.GetSize() != 1 || dweightDesc.GetLengths()[0] != input_dims[1])
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm,
                         "Instance Norm: The weight_grad_tensor has invalid size.");
#else
            return false;
#endif
        }
        if(dbiasDesc.GetSize() != 1 || dbiasDesc.GetLengths()[0] != input_dims[1])
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm,
                         "Instance Norm: The bias_grad_tensor has invalid size.");
#else
            return false;
#endif
        }
        if(meanVarDesc.GetSize() != 2 || meanVarDesc.GetLengths()[0] != input_dims[0] ||
           meanVarDesc.GetLengths()[1] != (input_dims[1] * 2))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm,
                         "The input tensor and mean, var tensor size don't match.");
#else
            return false;
#endif
        }
        return true;
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetDoutputDesc() const { return doutputDesc; }
    const TensorDescriptor& GetWeightDesc() const { return weightDesc; }
    const TensorDescriptor& GetMeanVarDesc() const { return meanVarDesc; }
    const TensorDescriptor& GetDinputDesc() const { return dinputDesc; }
    const TensorDescriptor& GetScaleGradDesc() const { return dweightDesc; }
    const TensorDescriptor& GetBiasGradDesc() const { return dbiasDesc; }

    NetworkConfig MakeNetworkConfig() const override;

public:
    TensorDescriptor inputDesc;
    TensorDescriptor doutputDesc;
    TensorDescriptor weightDesc;
    TensorDescriptor meanVarDesc;
    TensorDescriptor dinputDesc;
    TensorDescriptor dweightDesc;
    TensorDescriptor dbiasDesc;
};

} // namespace instancenorm

} // namespace miopen
