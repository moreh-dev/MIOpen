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

#include <miopen/problem_description_base.hpp>
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>
#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace softmaxcrossentropywithlogits {

struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& targetDesc_,
                          const TensorDescriptor& outputDesc_,
                          const TensorDescriptor& backpropDesc_)
        : inputDesc(inputDesc_),
          targetDesc(targetDesc_),
          outputDesc(outputDesc_),
          backpropDesc(backpropDesc_)
    {
        IsValidLength();
        IsAllValidStride();
        IsTargetProb();
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetTargetDesc() const { return targetDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetBackpropDesc() const { return backpropDesc; }

    size_t GetBatchSize() const { return inputDesc.GetLengths()[0]; }
    size_t GetNumClasses() const { return inputDesc.GetLengths()[1]; }
    size_t GetInputTotal() const { return inputDesc.GetElementSize(); }

    bool IsValidLength() const
    {
        if(inputDesc.GetLengths()[0] != outputDesc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftmaxCrossEntropyWithLogits: Tensor sizes do not match.");
        }
        for(int i = 0; i < inputDesc.GetSize(); ++i)
        {
            if(inputDesc.GetLengths()[i] != targetDesc.GetLengths()[i] ||
               inputDesc.GetLengths()[i] != backpropDesc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "SoftmaxCrossEntropyWithLogits: Tensor sizes do not match.");
            }
        }
        if(inputDesc.GetSize() > 2 || targetDesc.GetSize() > 2 || backpropDesc.GetSize() > 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftmaxCrossEntropyWithLogits: Input tensors size > 2 is not valid.");
        }

        if(outputDesc.GetSize() > 1)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftmaxCrossEntropyWithLogits: Output tensor size > 1 is not valid.");
        }
        return true;
    }

    bool IsValidStride(TensorDescriptor td) const
    {
        auto strides = td.GetStrides();
        auto lengths = td.GetLengths();
        std::vector<std::pair<size_t, size_t>> p;
        p.reserve(td.GetSize());
        std::transform(strides.begin(),
                       strides.end(),
                       lengths.begin(),
                       std::back_inserter(p),
                       [](size_t a, size_t b) { return std::make_pair(a, b); });
        std::sort(p.begin(), p.end());
        for(int i = 1; i < p.size(); ++i)
        {
            if(p[i].first != p[i - 1].first * p[i - 1].second)
                MIOPEN_THROW(miopenStatusBadParm,
                             "SoftmaxCrossEntropyWithLogits: Tensor strides do not valid.");
        }
        return true;
    }

    bool IsAllValidStride() const
    {
        return IsValidStride(inputDesc) && IsValidStride(targetDesc) && IsValidStride(outputDesc) &&
               IsValidStride(backpropDesc);
    }

    bool IsAllContiguous() const
    {
        auto isContiguous = [](TensorDescriptor td) {
            size_t s = 1;
            for(int i = td.GetSize() - 1; i >= 0; --i)
            {
                if(s != td.GetStrides()[i])
                {
                    return false;
                }
                s *= td.GetLengths()[i];
            }
            return true;
        };
        return isContiguous(inputDesc) && isContiguous(targetDesc) && isContiguous(outputDesc) &&
               isContiguous(backpropDesc);
    }

    bool IsTargetProb() const
    {
        // check if each batch of target represents a valid probability distribution
        // i.e. sum of each batch of target is 1 and each element is in [0, 1]

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor inputDesc;
    TensorDescriptor targetDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor backpropDesc;
};

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& backpropDesc_,
                          const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& inputGradDesc_,
                          const TensorDescriptor& targetGradDesc_)
        : outputGradDesc(outputGradDesc_),
          backpropDesc(backpropDesc_),
          inputDesc(inputDesc_),
          inputGradDesc(inputGradDesc_),
          targetGradDesc(targetGradDesc_)
    {
        IsValidLength();
        IsAllValidStride();
    }

    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetBackpropDesc() const { return backpropDesc; }
    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetTargetGradDesc() const { return targetGradDesc; }

    size_t GetNumClasses() const { return inputDesc.GetLengths()[1]; }
    size_t GetInputTotal() const { return inputDesc.GetElementSize(); }

    bool IsValidLength() const
    {
        if(inputDesc.GetLengths()[0] != outputGradDesc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftmaxCrossEntropyWithLogits: Tensor sizes do not match.");
        }
        for(int i = 0; i < inputDesc.GetSize(); ++i)
        {
            if(inputDesc.GetLengths()[i] != backpropDesc.GetLengths()[i] ||
               inputDesc.GetLengths()[i] != inputGradDesc.GetLengths()[i] ||
               inputDesc.GetLengths()[i] != targetGradDesc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "SoftmaxCrossEntropyWithLogits: Tensor sizes do not match.");
            }
        }
        if(backpropDesc.GetSize() > 2 || inputDesc.GetSize() > 2 || inputGradDesc.GetSize() > 2 ||
           targetGradDesc.GetSize() > 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftmaxCrossEntropyWithLogits: Input tensor size > 2 is not valid.");
        }

        if(outputGradDesc.GetSize() > 1)
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "SoftmaxCrossEntropyWithLogits: Output grad tensor size > 1 is not valid.");
        }
        return true;
    }

    bool IsValidStride(TensorDescriptor td) const
    {
        auto strides = td.GetStrides();
        auto lengths = td.GetLengths();
        std::vector<std::pair<size_t, size_t>> p;
        p.reserve(td.GetSize());
        std::transform(strides.begin(),
                       strides.end(),
                       lengths.begin(),
                       std::back_inserter(p),
                       [](size_t a, size_t b) { return std::make_pair(a, b); });
        std::sort(p.begin(), p.end());
        for(int i = 1; i < p.size(); ++i)
        {
            if(p[i].first != p[i - 1].first * p[i - 1].second)
                MIOPEN_THROW(miopenStatusBadParm,
                             "SoftmaxCrossEntropyWithLogits: Tensor strides do not valid.");
        }
        return true;
    }

    bool IsAllValidStride() const
    {
        return IsValidStride(outputGradDesc) && IsValidStride(backpropDesc) &&
               IsValidStride(inputDesc) && IsValidStride(inputGradDesc) &&
               IsValidStride(targetGradDesc);
    }

    bool IsAllContiguous() const
    {
        auto isContiguous = [](TensorDescriptor td) {
            size_t s = 1;
            for(int i = td.GetSize() - 1; i >= 0; --i)
            {
                if(s != td.GetStrides()[i])
                {
                    return false;
                }
                s *= td.GetLengths()[i];
            }
            return true;
        };
        return isContiguous(outputGradDesc) && isContiguous(backpropDesc) &&
               isContiguous(inputDesc) && isContiguous(inputGradDesc) &&
               isContiguous(targetGradDesc);
    }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor outputGradDesc;
    TensorDescriptor backpropDesc;
    TensorDescriptor inputDesc;
    TensorDescriptor inputGradDesc;
    TensorDescriptor targetGradDesc;
};

} // namespace softmaxcrossentropywithlogits

} // namespace miopen
