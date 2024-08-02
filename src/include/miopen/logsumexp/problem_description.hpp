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

#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace logsumexp {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& inputDesc_,
                       const TensorDescriptor& outputDesc_,
                       const std::vector<int>& dims_,
                       const bool keepdim_)
        : inputDesc(inputDesc_),
          outputDesc(outputDesc_),
          dims(dims_),
          keepdim(keepdim_),
          isForward(true)
    {
    }

    ProblemDescription(const TensorDescriptor& inputDesc_,
                       const TensorDescriptor& inputGradDesc_,
                       const TensorDescriptor& outputDesc_,
                       const TensorDescriptor& outputGradDesc_,
                       const std::vector<int>& dims_,
                       const bool keepdim_)
        : inputDesc(inputDesc_),
          inputGradDesc(inputGradDesc_),
          outputDesc(outputDesc_),
          outputGradDesc(outputGradDesc_),
          dims(dims_),
          keepdim(keepdim_),
          isForward(false)

    {
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }

    const std::vector<int>& GetDims() const { return dims; }
    bool GetKeepdim() const { return keepdim; }

    bool IsSameType() const
    {
        if(isForward)
        {
            if(!(inputDesc.GetType() == outputDesc.GetType()))
            {
                return false;
            }
        }
        else
        {
            if(!(inputDesc.GetType() == outputDesc.GetType() &&
                 inputDesc.GetType() == inputGradDesc.GetType() &&
                 inputDesc.GetType() == outputGradDesc.GetType()))
            {
                return false;
            }
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(isForward)
        {
            if(!(inputDesc.IsPacked() && outputDesc.IsPacked()))
            {
                return false;
            }
        }
        else
        {
            if(!(inputDesc.IsPacked() && inputGradDesc.IsPacked() && outputDesc.IsPacked() &&
                 outputGradDesc.IsPacked()))
            {
                return false;
            }
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor inputGradDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor outputGradDesc;

    std::vector<int> dims;
    bool keepdim;

    const bool isForward;
};

} // namespace logsumexp

} // namespace miopen
