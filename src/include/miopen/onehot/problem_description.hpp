/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace onehot {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& inDesc_,
                       const TensorDescriptor& outDesc_,
                       long inputSize_,
                       int numClasses_)
        : inDesc(inDesc_), outDesc(outDesc_), inputSize(inputSize_), numClasses(numClasses_)
    {
    }

    const TensorDescriptor& GetInDesc() const { return inDesc; }
    const TensorDescriptor& GetOutDesc() const { return outDesc; }
    long getInputSize() const { return inputSize; }
    int getNumClasses() const { return numClasses; }

    NetworkConfig MakeNetworkConfig() const override;

    bool IsSameType() const
    {
        if(inDesc.GetType() != outDesc.GetType())
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Reduce: Tensor types do not match.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(!(inDesc.IsPacked() && outDesc.IsPacked()))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Reduce: Unpacked tensors not supported.");
#else
            return false;
#endif
        }
        return true;
    }

private:
    TensorDescriptor inDesc;
    TensorDescriptor outDesc;
    long inputSize;
    int numClasses;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace onehot

} // namespace miopen
