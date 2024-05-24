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
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace tripletmarginloss {

bool checkSameType(const TensorDescriptor& x, const TensorDescriptor& y);
bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y);
bool checkSameStride(const TensorDescriptor& x, const TensorDescriptor& y);

struct ForwardProblemDescription : ProblemDescriptionBase
{

    ForwardProblemDescription(const TensorDescriptor& aDesc_,
                              const TensorDescriptor& pDesc_,
                              const TensorDescriptor& nDesc_,
                              const TensorDescriptor& oDesc_)
        : aDesc(aDesc_), pDesc(pDesc_), nDesc(nDesc_), oDesc(oDesc_)
    {
    }

    const TensorDescriptor& GetADesc() const { return aDesc; }
    const TensorDescriptor& GetPDesc() const { return pDesc; }
    const TensorDescriptor& GetNDesc() const { return nDesc; }
    const TensorDescriptor& GetODesc() const { return oDesc; }

    bool IsSameType() const
    {
        if(!checkSameType(aDesc, pDesc) || !checkSameType(aDesc, nDesc))
            MIOPEN_THROW(
                miopenStatusBadParm,
                "Triplet Margin Loss: Anchor, Positive, Negative tensor must have same type.");
        return true;
    }

    bool IsRightLength() const
    {
        if(aDesc.GetSize() != 2 || pDesc.GetSize() != 2 || nDesc.GetSize() != 2)
            MIOPEN_THROW(
                miopenStatusBadParm,
                "Triplet Margin Loss: Anchor, Positive, Negative tensor must have 2 dimensions.");
        if(oDesc.GetSize() != 1)
            MIOPEN_THROW(miopenStatusBadParm,
                         "Triplet Margin Loss: Output tensor must have 1 dimension.");
        if(!checkSameLength(aDesc, pDesc) || !checkSameLength(aDesc, nDesc))
            MIOPEN_THROW(
                miopenStatusBadParm,
                "Triplet Margin Loss: Anchor, Positive, Negative tensor sizes do not match.");
        return true;
    }

    bool IsReduced() const
    {
        if(oDesc.GetElementSize() != 1)
            return false;
        return true;
    }

    bool IsUnreduced() const
    {
        if(oDesc.GetLengths()[0] != aDesc.GetLengths()[0])
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor aDesc;
    TensorDescriptor pDesc;
    TensorDescriptor nDesc;
    TensorDescriptor oDesc;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace tripletmarginloss

} // namespace miopen
