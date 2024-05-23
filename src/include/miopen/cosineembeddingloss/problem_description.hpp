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

namespace cosineembeddingloss {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& input1Desc_,
                       const TensorDescriptor& input2Desc_,
                       const TensorDescriptor& targetDesc_,
                       const TensorDescriptor& outputDesc_,
                       const float margin_,
                       bool is_fwd_)
        : input1Desc(input1Desc_),
          input2Desc(input2Desc_),
          targetDesc(targetDesc_),
          outputDesc(outputDesc_),
          margin(margin_),
          is_fwd(is_fwd_)
    {
    }

    const TensorDescriptor& GetInput1Desc() const { return input1Desc; }
    const TensorDescriptor& GetInput2Desc() const { return input2Desc; }
    const TensorDescriptor& GetTargetDesc() const { return targetDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    size_t GetNtotal() const { return input1Desc.GetElementSize(); }

    bool IsValidLength() const
    {
        if(targetDesc.GetLengths()[0] != input1Desc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm, "CosineEmbeddingLoss: Tensor sizes do not match.");
        }
        for(int i = 0; i < input1Desc.GetSize(); ++i)
        {
            if(input1Desc.GetLengths()[i] != input2Desc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "CosineEmbeddingLoss: Tensor sizes do not match.");
            }
        }
        if(input1Desc.GetSize() > 2 || input2Desc.GetSize() > 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "CosineEmbeddingLoss: Input tensor size > 2 is not valid.");
        }

        if(targetDesc.GetSize() > 1)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "CosineEmbeddingLoss: Target tensor size > 1 is not valid.");
        }

        if(outputDesc.GetSize() > 1)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "CosineEmbeddingLoss: Output tensor size > 1 is not valid.");
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
                             "CosineEmbeddingLoss: Tensor strides do not valid.");
        }
        return true;
    }

    bool IsAllValidStride() const
    {
        return IsValidStride(input1Desc) && IsValidStride(input2Desc) &&
               IsValidStride(targetDesc) && IsValidStride(outputDesc);
    }

protected:
    TensorDescriptor input1Desc;
    TensorDescriptor input2Desc;
    TensorDescriptor targetDesc;
    TensorDescriptor outputDesc;

    float margin;
    bool is_fwd;

    NetworkConfig MakeForwardNetworkConfig() const;
};

struct FwdUnreducedProblemDescription : ProblemDescription
{
    FwdUnreducedProblemDescription(const TensorDescriptor& input1Desc_,
                                   const TensorDescriptor& input2Desc_,
                                   const TensorDescriptor& targetDesc_,
                                   const TensorDescriptor& outputDesc_,
                                   const float margin_)
        : ProblemDescription(input1Desc_, input2Desc_, targetDesc_, outputDesc_, margin_, true)
    {
        IsValidLength();
        IsAllValidStride();
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    NetworkConfig MakeForwardNetworkConfig() const;
};

struct FwdReducedProblemDescription : ProblemDescription
{
    FwdReducedProblemDescription(const TensorDescriptor& input1Desc_,
                                 const TensorDescriptor& input2Desc_,
                                 const TensorDescriptor& targetDesc_,
                                 const TensorDescriptor& outputDesc_,
                                 const float margin_)
        : ProblemDescription(input1Desc_, input2Desc_, targetDesc_, outputDesc_, margin_, true)
    {
        IsValidLength();
        IsAllValidStride();
    }

    bool IsValidLength() const
    {
        if(outputDesc.GetLengths()[0] != 1)
            MIOPEN_THROW(miopenStatusBadParm,
                         "CosineEmbeddingLoss: Output Tensor length must be (1).");
        if(!ProblemDescription::IsValidLength())
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    NetworkConfig MakeForwardNetworkConfig() const;
};

struct BwdUnreducedProblemDescription : ProblemDescription
{
    BwdUnreducedProblemDescription(const TensorDescriptor& input1Desc_,
                                   const TensorDescriptor& input2Desc_,
                                   const TensorDescriptor& targetDesc_,
                                   const TensorDescriptor& outputGradDesc_,
                                   const TensorDescriptor& input1GradDesc_,
                                   const TensorDescriptor& input2GradDesc_,
                                   const float margin_)
        : ProblemDescription(input1Desc_, input2Desc_, targetDesc_, outputGradDesc_, margin_, false)
    {
        input1GradDesc = input1GradDesc_;
        input2GradDesc = input2GradDesc_;
        IsValidLength();
        IsAllValidStride();
    }

    bool IsAllValidStride() const
    {
        if(!ProblemDescription::IsAllValidStride())
            return false;
        return IsValidStride(input1GradDesc) && IsValidStride(input2GradDesc);
    }

    bool IsValidLength() const
    {
        if(input1Desc.GetSize() > 2 || input2Desc.GetSize() > 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "CosineEmbeddingLoss: Input tensor size > 2 is not valid.");
        }

        for(int i = 0; i < input1Desc.GetSize(); ++i)
        {
            if(input1Desc.GetLengths()[i] != input2Desc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "CosineEmbeddingLoss: Tensor sizes do not match.");
            }
        }

        if(!ProblemDescription::IsValidLength())
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor input1GradDesc;
    TensorDescriptor input2GradDesc;
    NetworkConfig MakeForwardNetworkConfig() const;
};

struct BwdReducedProblemDescription : ProblemDescription
{
    BwdReducedProblemDescription(const TensorDescriptor& input1Desc_,
                                 const TensorDescriptor& input2Desc_,
                                 const TensorDescriptor& targetDesc_,
                                 const TensorDescriptor& outputGradDesc_,
                                 const TensorDescriptor& input1GradDesc_,
                                 const TensorDescriptor& input2GradDesc_,
                                 const float margin_)
        : ProblemDescription(input1Desc_, input2Desc_, targetDesc_, outputGradDesc_, margin_, false)
    {
        input1GradDesc = input1GradDesc_;
        input2GradDesc = input2GradDesc_;
        IsValidLength();
        IsAllValidStride();
    }

    bool IsAllValidStride() const
    {
        if(!ProblemDescription::IsAllValidStride())
            return false;
        return IsValidStride(input1GradDesc) && IsValidStride(input2GradDesc);
    }

    bool IsValidLength() const
    {
        if(outputDesc.GetLengths()[0] != 1)
            MIOPEN_THROW(miopenStatusBadParm,
                         "CosineEmbeddingLoss: Output Tensor length must be (1).");

        if(input1Desc.GetSize() > 2 || input2Desc.GetSize() > 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "CosineEmbeddingLoss: Input tensor size > 2 is not valid.");
        }

        for(int i = 0; i < input1Desc.GetSize(); ++i)
        {
            if(input1Desc.GetLengths()[i] != input2Desc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "CosineEmbeddingLoss: Tensor sizes do not match.");
            }
        }

        if(!ProblemDescription::IsValidLength())
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor input1GradDesc;
    TensorDescriptor input2GradDesc;
    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace cosineembeddingloss

} // namespace miopen
