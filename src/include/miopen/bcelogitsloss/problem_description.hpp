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

namespace bcelogitsloss {

bool checkSameType(const TensorDescriptor& x, const TensorDescriptor& y);
bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y);
bool checkSameStride(const TensorDescriptor& x, const TensorDescriptor& y);
bool checkContiguous(const TensorDescriptor& x);

struct BCELogitsLossFwdProblemDescription : ProblemDescriptionBase
{
    BCELogitsLossFwdProblemDescription(const TensorDescriptor& iDesc_,
                                       const TensorDescriptor& tDesc_,
                                       const TensorDescriptor& wDesc_,
                                       const TensorDescriptor& pwDesc_,
                                       const TensorDescriptor& oDesc_)
        : iDesc(iDesc_), tDesc(tDesc_), wDesc(wDesc_), pwDesc(pwDesc_), oDesc(oDesc_)
    {
    }

    const TensorDescriptor& GetIDesc() const { return iDesc; }
    const TensorDescriptor& GetTDesc() const { return tDesc; }
    const TensorDescriptor& GetWDesc() const { return wDesc; }
    const TensorDescriptor& GetPWDesc() const { return pwDesc; }
    const TensorDescriptor& GetODesc() const { return oDesc; }

    bool IsSameType() const
    {
        if(!checkSameType(iDesc, tDesc) || !checkSameType(iDesc, wDesc) ||
           !checkSameType(iDesc, pwDesc))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Reduce: Tensor types do not match.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsRightLength() const
    {
        if(!checkSameLength(iDesc, tDesc))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "BCELogitsLoss: Tensor sizes do not match.");
#else
            return false;
#endif
        }
        return true;
    }

protected:
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
    TensorDescriptor wDesc;
    TensorDescriptor pwDesc;
    TensorDescriptor oDesc;

    NetworkConfig MakeForwardNetworkConfig() const;
};

struct ReducedForwardProblemDescription : BCELogitsLossFwdProblemDescription
{
    ReducedForwardProblemDescription(const TensorDescriptor& iDesc_,
                                     const TensorDescriptor& tDesc_,
                                     const TensorDescriptor& wDesc_,
                                     const TensorDescriptor& pwDesc_,
                                     const TensorDescriptor& oDesc_)
        : BCELogitsLossFwdProblemDescription(iDesc_, tDesc_, wDesc_, pwDesc_, oDesc_)
    {
    }

    bool IsRightLength() const
    {
        if(!BCELogitsLossFwdProblemDescription::IsRightLength())
            return false;
        if(oDesc.GetSize() != 1 || oDesc.GetLengths()[0] != 1)
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "BCELogitsLoss: Output Tensor size must be (1).");
#else
            return false;
#endif
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;
};

struct BCELogitsLossBwdProblemDescription : ProblemDescriptionBase
{
    BCELogitsLossBwdProblemDescription(const TensorDescriptor& iDesc_,
                                       const TensorDescriptor& tDesc_,
                                       const TensorDescriptor& wDesc_,
                                       const TensorDescriptor& pwDesc_,
                                       const TensorDescriptor& doDesc_,
                                       const TensorDescriptor& diDesc_,
                                       const TensorDescriptor& dtDesc_)
        : iDesc(iDesc_),
          tDesc(tDesc_),
          wDesc(wDesc_),
          pwDesc(pwDesc_),
          doDesc(doDesc_),
          diDesc(diDesc_),
          dtDesc(dtDesc_)
    {
    }

    const TensorDescriptor& GetIDesc() const { return iDesc; }
    const TensorDescriptor& GetTDesc() const { return tDesc; }
    const TensorDescriptor& GetWDesc() const { return wDesc; }
    const TensorDescriptor& GetPWDesc() const { return pwDesc; }
    const TensorDescriptor& GetDODesc() const { return doDesc; }
    const TensorDescriptor& GetDIDesc() const { return diDesc; }
    const TensorDescriptor& GetDTDesc() const { return dtDesc; }

    bool IsSameType() const
    {
        if(!checkSameType(iDesc, tDesc) || !checkSameType(iDesc, wDesc) ||
           !checkSameType(iDesc, pwDesc) || !checkSameType(iDesc, diDesc) ||
           !checkSameType(tDesc, dtDesc))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Reduce: Tensor types do not match.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsRightLength() const
    {
        if(!checkSameLength(iDesc, tDesc) || !checkSameLength(iDesc, diDesc) ||
           !checkSameLength(tDesc, dtDesc))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "BCELogitsLoss: Tensor sizes do not match.");
#else
            return false;
#endif
        }
        return true;
    }

protected:
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
    TensorDescriptor wDesc;
    TensorDescriptor pwDesc;
    TensorDescriptor doDesc;
    TensorDescriptor diDesc;
    TensorDescriptor dtDesc;

    NetworkConfig MakeBackwardNetworkConfig() const;
};

struct ReducedBackwardProblemDescription : BCELogitsLossBwdProblemDescription
{
    ReducedBackwardProblemDescription(const TensorDescriptor& iDesc_,
                                      const TensorDescriptor& tDesc_,
                                      const TensorDescriptor& wDesc_,
                                      const TensorDescriptor& pwDesc_,
                                      const TensorDescriptor& doDesc_,
                                      const TensorDescriptor& diDesc_,
                                      const TensorDescriptor& dtDesc_)
        : BCELogitsLossBwdProblemDescription(
              iDesc_, tDesc_, wDesc_, pwDesc_, doDesc_, diDesc_, dtDesc_)
    {
    }

    bool IsRightLength() const
    {
        if(!BCELogitsLossBwdProblemDescription::IsRightLength())
            return false;
        if(doDesc.GetSize() != 1 || doDesc.GetLengths()[0] != 1)
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm,
                         "BCELogitsLoss: Output Gradient Tensor size must be (1).");
#else
            return false;
#endif
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;
};

} // namespace bcelogitsloss

} // namespace miopen
