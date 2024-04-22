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

#include "miopen/miopen.h"
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace take {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& indexDesc_,
                       const TensorDescriptor& yDesc_)
        : xDesc(xDesc_), indexDesc(indexDesc_), yDesc(yDesc_)
    {
    }

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetIndexDesc() const { return indexDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType())
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Take: Tensor types do not match.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsRightLength() const
    {
        int32_t posy = 0;
        for(int32_t i = 0; i < indexDesc.GetLengths().size(); i++)
        {
            if(indexDesc.GetLengths()[i] != yDesc.GetLengths()[posy])
            {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm, "Take: Tensor dimension lengths do not match.");
#else
                return false;
#endif
            }

            posy++;
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(!(xDesc.IsPacked() && indexDesc.IsPacked() && yDesc.IsPacked()))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Take: Unpacked tensors not supported.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsInt32Index() const
    {
        if(indexDesc.GetType() != miopenInt32)
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Take: Tensor index must be an int32 tensor.");
#else
            return false;
#endif
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor xDesc;
    TensorDescriptor indexDesc;
    TensorDescriptor yDesc;
};

} // namespace take

} // namespace miopen
