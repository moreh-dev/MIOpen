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

namespace pad_reflection {

struct PadReflection1dFwdContiguousProblemDescription : ProblemDescriptionBase
{
    PadReflection1dFwdContiguousProblemDescription(const TensorDescriptor& xDesc_,
                                                   const TensorDescriptor& yDesc_,
                                                   const size_t num_padding_)
        : xDesc(xDesc_), yDesc(yDesc_), num_padding(num_padding_)
    {
    }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    size_t GetNumPadding() const { return num_padding; }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType())
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Tensor types do not match.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(!(xDesc.IsPacked() && yDesc.IsPacked()))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Unpacked tensors not supported.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsRightNumPadding() const
    {
        if(!(num_padding == 1))
        {
            // if(!((num_padding == 4 && xDesc.GetSize() == 4) || xDesc.GetSize() == 3))
            // {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Padding input accepts 1 value only");
#else
            return false;
#endif
            // }
        }
        return true;
    }

    bool IsRightDim() const
    {
        if(!(xDesc.GetSize() == 3 && yDesc.GetSize() == 3))
        {
            // if(!((num_padding == 4 && xDesc.GetSize() == 4) || xDesc.GetSize() == 3))
            // {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Padding input accepts 1 value only");
#else
            return false;
#endif
            // }
        }
        return true;
    }
    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor xDesc;
    TensorDescriptor yDesc;
    size_t num_padding;
    // NetworkConfig MakeNetworkConfig() const;
};

struct PadReflection1dFwdProblemDescription : ProblemDescriptionBase
{
    PadReflection1dFwdProblemDescription(const TensorDescriptor& xDesc_,
                                         const TensorDescriptor& yDesc_,
                                         const size_t num_padding_)
        : xDesc(xDesc_), yDesc(yDesc_), num_padding(num_padding_)
    {
    }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    size_t GetNumPadding() const { return num_padding; }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType())
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Tensor types do not match.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(!(xDesc.IsPacked() && yDesc.IsPacked()))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Unpacked tensors not supported.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsRightNumPadding() const
    {
        if(!(num_padding == 1))
        {
            // if(!((num_padding == 4 && xDesc.GetSize() == 4) || xDesc.GetSize() == 3))
            // {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Padding input accepts 1 value only");
#else
            return false;
#endif
            // }
        }
        return true;
    }

    bool IsRightDim() const
    {
        if(!(xDesc.GetSize() == 3 && yDesc.GetSize() == 3))
        {
            // if(!((num_padding == 4 && xDesc.GetSize() == 4) || xDesc.GetSize() == 3))
            // {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Padding input accepts 1 value only");
#else
            return false;
#endif
            // }
        }
        return true;
    }
    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor xDesc;
    TensorDescriptor yDesc;
    size_t num_padding;
    // NetworkConfig MakeNetworkConfig() const;
};

struct PadReflection1dBwdContiguousProblemDescription : ProblemDescriptionBase
{
    PadReflection1dBwdContiguousProblemDescription(const TensorDescriptor& xDesc_,
                                                   const TensorDescriptor& yDesc_,
                                                   const size_t num_padding_)
        : xDesc(xDesc_), yDesc(yDesc_), num_padding(num_padding_)
    {
    }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    size_t GetNumPadding() const { return num_padding; }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType())
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Tensor types do not match.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(!(xDesc.IsPacked() && yDesc.IsPacked()))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Unpacked tensors not supported.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsRightNumPadding() const
    {
        if(!(num_padding == 1))
        {
            // if(!((num_padding == 4 && xDesc.GetSize() == 4) || xDesc.GetSize() == 3))
            // {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Padding input accepts 1 value only");
#else
            return false;
#endif
            // }
        }
        return true;
    }

    bool IsRightDim() const
    {
        if(!(xDesc.GetSize() == 3 && yDesc.GetSize() == 3))
        {
            // if(!((num_padding == 4 && xDesc.GetSize() == 4) || xDesc.GetSize() == 3))
            // {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Padding input accepts 1 value only");
#else
            return false;
#endif
            // }
        }
        return true;
    }
    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor xDesc;
    TensorDescriptor yDesc;
    size_t num_padding;
    // NetworkConfig MakeNetworkConfig() const;
};

struct PadReflection1dBwdProblemDescription : ProblemDescriptionBase
{
    PadReflection1dBwdProblemDescription(const TensorDescriptor& xDesc_,
                                         const TensorDescriptor& yDesc_,
                                         const size_t num_padding_)
        : xDesc(xDesc_), yDesc(yDesc_), num_padding(num_padding_)
    {
    }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    size_t GetNumPadding() const { return num_padding; }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType())
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Tensor types do not match.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(!(xDesc.IsPacked() && yDesc.IsPacked()))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Unpacked tensors not supported.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsRightNumPadding() const
    {
        if(!(num_padding == 1))
        {
            // if(!((num_padding == 4 && xDesc.GetSize() == 4) || xDesc.GetSize() == 3))
            // {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Padding input accepts 1 value only");
#else
            return false;
#endif
            // }
        }
        return true;
    }

    bool IsRightDim() const
    {
        if(!(xDesc.GetSize() == 3 && yDesc.GetSize() == 3))
        {
            // if(!((num_padding == 4 && xDesc.GetSize() == 4) || xDesc.GetSize() == 3))
            // {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Padding input accepts 1 value only");
#else
            return false;
#endif
            // }
        }
        return true;
    }
    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor xDesc;
    TensorDescriptor yDesc;
    size_t num_padding;
    // NetworkConfig MakeNetworkConfig() const;
};

} // namespace pad_reflection

} // namespace miopen
