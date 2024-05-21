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

struct PadReflectionFwdProblemDescription : ProblemDescriptionBase
{
    PadReflectionFwdProblemDescription(const TensorDescriptor& xDesc_,
                                       const TensorDescriptor& yDesc_,
                                       const size_t* padding_,
                                       const size_t num_padding_)
        : xDesc(xDesc_), yDesc(yDesc_), padding(padding_), num_padding(num_padding_)
    {
        if(!IsSameType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Tensor types do not match.");
        }
        if(!IsRightNumPadding())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Padding input accepts 1 value only");
        }
        if(!IsRightDim())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Only accept 1d tensor with NCW");
        }
        if(!IsRightOutputSize())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Pad Reflection: Doesn't allow Output_W < padding * 2 + Input_W");
        }
    }

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    size_t GetNumPadding() const { return num_padding; }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType())
        {
            return false;
        }
        return true;
    }

    bool IsRightNumPadding() const
    {
        if(!(num_padding == 1))
        {
            return false;
        }
        return true;
    }

    bool IsRightDim() const
    {
        if(!(xDesc.GetSize() == 3 && yDesc.GetSize() == 3))
        {
            return false;
        }
        return true;
    }

    bool IsContiguous() const { return xDesc.IsContiguous() && yDesc.IsContiguous(); }

    bool IsRightOutputSize() const
    {
        auto input_lens      = xDesc.GetLengths();
        auto output_lens     = yDesc.GetLengths();
        auto input_last_len  = input_lens.back();
        auto output_last_len = output_lens.back();
        auto min_output_size = padding[0] * 2 + input_last_len;
        if(min_output_size > output_last_len)
        {
            return false;
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor& xDesc;
    const TensorDescriptor& yDesc;
    const size_t* padding;
    const size_t num_padding;
};

struct PadReflectionBwdProblemDescription : ProblemDescriptionBase
{
    PadReflectionBwdProblemDescription(const TensorDescriptor& xDesc_,
                                       const TensorDescriptor& yDesc_,
                                       const size_t* padding_,
                                       const size_t num_padding_)
        : xDesc(xDesc_), yDesc(yDesc_), padding(padding_), num_padding(num_padding_)
    {
        if(!IsSameType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Tensor types do not match.");
        }
        if(!IsRightNumPadding())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Padding input accepts 1 value only");
        }
        if(!IsRightDim())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Pad Reflection: Only accept 1d tensor with NCW");
        }
        if(!IsRightOutputSize())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Pad Reflection: Doesn't allow Output_W < padding * 2 + Input_W");
        }
    }

    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    size_t GetNumPadding() const { return num_padding; }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType())
        {
            return false;
        }
        return true;
    }

    bool IsRightNumPadding() const
    {
        if(!(num_padding == 1))
        {
            return false;
        }
        return true;
    }

    bool IsRightDim() const
    {
        if(!(xDesc.GetSize() == 3 && yDesc.GetSize() == 3))
        {
            return false;
        }
        return true;
    }

    bool IsContiguous() const { return xDesc.IsContiguous() && yDesc.IsContiguous(); }

    bool IsRightOutputSize() const
    {
        auto input_lens      = xDesc.GetLengths();
        auto output_lens     = yDesc.GetLengths();
        auto input_last_len  = input_lens.back();
        auto output_last_len = output_lens.back();
        auto min_output_size = padding[0] * 2 + input_last_len;
        if(min_output_size > output_last_len)
        {
            return false;
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor& xDesc;
    const TensorDescriptor& yDesc;
    const size_t* padding;
    const size_t num_padding;
};

} // namespace pad_reflection

} // namespace miopen
