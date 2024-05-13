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

namespace SGD {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& paramInDesc_,
                       const TensorDescriptor& paramOutDesc_,
                       const TensorDescriptor& gradDesc_,
                       const TensorDescriptor& momentumBufferInDesc_,
                       const TensorDescriptor& momentumBufferOutDesc_,
                       double lr_,
                       double momentum_,
                       double dampening_,
                       double weightDecay_,
                       char nesterov_,
                       char momentumInitialized_)
        : paramInDesc(paramInDesc_),
          paramOutDesc(paramOutDesc_),
          gradDesc(gradDesc_),
          momentumBufferInDesc(momentumBufferInDesc_),
          momentumBufferOutDesc(momentumBufferOutDesc_),
          lr(lr_),
          momentum(momentum_),
          dampening(dampening_),
          weightDecay(weightDecay_),
          nesterov(nesterov_),
          momentumInitialized(momentumInitialized_)
    {
    }

    const TensorDescriptor& GetParamInDesc() const { return paramInDesc; }
    const TensorDescriptor& GetParamOutDesc() const { return paramOutDesc; }
    const TensorDescriptor& GetGradDesc() const { return gradDesc; }
    const TensorDescriptor& GetMomentumBufferInDesc() const { return momentumBufferInDesc; }
    const TensorDescriptor& GetMomentumBufferOutDesc() const { return momentumBufferOutDesc; }

    bool IsSameType() const
    {
        if(paramInDesc.GetType() != paramOutDesc.GetType())
        {
            return false;
        }
        if(paramOutDesc.GetType() != gradDesc.GetType())
        {
            return false;
        }
        if(gradDesc.GetType() != momentumBufferInDesc.GetType())
        {
            return false;
        }
        if(momentumBufferInDesc.GetType() != momentumBufferOutDesc.GetType())
        {
            return false;
        }
        return true;
    }

    bool IsRightLength() const
    {
        for(int32_t i = 0; i < paramInDesc.GetLengths().size(); ++i)
        {
            size_t len = paramInDesc.GetLengths()[i];
            if(paramOutDesc.GetLengths()[i] != len)
            {
                return false;
            }
            if(gradDesc.GetLengths()[i] != len)
            {
                return false;
            }
            if(momentumBufferInDesc.GetLengths()[i] != len)
            {
                return false;
            }
            if(momentumBufferOutDesc.GetLengths()[i] != len)
            {
                return false;
            }
        }
        return true;
    }

    bool IsContiguous(const TensorDescriptor& tensor) const
    {
        std::vector<size_t> lengths = tensor.GetLengths();
        std::vector<size_t> strides = tensor.GetStrides();
        size_t n_dims               = lengths.size();

        size_t expected_stride = 1;

        for(int i = n_dims - 1; i >= 0; --i)
        {
            if(strides[i] != expected_stride)
            {
                return false;
            }
            expected_stride *= lengths[i];
        }

        return true;
    }

    bool IsSameStrides() const
    {
        std::vector<size_t> paramInStrides           = paramInDesc.GetStrides();
        std::vector<size_t> paramOutStrides          = paramOutDesc.GetStrides();
        std::vector<size_t> gradStrides              = gradDesc.GetStrides();
        std::vector<size_t> momentumBufferInStrides  = momentumBufferInDesc.GetStrides();
        std::vector<size_t> momentumBufferOutStrides = momentumBufferOutDesc.GetStrides();

        if(paramInStrides != paramOutStrides)
        {
            return false;
        }
        if(paramOutStrides != gradStrides)
        {
            return false;
        }
        if(gradStrides != momentumBufferInStrides)
        {
            return false;
        }
        if(momentumBufferInStrides != momentumBufferOutStrides)
        {
            return false;
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor paramInDesc;
    TensorDescriptor paramOutDesc;
    TensorDescriptor gradDesc;
    TensorDescriptor momentumBufferInDesc;
    TensorDescriptor momentumBufferOutDesc;
    double lr                = 0;
    double momentum          = 0;
    double dampening         = 0;
    double weightDecay       = 0;
    char nesterov            = 0;
    char momentumInitialized = 0;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace SGD
} // namespace miopen
