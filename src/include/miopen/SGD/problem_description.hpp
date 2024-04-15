#pragma once

#include "miopen/errors.hpp"
#include "miopen/miopen.h"
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
                       char momentum_initialized_)
        : paramInDesc(paramInDesc_), paramOutDesc(paramOutDesc_), gradDesc(gradDesc_), momentumBufferInDesc(momentumBufferInDesc_), momentumBufferOutDesc(momentumBufferOutDesc_),
        lr(lr_), momentum(momentum_), dampening(dampening_), weightDecay(weightDecay_), nesterov(nesterov_), momentum_initialized(momentum_initialized_) 
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
            if (paramOutDesc.GetLengths()[i] != len)
            {
                return false;
            }
            if (gradDesc.GetLengths()[i] != len)
            {
                return false;
            }
            if (momentumBufferInDesc.GetLengths()[i] != len)
            {
                return false;
            }
            if (momentumBufferOutDesc.GetLengths()[i] != len)
            {
                return false;
            }
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if (!(paramInDesc.IsPacked() && paramOutDesc.IsPacked() && gradDesc.IsPacked() && momentumBufferInDesc.IsPacked() && momentumBufferOutDesc.IsPacked()))
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
    double lr = 0;
    double momentum = 0;
    double dampening = 0;
    double weightDecay = 0;
    char nesterov = 0;
    char momentum_initialized = 0;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace SGD
} // namespace miopen
