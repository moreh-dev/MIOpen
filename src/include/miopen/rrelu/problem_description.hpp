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

namespace rrelu {

bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y);
bool checkContiguous(const TensorDescriptor& x);

struct ForwardProblemDescription : ProblemDescriptionBase
{
    ForwardProblemDescription(const TensorDescriptor& inputDesc_,
                              const TensorDescriptor& outputDesc_)
        : inputDesc(inputDesc_), outputDesc(outputDesc_)
    {
        if(!IsSameLength())
            MIOPEN_THROW(miopenStatusBadParm,
                         "RReLU: Input and Output tensor must have same size.");
    }

    ForwardProblemDescription(const TensorDescriptor& inputDesc_,
                              const TensorDescriptor& outputDesc_,
                              const TensorDescriptor& noiseDesc_)
        : inputDesc(inputDesc_), outputDesc(outputDesc_), noiseDesc(noiseDesc_)
    {
        if(!IsSameLength())
            MIOPEN_THROW(miopenStatusBadParm,
                         "RReLU: Input and Output tensor must have same size.");
        if(inputDesc.GetElementSize() != noiseDesc.GetElementSize())
            MIOPEN_THROW(miopenStatusBadParm, "RReLU: Input and Noise tensor must have same size.");
        if(!noiseDesc.IsPacked())
            MIOPEN_THROW(miopenStatusBadParm, "RReLU: Noise tensor must be packed.");
        if(noiseDesc.GetType() != miopenFloat)
            MIOPEN_THROW(miopenStatusBadParm, "RReLU: Noise tensor only works with float32.");
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetNoiseDesc() const { return noiseDesc; }

    bool IsSameLength() const
    {
        if(!checkSameLength(inputDesc, outputDesc))
            return false;
        return true;
    }

    bool IsAllContiguous() const
    {
        if(!checkContiguous(inputDesc))
            return false;
        if(!checkContiguous(outputDesc))
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor noiseDesc;

    NetworkConfig MakeForwardNetworkConfig() const;
};

struct BackwardProblemDescription : ProblemDescriptionBase
{
    BackwardProblemDescription(const TensorDescriptor& noiseDesc_,
                               const TensorDescriptor& doutputDesc_,
                               const TensorDescriptor& dinputDesc_)
        : noiseDesc(noiseDesc_), dinputDesc(dinputDesc_), doutputDesc(doutputDesc_)
    {
        if(!IsSameLength())
            MIOPEN_THROW(miopenStatusBadParm, "RReLU: Tensors must have same size.");
        if(dinputDesc.GetElementSize() != noiseDesc.GetElementSize())
            MIOPEN_THROW(miopenStatusBadParm, "RReLU: Input and Noise tensor must have same size.");
        if(!noiseDesc.IsPacked())
            MIOPEN_THROW(miopenStatusBadParm, "RReLU: Noise tensor must be packed.");
        if(noiseDesc.GetType() != miopenFloat)
            MIOPEN_THROW(miopenStatusBadParm, "RReLU: Noise tensor only works with float32.");
    }

    const TensorDescriptor& GetdInputDesc() const { return dinputDesc; }
    const TensorDescriptor& GetdOutputDesc() const { return doutputDesc; }
    const TensorDescriptor& GetNoiseDesc() const { return noiseDesc; }

    bool IsSameLength() const
    {
        if(!checkSameLength(dinputDesc, doutputDesc))
            return false;
        return true;
    }

    bool IsAllContiguous() const
    {
        if(!checkContiguous(doutputDesc))
            return false;
        if(!checkContiguous(dinputDesc))
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor noiseDesc;
    TensorDescriptor dinputDesc;
    TensorDescriptor doutputDesc;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace rrelu

} // namespace miopen
