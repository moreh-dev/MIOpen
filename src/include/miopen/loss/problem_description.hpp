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

#include "miopen/errors.hpp"
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace loss {

bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y);

struct HingeEmbeddingLossProblemDescription : ProblemDescriptionBase
{
    HingeEmbeddingLossProblemDescription(const TensorDescriptor& inputDesc_,
                                         const TensorDescriptor& targetDesc_)
        : inputDesc(inputDesc_), targetDesc(targetDesc_)
    {
        if(!checkSameLength(inputDesc, targetDesc))
            MIOPEN_THROW(miopenStatusBadParm, "Loss: Input, target tensor sizes do not match.");
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetTargetDesc() const { return targetDesc; }

public:
    TensorDescriptor inputDesc;
    TensorDescriptor targetDesc;
};

struct HingeEmbeddingLossFwdProblemDescription : HingeEmbeddingLossProblemDescription
{
    HingeEmbeddingLossFwdProblemDescription(const TensorDescriptor& inputDesc_,
                                            const TensorDescriptor& targetDesc_,
                                            const TensorDescriptor& outputDesc_)
        : HingeEmbeddingLossProblemDescription(inputDesc_, targetDesc_), outputDesc(outputDesc_)
    {
    }

    NetworkConfig MakeNetworkConfig() const override;
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

public:
    TensorDescriptor outputDesc;
};

struct HingeEmbeddingLossBwdProblemDescription : HingeEmbeddingLossProblemDescription
{
    HingeEmbeddingLossBwdProblemDescription(const TensorDescriptor& inputDesc_,
                                            const TensorDescriptor& targetDesc_,
                                            const TensorDescriptor& doutputDesc_,
                                            const TensorDescriptor& dinputDesc_)
        : HingeEmbeddingLossProblemDescription(inputDesc_, targetDesc_),
          doutputDesc(doutputDesc_),
          dinputDesc(dinputDesc_)
    {
    }

    NetworkConfig MakeNetworkConfig() const override;
    const TensorDescriptor& GetDoutputDesc() const { return doutputDesc; }
    const TensorDescriptor& GetDinputDesc() const { return dinputDesc; }

public:
    TensorDescriptor doutputDesc;
    TensorDescriptor dinputDesc;
};

struct HingeEmbeddingLossUnreducedFwdProblemDescription : HingeEmbeddingLossProblemDescription
{
    HingeEmbeddingLossUnreducedFwdProblemDescription(const TensorDescriptor& inputDesc_,
                                                     const TensorDescriptor& targetDesc_,
                                                     const TensorDescriptor& outputDesc_)
        : HingeEmbeddingLossProblemDescription(inputDesc_, targetDesc_), outputDesc(outputDesc_)
    {
    }

    NetworkConfig MakeNetworkConfig() const override;
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

public:
    TensorDescriptor outputDesc;
};

struct HingeEmbeddingLossUnreducedBwdProblemDescription : HingeEmbeddingLossProblemDescription
{
    HingeEmbeddingLossUnreducedBwdProblemDescription(const TensorDescriptor& inputDesc_,
                                                     const TensorDescriptor& targetDesc_,
                                                     const TensorDescriptor& doutputDesc_,
                                                     const TensorDescriptor& dinputDesc_)
        : HingeEmbeddingLossProblemDescription(inputDesc_, targetDesc_),
          doutputDesc(doutputDesc_),
          dinputDesc(dinputDesc_)
    {
    }

    NetworkConfig MakeNetworkConfig() const override;
    const TensorDescriptor& GetDoutputDesc() const { return doutputDesc; }
    const TensorDescriptor& GetDinputDesc() const { return dinputDesc; }

public:
    TensorDescriptor doutputDesc;
    TensorDescriptor dinputDesc;
};

} // namespace loss

} // namespace miopen
