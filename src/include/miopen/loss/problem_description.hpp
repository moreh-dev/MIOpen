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
    HingeEmbeddingLossProblemDescription(const TensorDescriptor& iDesc_,
                                         const TensorDescriptor& tDesc_)
        : iDesc(iDesc_), tDesc(tDesc_)
    {
        if(!checkSameLength(iDesc, tDesc))
            MIOPEN_THROW(miopenStatusBadParm, "Loss: Input, target tensor sizes do not match.");
    }

    const TensorDescriptor& GetIDesc() const { return iDesc; }
    const TensorDescriptor& GetTDesc() const { return tDesc; }

public:
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
};

struct HingeEmbeddingLossFwdProblemDescription : HingeEmbeddingLossProblemDescription
{
    HingeEmbeddingLossFwdProblemDescription(const TensorDescriptor& iDesc_,
                                            const TensorDescriptor& tDesc_,
                                            const TensorDescriptor& oDesc_)
        : HingeEmbeddingLossProblemDescription(iDesc_, tDesc_), oDesc(oDesc_)
    {
    }

    NetworkConfig MakeNetworkConfig() const override;
    const TensorDescriptor& GetODesc() const { return oDesc; }

public:
    TensorDescriptor oDesc;
};

struct HingeEmbeddingLossBwdProblemDescription : HingeEmbeddingLossProblemDescription
{
    HingeEmbeddingLossBwdProblemDescription(const TensorDescriptor& iDesc_,
                                            const TensorDescriptor& tDesc_,
                                            const TensorDescriptor& dODesc_,
                                            const TensorDescriptor& dIDesc_)
        : HingeEmbeddingLossProblemDescription(iDesc_, tDesc_), dODesc(dODesc_), dIDesc(dIDesc_)
    {
    }

    NetworkConfig MakeNetworkConfig() const override;
    const TensorDescriptor& GetdODesc() const { return dODesc; }
    const TensorDescriptor& GetdIDesc() const { return dIDesc; }

public:
    TensorDescriptor dODesc;
    TensorDescriptor dIDesc;
};

struct HingeEmbeddingLossUnreducedFwdProblemDescription : HingeEmbeddingLossProblemDescription
{
    HingeEmbeddingLossUnreducedFwdProblemDescription(const TensorDescriptor& iDesc_,
                                                     const TensorDescriptor& tDesc_,
                                                     const TensorDescriptor& oDesc_)
        : HingeEmbeddingLossProblemDescription(iDesc_, tDesc_), oDesc(oDesc_)
    {
    }

    NetworkConfig MakeNetworkConfig() const override;
    const TensorDescriptor& GetODesc() const { return oDesc; }

public:
    TensorDescriptor oDesc;
};

struct HingeEmbeddingLossUnreducedBwdProblemDescription : HingeEmbeddingLossProblemDescription
{
    HingeEmbeddingLossUnreducedBwdProblemDescription(const TensorDescriptor& iDesc_,
                                                     const TensorDescriptor& tDesc_,
                                                     const TensorDescriptor& dODesc_,
                                                     const TensorDescriptor& dIDesc_)
        : HingeEmbeddingLossProblemDescription(iDesc_, tDesc_), dODesc(dODesc_), dIDesc(dIDesc_)
    {
    }

    NetworkConfig MakeNetworkConfig() const override;
    const TensorDescriptor& GetdODesc() const { return dODesc; }
    const TensorDescriptor& GetdIDesc() const { return dIDesc; }

public:
    TensorDescriptor dODesc;
    TensorDescriptor dIDesc;
};

} // namespace loss

} // namespace miopen
