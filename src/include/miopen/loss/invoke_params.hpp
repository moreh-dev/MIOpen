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

#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>

#include <limits>

namespace miopen {

namespace loss {

struct HingeEmbeddingLossInvokeParams : public miopen::InvokeParams
{
    HingeEmbeddingLossInvokeParams() = default;

    const TensorDescriptor* inputDesc  = nullptr;
    const TensorDescriptor* targetDesc = nullptr;

    ConstData_t input          = nullptr;
    ConstData_t target         = nullptr;
    Data_t workspace           = nullptr;
    std::size_t workspace_size = 0;
    float margin               = 1.0f;

    std::size_t GetWorkspaceSize() const { return workspace_size; }
    Data_t GetWorkspace() const { return workspace; }
};

struct FwdInvokeParams : HingeEmbeddingLossInvokeParams
{
    FwdInvokeParams() = default;

    const TensorDescriptor* outputDesc = nullptr;
    Data_t output                      = nullptr;
    float divisor                      = 1.0f;
};

struct BwdInvokeParams : HingeEmbeddingLossInvokeParams
{
    BwdInvokeParams() = default;

    const TensorDescriptor* doutputDesc = nullptr;
    const TensorDescriptor* dinputDesc  = nullptr;

    ConstData_t doutput = nullptr;
    ConstData_t dinput  = nullptr;

    float divisor = 1.0f;
};

struct UnreducedFwdInvokeParams : HingeEmbeddingLossInvokeParams
{
    UnreducedFwdInvokeParams() = default;

    const TensorDescriptor* outputDesc = nullptr;
    Data_t output                      = nullptr;
};

struct UnreducedBwdInvokeParams : HingeEmbeddingLossInvokeParams
{
    UnreducedBwdInvokeParams() = default;

    const TensorDescriptor* doutputDesc = nullptr;
    const TensorDescriptor* dinputDesc  = nullptr;

    ConstData_t doutput = nullptr;
    ConstData_t dinput  = nullptr;
    Data_t output       = nullptr;
};

} // namespace loss

} // namespace miopen
