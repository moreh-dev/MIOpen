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

    const TensorDescriptor* iDesc = nullptr;
    const TensorDescriptor* tDesc = nullptr;

    ConstData_t i              = nullptr;
    ConstData_t t              = nullptr;
    Data_t o                   = nullptr;
    Data_t workspace           = nullptr;
    std::size_t workspace_size = 0;
    float margin               = 1.0f;
    float divisor              = 1.0f;

    std::size_t GetWorkspaceSize() const { return workspace_size; }
    Data_t GetWorkspace() const { return workspace; }
};

struct FwdInvokeParams : HingeEmbeddingLossInvokeParams
{
    // FwdInvokeParams() = default;

    // const TensorDescriptor* iDesc = nullptr;
    // const TensorDescriptor* tDesc = nullptr;
    const TensorDescriptor* oDesc = nullptr;

    // ConstData_t i              = nullptr;
    // ConstData_t t              = nullptr;
    // Data_t o                   = nullptr;
    // Data_t workspace           = nullptr;
    // std::size_t workspace_size = 0;
    // float margin               = 1.0f;
    // float divisor              = 1.0f;

    // std::size_t GetWorkspaceSize() const { return workspace_size; }
    // Data_t GetWorkspace() const { return workspace; }
};

struct BwdInvokeParams : public miopen::InvokeParams
{
    BwdInvokeParams() = default;

    const TensorDescriptor* iDesc  = nullptr;
    const TensorDescriptor* tDesc  = nullptr;
    const TensorDescriptor* dODesc = nullptr;
    const TensorDescriptor* dIDesc = nullptr;

    ConstData_t i              = nullptr;
    ConstData_t t              = nullptr;
    ConstData_t dO             = nullptr;
    ConstData_t dI             = nullptr;
    Data_t o                   = nullptr;
    Data_t workspace           = nullptr;
    std::size_t workspace_size = 0;
    float margin               = 1.0f;
    float divisor              = 1.0f;

    std::size_t GetWorkspaceSize() const { return workspace_size; }
    Data_t GetWorkspace() const { return workspace; }
};

struct UnreducedFwdInvokeParams : public miopen::InvokeParams
{
    UnreducedFwdInvokeParams() = default;

    const TensorDescriptor* iDesc = nullptr;
    const TensorDescriptor* tDesc = nullptr;
    const TensorDescriptor* oDesc = nullptr;

    ConstData_t i              = nullptr;
    ConstData_t t              = nullptr;
    Data_t o                   = nullptr;
    Data_t workspace           = nullptr;
    std::size_t workspace_size = 0;
    float margin               = 1.0f;

    std::size_t GetWorkspaceSize() const { return workspace_size; }
    Data_t GetWorkspace() const { return workspace; }
};

struct UnreducedBwdInvokeParams : public miopen::InvokeParams
{
    UnreducedBwdInvokeParams() = default;

    const TensorDescriptor* iDesc  = nullptr;
    const TensorDescriptor* tDesc  = nullptr;
    const TensorDescriptor* dODesc = nullptr;
    const TensorDescriptor* dIDesc = nullptr;

    ConstData_t i              = nullptr;
    ConstData_t t              = nullptr;
    ConstData_t dO             = nullptr;
    ConstData_t dI             = nullptr;
    Data_t o                   = nullptr;
    Data_t workspace           = nullptr;
    std::size_t workspace_size = 0;
    float margin               = 1.0f;

    std::size_t GetWorkspaceSize() const { return workspace_size; }
    Data_t GetWorkspace() const { return workspace; }
};

} // namespace loss

} // namespace miopen
