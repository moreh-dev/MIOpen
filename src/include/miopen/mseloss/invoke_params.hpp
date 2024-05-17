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

#include "miopen/common.hpp"
#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>

namespace miopen {
namespace mseloss {
namespace forward {
struct ReducedInvokeParams : public miopen::InvokeParams
{
    ReducedInvokeParams() = default;

    const TensorDescriptor* xDesc;
    const TensorDescriptor* yDesc;
    const TensorDescriptor* zDesc;

    ConstData_t x;
    ConstData_t y;
    Data_t z;

    float divisor = 1.0f;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

struct UnreducedInvokeParams : public miopen::InvokeParams
{
    UnreducedInvokeParams() = default;

    const TensorDescriptor* xDesc;
    const TensorDescriptor* yDesc;
    const TensorDescriptor* zDesc;

    ConstData_t x;
    ConstData_t y;
    Data_t z;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

} // namespace forward
namespace backward {
struct ReducedInvokeParams : public miopen::InvokeParams
{
    ReducedInvokeParams() = default;

    const TensorDescriptor* xDesc;
    const TensorDescriptor* yDesc;
    const TensorDescriptor* dzDesc;
    const TensorDescriptor* dxDesc;
    const TensorDescriptor* dyDesc;

    ConstData_t x;
    ConstData_t y;
    ConstData_t dz;
    Data_t dx;
    Data_t dy;

    float divisor = 1.0f;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

struct UnreducedInvokeParams : public miopen::InvokeParams
{

    UnreducedInvokeParams() = default;

    const TensorDescriptor* xDesc;
    const TensorDescriptor* yDesc;
    const TensorDescriptor* dzDesc;
    const TensorDescriptor* dxDesc;
    const TensorDescriptor* dyDesc;

    ConstData_t x;
    ConstData_t y;
    ConstData_t dz;
    Data_t dx;
    Data_t dy;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};
} // namespace backward
} // namespace mseloss
} // namespace miopen
