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
#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>

#include <limits>

namespace miopen {

namespace instancenorm {

struct InvokeParams : public miopen::InvokeParams
{
    InvokeParams() = default;

    const TensorDescriptor* inputDesc   = nullptr;
    const TensorDescriptor* outputDesc  = nullptr;
    const TensorDescriptor* weightDesc  = nullptr;
    const TensorDescriptor* biasDesc    = nullptr;
    const TensorDescriptor* meanInDesc  = nullptr;
    const TensorDescriptor* varInDesc   = nullptr;
    const TensorDescriptor* meanOutDesc = nullptr;
    const TensorDescriptor* varOutDesc  = nullptr;
    const TensorDescriptor* meanVarDesc = nullptr;

    ConstData_t input  = nullptr;
    Data_t output      = nullptr;
    ConstData_t weight = nullptr;
    ConstData_t bias   = nullptr;
    ConstData_t meanIn = nullptr;
    ConstData_t varIn  = nullptr;
    Data_t meanOut     = nullptr;
    Data_t varOut      = nullptr;
    Data_t meanVar     = nullptr;

    float epsilon      = 1e-05f;
    float momentum     = 0.1f;
    bool useInputStats = false;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

} // namespace instancenorm

} // namespace miopen
