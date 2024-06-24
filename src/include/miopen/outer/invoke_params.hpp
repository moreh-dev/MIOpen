/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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


namespace miopen {

namespace outer {

struct InvokeParams : public miopen::InvokeParams
{
    InvokeParams(
                    const TensorDescriptor& x1Desc_,
                    ConstData_t x1_,
                    const TensorDescriptor& x2Desc_,
                    ConstData_t x2_,
                    const TensorDescriptor& yDesc_,
                    Data_t y_
                )
        : x1Desc(x1Desc_), x1(x1_), x2Desc(x2Desc_), x2(x2_), yDesc(yDesc_), y(y_)
    {
    }

    TensorDescriptor x1Desc{};
    ConstData_t x1                       = nullptr;
    TensorDescriptor x2Desc{};
    ConstData_t x2                       = nullptr;
    TensorDescriptor yDesc{};
    Data_t y    = nullptr;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

} // namespace outer
} // namespace miopen