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

namespace miopen {

namespace indexselect {

struct InvokeParamsForward : public miopen::InvokeParams
{
    InvokeParamsForward(const TensorDescriptor& xDesc_,
                        ConstData_t x_,
                        const TensorDescriptor& indicesDesc_,
                        ConstData_t indices_,
                        const TensorDescriptor& yDesc_,
                        Data_t y_,
                        size_t dim_)
        : xDesc(xDesc_),
          x(x_),
          indicesDesc(indicesDesc_),
          indices(indices_),
          yDesc(yDesc_),
          y(y_),
          dim(dim_)
    {
    }

    TensorDescriptor xDesc{};
    ConstData_t x = nullptr;
    TensorDescriptor indicesDesc{};
    ConstData_t indices = nullptr;
    TensorDescriptor yDesc{};
    Data_t y   = nullptr;
    size_t dim = 0;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

struct InvokeParamsBackward : public miopen::InvokeParams
{
    InvokeParamsBackward(const TensorDescriptor& xGradDesc_,
                         Data_t xGrad_,
                         const TensorDescriptor& indicesDesc_,
                         Data_t indices_,
                         const TensorDescriptor& yGradDesc_,
                         Data_t yGrad_,
                         size_t dim_)
        : xGradDesc(xGradDesc_),
          xGrad(xGrad_),
          indicesDesc(indicesDesc_),
          indices(indices_),
          yGradDesc(yGradDesc_),
          yGrad(yGrad_),
          dim(dim_)
    {
    }

    TensorDescriptor xGradDesc{};
    Data_t xGrad = nullptr;
    TensorDescriptor indicesDesc{};
    Data_t indices = nullptr;
    TensorDescriptor yGradDesc{};
    Data_t yGrad = nullptr;
    size_t dim   = 0;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

} // namespace indexselect
} // namespace miopen
