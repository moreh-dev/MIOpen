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

#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace repeat {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& xdyDesc_,
                       const TensorDescriptor& ydxDesc_,
                       int32_t offset_,
                       const std::vector<int>& sizes_vector_,
                       bool isForward_,
                       ConstData_t xdy_,
                       Data_t ydx_)
        : offset(offset_), sizes_vector(sizes_vector_), isForward(isForward_)
    {
        if(offset < 0)
        {
            MIOPEN_THROW(miopenStatusBadParm, "repeat::ProblemDescription: offset is negative");
        }

        if(isForward)
        {
            xDesc = xdyDesc_;
            yDesc = ydxDesc_;
            x     = xdy_;
            y     = ydx_;

            if(xDesc.GetType() != yDesc.GetType())
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "repeat::ProblemDescription: Tensor types do not match.");
            }

            if(x == nullptr || y == nullptr)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "repeat::ProblemDescription: data pointer is null pointer.");
            }
        }
        else
        {
            dyDesc = xdyDesc_;
            dxDesc = ydxDesc_;
            dy     = xdy_;
            dx     = ydx_;

            if(dyDesc.GetType() != dxDesc.GetType())
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "repeat::ProblemDescription: Tensor types do not match.");
            }

            if(dy == nullptr || dx == nullptr)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "repeat::ProblemDescription: data pointer is null pointer.");
            }
        }
    }

    // For forward pass
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }

    // For backward pass
    const TensorDescriptor& GetDyDesc() const { return dyDesc; }
    const TensorDescriptor& GetDxDesc() const { return dxDesc; }

    int32_t GetOffset() const { return offset; }
    const std::vector<int>& GetSizesVector() const { return sizes_vector; }

    bool IsSameType() const
    {
        if(isForward)
        {
            if(xDesc.GetType() != yDesc.GetType())
            {
                return false;
            }
            return true;
        }
        else
        {
            if(yDesc.GetType() != xDesc.GetType())
            {
                return false;
            }
            return true;
        }
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor xDesc;
    TensorDescriptor yDesc;
    TensorDescriptor dyDesc;
    TensorDescriptor dxDesc;

    int32_t offset;
    std::vector<int> sizes_vector;

    const bool isForward;

    ConstData_t x;
    Data_t y;
    ConstData_t dy;
    Data_t dx;
};

} // namespace repeat

} // namespace miopen
