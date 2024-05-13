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

#include "miopen/miopen.h"
#include "miopen/pad_reflection/problem_description.hpp"
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/pad_reflection/invoke_params.hpp>
#include <miopen/pad_reflection/solvers.hpp>
#include <miopen/pad_reflection.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t PadReflection1dFwdContiguous(Handle& handle,
                                            const TensorDescriptor& xDesc,
                                            ConstData_t x,
                                            const TensorDescriptor& yDesc,
                                            Data_t y,
                                            const size_t* padding,
                                            const size_t num_padding)
{
    const auto problem =
        pad_reflection::PadReflection1dFwdContiguousProblemDescription{xDesc, yDesc, num_padding};

    const auto invoke_params = [&]() {
        auto tmp        = pad_reflection::InvokeParams{};
        tmp.type        = InvokeType::Run;
        tmp.xDesc       = &xDesc;
        tmp.yDesc       = &yDesc;
        tmp.x           = x;
        tmp.y           = y;
        tmp.padding     = padding;
        tmp.num_padding = num_padding;
        return tmp;
    }();

    const auto algo = AlgorithmName{"PadReflection1dFwdContiguous"};
    const auto solvers =
        solver::SolverContainer<solver::pad_reflection::PadReflection1dFwdContiguous>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t PadReflection1dFwd(Handle& handle,
                                  const TensorDescriptor& xDesc,
                                  ConstData_t x,
                                  const TensorDescriptor& yDesc,
                                  Data_t y,
                                  const size_t* padding,
                                  const size_t num_padding)
{
    const auto problem =
        pad_reflection::PadReflection1dFwdProblemDescription{xDesc, yDesc, num_padding};

    const auto invoke_params = [&]() {
        auto tmp        = pad_reflection::InvokeParams{};
        tmp.type        = InvokeType::Run;
        tmp.xDesc       = &xDesc;
        tmp.yDesc       = &yDesc;
        tmp.x           = x;
        tmp.y           = y;
        tmp.padding     = padding;
        tmp.num_padding = num_padding;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"PadReflection1dFwd"};
    const auto solvers = solver::SolverContainer<solver::pad_reflection::PadReflection1dFwd>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t PadReflection1dBwdContiguous(Handle& handle,
                                            const TensorDescriptor& xDesc,
                                            ConstData_t x,
                                            const TensorDescriptor& yDesc,
                                            Data_t y,
                                            const size_t* padding,
                                            const size_t num_padding)
{
    const auto problem =
        pad_reflection::PadReflection1dBwdContiguousProblemDescription{xDesc, yDesc, num_padding};

    const auto invoke_params = [&]() {
        auto tmp        = pad_reflection::InvokeParams{};
        tmp.type        = InvokeType::Run;
        tmp.xDesc       = &xDesc;
        tmp.yDesc       = &yDesc;
        tmp.x           = x;
        tmp.y           = y;
        tmp.padding     = padding;
        tmp.num_padding = num_padding;
        return tmp;
    }();

    const auto algo = AlgorithmName{"PadReflection1dBwdContiguous"};
    const auto solvers =
        solver::SolverContainer<solver::pad_reflection::PadReflection1dBwdContiguous>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t PadReflection1dBwd(Handle& handle,
                                  const TensorDescriptor& xDesc,
                                  ConstData_t x,
                                  const TensorDescriptor& yDesc,
                                  Data_t y,
                                  const size_t* padding,
                                  const size_t num_padding)
{
    const auto problem =
        pad_reflection::PadReflection1dBwdProblemDescription{xDesc, yDesc, num_padding};

    const auto invoke_params = [&]() {
        auto tmp        = pad_reflection::InvokeParams{};
        tmp.type        = InvokeType::Run;
        tmp.xDesc       = &xDesc;
        tmp.yDesc       = &yDesc;
        tmp.x           = x;
        tmp.y           = y;
        tmp.padding     = padding;
        tmp.num_padding = num_padding;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"PadReflection1dBwd"};
    const auto solvers = solver::SolverContainer<solver::pad_reflection::PadReflection1dBwd>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
