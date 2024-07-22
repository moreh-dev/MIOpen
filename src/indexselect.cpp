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

#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/indexselect/invoke_params.hpp>
#include <miopen/indexselect/solvers.hpp>
#include <miopen/indexselect.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t IndexSelectForward(Handle& handle,
                                  const TensorDescriptor& xDesc,
                                  ConstData_t x,
                                  const TensorDescriptor& indicesDesc,
                                  ConstData_t indices,
                                  const TensorDescriptor& yDesc,
                                  Data_t y,
                                  size_t dim)
{
    const auto problem = indexselect::ProblemDescription(xDesc, indicesDesc, yDesc, dim, true);

    const auto invoke_params =
        indexselect::InvokeParamsForward{xDesc, x, indicesDesc, indices, yDesc, y, dim};

    const auto algo = AlgorithmName{"IndexSelectForward"};

    const auto solvers = solver::SolverContainer<solver::indexselect::IndexSelectForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t IndexSelectBackward(Handle& handle,
                                   const TensorDescriptor& xGradDesc,
                                   Data_t xGrad,
                                   const TensorDescriptor& indicesDesc,
                                   ConstData_t indices,
                                   const TensorDescriptor& yGradDesc,
                                   ConstData_t yGrad,
                                   size_t dim)
{
    const auto problem =
        indexselect::ProblemDescription(xGradDesc, indicesDesc, yGradDesc, dim, false);

    const auto invoke_params = indexselect::InvokeParamsBackward{
        xGradDesc, xGrad, indicesDesc, indices, yGradDesc, yGrad, dim};

    const auto algo    = AlgorithmName{"IndexSelectBackward"};
    const auto solvers = solver::SolverContainer<solver::indexselect::IndexSelectBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
