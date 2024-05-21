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
#include <miopen/loss/invoke_params.hpp>
#include <miopen/loss/problem_description.hpp>
#include <miopen/loss/solvers.hpp>
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

size_t GetHingeEmbeddingLossForwardWorkspaceSize(Handle& handle,
                                                 const TensorDescriptor& iDesc,
                                                 const TensorDescriptor& tDesc,
                                                 const TensorDescriptor& oDesc)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = loss::HingeEmbeddingLossFwdProblemDescription{iDesc, tDesc, oDesc};

    const auto algo    = AlgorithmName{"HingeEmbeddingLossFwd"};
    const auto solvers = solver::SolverContainer<solver::loss::HingeEmbeddingLossFwd>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t HingeEmbeddingLossForward(Handle& handle,
                                         Data_t workspace,
                                         size_t workspaceSizeInBytes,
                                         const TensorDescriptor& iDesc,
                                         ConstData_t i,
                                         const TensorDescriptor& tDesc,
                                         ConstData_t t,
                                         const TensorDescriptor& oDesc,
                                         Data_t o,
                                         float margin,
                                         float divisor)
{
    const auto problem = loss::HingeEmbeddingLossFwdProblemDescription{iDesc, tDesc, oDesc};

    const auto invoke_params = [&]() {
        auto tmp           = loss::FwdInvokeParams{};
        tmp.iDesc          = &iDesc;
        tmp.tDesc          = &tDesc;
        tmp.oDesc          = &oDesc;
        tmp.i              = i;
        tmp.t              = t;
        tmp.o              = o;
        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        tmp.margin         = margin;
        tmp.divisor        = divisor;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"HingeEmbeddingLossFwd"};
    const auto solvers = solver::SolverContainer<solver::loss::HingeEmbeddingLossFwd>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t HingeEmbeddingLossUnreducedForward(Handle& handle,
                                                  const TensorDescriptor& iDesc,
                                                  ConstData_t i,
                                                  const TensorDescriptor& tDesc,
                                                  ConstData_t t,
                                                  const TensorDescriptor& oDesc,
                                                  Data_t o,
                                                  float margin)
{
    const auto problem =
        loss::HingeEmbeddingLossUnreducedFwdProblemDescription{iDesc, tDesc, oDesc};

    const auto invoke_params = [&]() {
        auto tmp   = loss::UnreducedFwdInvokeParams{};
        tmp.iDesc  = &iDesc;
        tmp.tDesc  = &tDesc;
        tmp.oDesc  = &oDesc;
        tmp.i      = i;
        tmp.t      = t;
        tmp.o      = o;
        tmp.margin = margin;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"HingeEmbeddingLossUnreducedFwd"};
    const auto solvers = solver::SolverContainer<solver::loss::HingeEmbeddingLossUnreducedFwd>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t HingeEmbeddingLossUnreducedBackward(Handle& handle,
                                                   const TensorDescriptor& iDesc,
                                                   ConstData_t i,
                                                   const TensorDescriptor& tDesc,
                                                   ConstData_t t,
                                                   const TensorDescriptor& dODesc,
                                                   ConstData_t dO,
                                                   const TensorDescriptor& dIDesc,
                                                   Data_t dI,
                                                   float margin)
{
    const auto problem =
        loss::HingeEmbeddingLossUnreducedBwdProblemDescription{iDesc, tDesc, dODesc, dIDesc};

    const auto invoke_params = [&]() {
        auto tmp   = loss::UnreducedBwdInvokeParams{};
        tmp.iDesc  = &iDesc;
        tmp.tDesc  = &tDesc;
        tmp.dODesc = &dODesc;
        tmp.dIDesc = &dIDesc;
        tmp.i      = i;
        tmp.t      = t;
        tmp.dO     = dO;
        tmp.dI     = dI;
        tmp.margin = margin;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"HingeEmbeddingLossUnreducedBwd"};
    const auto solvers = solver::SolverContainer<solver::loss::HingeEmbeddingLossUnreducedBwd>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
