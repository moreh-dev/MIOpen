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
                                                 const TensorDescriptor& inputDesc,
                                                 const TensorDescriptor& targetDesc,
                                                 const TensorDescriptor& outputDesc,
                                                 miopenLossReductionMode_t reduction)
{
    if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    {
        return 0;
    }
    auto ctx = ExecutionContext{&handle};
    const auto problem =
        loss::HingeEmbeddingLossFwdProblemDescription{inputDesc, targetDesc, outputDesc};

    const auto algo    = AlgorithmName{"HingeEmbeddingLossFwd"};
    const auto solvers = solver::SolverContainer<solver::loss::HingeEmbeddingLossFwd>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t HingeEmbeddingLossForward(Handle& handle,
                                         Data_t workspace,
                                         size_t workspaceSizeInBytes,
                                         const TensorDescriptor& inputDesc,
                                         ConstData_t input,
                                         const TensorDescriptor& targetDesc,
                                         ConstData_t target,
                                         const TensorDescriptor& outputDesc,
                                         Data_t output,
                                         float margin,
                                         const miopenLossReductionMode_t reduction)
{
    const auto problem =
        loss::HingeEmbeddingLossFwdProblemDescription{inputDesc, targetDesc, outputDesc};

    const auto invoke_params = [&]() {
        auto tmp           = loss::FwdInvokeParams{};
        tmp.inputDesc      = &inputDesc;
        tmp.targetDesc     = &targetDesc;
        tmp.outputDesc     = &outputDesc;
        tmp.input          = input;
        tmp.target         = target;
        tmp.output         = output;
        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        tmp.margin         = margin;
        tmp.reduction      = reduction;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"HingeEmbeddingLossFwd"};
    const auto solvers = solver::SolverContainer<solver::loss::HingeEmbeddingLossFwd>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t HingeEmbeddingLossBackward(Handle& handle,
                                          const TensorDescriptor& inputDesc,
                                          ConstData_t input,
                                          const TensorDescriptor& targetDesc,
                                          ConstData_t target,
                                          const TensorDescriptor& doutputDesc,
                                          ConstData_t doutput,
                                          const TensorDescriptor& dinputDesc,
                                          Data_t dinput,
                                          float margin,
                                          float divisor)
{
    const auto problem = loss::HingeEmbeddingLossBwdProblemDescription{
        inputDesc, targetDesc, doutputDesc, dinputDesc};

    const auto invoke_params = [&]() {
        auto tmp        = loss::BwdInvokeParams{};
        tmp.inputDesc   = &inputDesc;
        tmp.targetDesc  = &targetDesc;
        tmp.doutputDesc = &doutputDesc;
        tmp.dinputDesc  = &dinputDesc;
        tmp.input       = input;
        tmp.target      = target;
        tmp.doutput     = doutput;
        tmp.dinput      = dinput;
        tmp.margin      = margin;
        tmp.divisor     = divisor;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"HingeEmbeddingLossBwd"};
    const auto solvers = solver::SolverContainer<solver::loss::HingeEmbeddingLossBwd>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t HingeEmbeddingLossUnreducedForward(Handle& handle,
                                                  const TensorDescriptor& inputDesc,
                                                  ConstData_t input,
                                                  const TensorDescriptor& targetDesc,
                                                  ConstData_t target,
                                                  const TensorDescriptor& outputDesc,
                                                  Data_t output,
                                                  float margin)
{
    const auto problem =
        loss::HingeEmbeddingLossUnreducedFwdProblemDescription{inputDesc, targetDesc, outputDesc};

    const auto invoke_params = [&]() {
        auto tmp       = loss::UnreducedFwdInvokeParams{};
        tmp.inputDesc  = &inputDesc;
        tmp.targetDesc = &targetDesc;
        tmp.outputDesc = &outputDesc;
        tmp.input      = input;
        tmp.target     = target;
        tmp.output     = output;
        tmp.margin     = margin;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"HingeEmbeddingLossUnreducedFwd"};
    const auto solvers = solver::SolverContainer<solver::loss::HingeEmbeddingLossUnreducedFwd>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t HingeEmbeddingLossUnreducedBackward(Handle& handle,
                                                   const TensorDescriptor& inputDesc,
                                                   ConstData_t input,
                                                   const TensorDescriptor& targetDesc,
                                                   ConstData_t target,
                                                   const TensorDescriptor& doutputDesc,
                                                   ConstData_t doutput,
                                                   const TensorDescriptor& dinputDesc,
                                                   Data_t dinput,
                                                   float margin)
{
    const auto problem = loss::HingeEmbeddingLossUnreducedBwdProblemDescription{
        inputDesc, targetDesc, doutputDesc, dinputDesc};

    const auto invoke_params = [&]() {
        auto tmp        = loss::UnreducedBwdInvokeParams{};
        tmp.inputDesc   = &inputDesc;
        tmp.targetDesc  = &targetDesc;
        tmp.doutputDesc = &doutputDesc;
        tmp.dinputDesc  = &dinputDesc;
        tmp.input       = input;
        tmp.target      = target;
        tmp.doutput     = doutput;
        tmp.dinput      = dinput;
        tmp.margin      = margin;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"HingeEmbeddingLossUnreducedBwd"};
    const auto solvers = solver::SolverContainer<solver::loss::HingeEmbeddingLossUnreducedBwd>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
