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

#include <miopen/bcelogitsloss/invoke_params.hpp>
#include <miopen/bcelogitsloss/solvers.hpp>
#include <miopen/bcelogitsloss.hpp>
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

size_t GetBCELogitsLossReducedForwardWorkspaceSize(Handle& handle,
                                                   const TensorDescriptor& iDesc,
                                                   const TensorDescriptor& tDesc,
                                                   const TensorDescriptor& wDesc,
                                                   const TensorDescriptor& pwDesc,
                                                   const TensorDescriptor& oDesc)
{
    auto ctx = ExecutionContext{&handle};
    const auto problem =
        bcelogitsloss::ReducedForwardProblemDescription{iDesc, tDesc, wDesc, pwDesc, oDesc};

    const auto algo    = AlgorithmName{"BCELogitsLossReducedForward"};
    const auto solvers = solver::SolverContainer<solver::bcelogitsloss::ReducedForward5d>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t BCELogitsLossReducedForward(Handle& handle,
                                           Data_t workspace,
                                           size_t workspaceSizeInBytes,
                                           const TensorDescriptor& iDesc,
                                           ConstData_t i,
                                           const TensorDescriptor& tDesc,
                                           ConstData_t t,
                                           const TensorDescriptor& wDesc,
                                           ConstData_t w,
                                           const TensorDescriptor& pwDesc,
                                           ConstData_t pw,
                                           const TensorDescriptor& oDesc,
                                           Data_t o,
                                           float divisor)
{
    const auto problem =
        bcelogitsloss::ReducedForwardProblemDescription{iDesc, tDesc, wDesc, pwDesc, oDesc};

    const auto invoke_params = [&]() {
        auto tmp           = bcelogitsloss::InvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.iDesc          = &iDesc;
        tmp.tDesc          = &tDesc;
        tmp.wDesc          = &wDesc;
        tmp.pwDesc         = &pwDesc;
        tmp.oDesc          = &oDesc;
        tmp.i              = i;
        tmp.t              = t;
        tmp.w              = w;
        tmp.pw             = pw;
        tmp.o              = o;
        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        tmp.divisor        = divisor;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"BCELogitsLossReducedForward"};
    const auto solvers = solver::SolverContainer<solver::bcelogitsloss::ReducedForward5d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t BCELogitsLossReducedBackward(Handle& handle,
                                            const TensorDescriptor& iDesc,
                                            ConstData_t i,
                                            const TensorDescriptor& tDesc,
                                            ConstData_t t,
                                            const TensorDescriptor& wDesc,
                                            ConstData_t w,
                                            const TensorDescriptor& pwDesc,
                                            ConstData_t pw,
                                            const TensorDescriptor& doDesc,
                                            ConstData_t dO,
                                            const TensorDescriptor& diDesc,
                                            Data_t dI,
                                            const TensorDescriptor& dtDesc,
                                            Data_t dT,
                                            float divisor)
{
    const auto problem = bcelogitsloss::ReducedBackwardProblemDescription{
        iDesc, tDesc, wDesc, pwDesc, doDesc, diDesc, dtDesc};

    const auto invoke_params = [&]() {
        auto tmp    = bcelogitsloss::InvokeParams{};
        tmp.type    = InvokeType::Run;
        tmp.iDesc   = &iDesc;
        tmp.tDesc   = &tDesc;
        tmp.wDesc   = &wDesc;
        tmp.pwDesc  = &pwDesc;
        tmp.doDesc  = &doDesc;
        tmp.diDesc  = &diDesc;
        tmp.dtDesc  = &dtDesc;
        tmp.i       = i;
        tmp.t       = t;
        tmp.w       = w;
        tmp.pw      = pw;
        tmp.i_grad  = dI;
        tmp.t_grad  = dT;
        tmp.o_grad  = dO;
        tmp.divisor = divisor;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"BCELogitsLossReducedBackward"};
    const auto solvers = solver::SolverContainer<solver::bcelogitsloss::ReducedBackward5d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
