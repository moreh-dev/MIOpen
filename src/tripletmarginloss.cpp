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
#include <miopen/tensor.hpp>
#include <miopen/tripletmarginloss.hpp>
#include <miopen/tripletmarginloss/invoke_params.hpp>
#include <miopen/tripletmarginloss/solvers.hpp>

namespace miopen {

size_t GetTripletMarginLossUnreducedForwardWorkspaceSize(Handle& handle,
                                                         const TensorDescriptor& aDesc,
                                                         const TensorDescriptor& oDesc)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = tripletmarginloss::ProblemDescription{aDesc, aDesc, aDesc, oDesc};

    const auto solvers = solver::SolverContainer<solver::tripletmarginloss::UnreducedForward2d>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t TripletMarginLossUnreducedForward(Handle& handle,
                                                 Data_t workspace,
                                                 size_t workspaceSizeInBytes,
                                                 const TensorDescriptor& aDesc,
                                                 ConstData_t anchor,
                                                 const TensorDescriptor& pDesc,
                                                 ConstData_t positive,
                                                 const TensorDescriptor& nDesc,
                                                 ConstData_t negative,
                                                 const TensorDescriptor& oDesc,
                                                 Data_t o,
                                                 float margin,
                                                 int p,
                                                 float eps,
                                                 bool swap)
{
    const auto problem = tripletmarginloss::ProblemDescription{aDesc, pDesc, nDesc, oDesc};

    const auto invoke_params = [&]() {
        auto tmp           = tripletmarginloss::InvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.aDesc          = &aDesc;
        tmp.pDesc          = &pDesc;
        tmp.nDesc          = &nDesc;
        tmp.oDesc          = &oDesc;
        tmp.anchor         = anchor;
        tmp.positive       = positive;
        tmp.negative       = negative;
        tmp.o              = o;
        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        tmp.margin         = margin;
        tmp.p              = p;
        tmp.eps            = eps;
        tmp.swap           = swap;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"SmoothL1LossUnreducedForward"};
    const auto solvers = solver::SolverContainer<solver::tripletmarginloss::UnreducedForward2d>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
