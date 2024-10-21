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

#include <miopen/find_solution.hpp>
#include <miopen/gather.hpp>
#include <miopen/gather/invoke_params.hpp>
#include <miopen/gather/problem_description.hpp>
#include <miopen/gather/solvers.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/names.hpp>

namespace miopen {

static auto GatherBackwardSolvers()
{
    return solver::SolverContainer<solver::gather::GatherV2Backward>{};
}

GatherDescriptor::GatherDescriptor() {}

GatherDescriptor::GatherDescriptor(miopenGatherMode_t m, uint32_t dim_, uint32_t batch_dims_)
    : mode(m), dim(dim_), batch_dims(batch_dims_)
{
}

miopenStatus_t GatherDescriptor::Backward(Handle& handle,
                                          const TensorDescriptor& outputGradDesc,
                                          ConstData_t outputgrad,
                                          const TensorDescriptor& indiceDesc,
                                          ConstData_t indices,
                                          const TensorDescriptor& paramGradDesc,
                                          Data_t paramGrad) const
{
    const auto problem =
        gather::BwdProblemDescription{*this, outputGradDesc, indiceDesc, paramGradDesc};

    const auto invoke_params = [&]() {
        auto tmp           = gather::BwdInvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.outputGradDesc = &outputGradDesc;
        tmp.indicesDesc    = &indiceDesc;
        tmp.paramGradDesc  = &paramGradDesc;
        tmp.outputGrad     = outputgrad;
        tmp.indices        = indices;
        tmp.paramGrad      = paramGrad;
        tmp.dim            = getDim();
        tmp.batch_dims     = getBatchDims();
        return tmp;
    }();

    const auto algo = AlgorithmName{"miopenGatherBackward"};
    GatherBackwardSolvers().ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

std::ostream& operator<<(std::ostream& stream, const GatherDescriptor& x)
{
    MIOPEN_LOG_ENUM(stream, x.mode, MIOPEN_GATHER, MIOPEN_GATHER_V2, MIOPEN_GATHER_ND) << ", ";
    return stream;
}

} // namespace miopen
