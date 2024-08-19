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
#include <miopen/logsumexp/invoke_params.hpp>
#include <miopen/logsumexp/solvers.hpp>
#include <miopen/logsumexp.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t LogsumexpForward(Handle& handle,
                                const TensorDescriptor& inputDesc,
                                ConstData_t input,
                                const TensorDescriptor& outputDesc,
                                Data_t output,
                                const int* dims,
                                int num_dims)
{
    std::vector dims_vector(dims, dims + num_dims);

    const auto problem = logsumexp::ProblemDescription{inputDesc, outputDesc, dims_vector};

    const auto invoke_params = [&]() {
        auto tmp       = logsumexp::LogsumexpForwardInvokeParams{};
        tmp.type       = InvokeType::Run;
        tmp.inputDesc  = &inputDesc;
        tmp.outputDesc = &outputDesc;
        tmp.input      = input;
        tmp.output     = output;
        tmp.dims       = &dims_vector;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"LogsumexpForward"};
    const auto solvers = solver::SolverContainer<solver::logsumexp::LogsumexpForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t LogsumexpBackward(Handle& handle,
                                 const TensorDescriptor& inputDesc,
                                 ConstData_t input,
                                 const TensorDescriptor& inputGradDesc,
                                 Data_t inputGrad,
                                 const TensorDescriptor& outputDesc,
                                 ConstData_t output,
                                 const TensorDescriptor& outputGradDesc,
                                 ConstData_t outputGrad,
                                 const int* dims,
                                 int num_dims)
{
    std::vector dims_vector(dims, dims + num_dims);

    const auto problem = logsumexp::ProblemDescription{
        inputDesc, inputGradDesc, outputDesc, outputGradDesc, dims_vector};

    const auto invoke_params = [&]() {
        auto tmp           = logsumexp::LogsumexpBackwardInvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.inputDesc      = &inputDesc;
        tmp.inputGradDesc  = &inputGradDesc;
        tmp.outputDesc     = &outputDesc;
        tmp.outputGradDesc = &outputGradDesc;
        tmp.input          = input;
        tmp.inputGrad      = inputGrad;
        tmp.output         = output;
        tmp.outputGrad     = outputGrad;
        tmp.dims           = &dims_vector;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"LogsumexpBackward"};
    const auto solvers = solver::SolverContainer<solver::logsumexp::LogsumexpBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
