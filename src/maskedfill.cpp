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

#include <miopen/maskedfill.hpp>

#include <miopen/maskedfill/problem_description.hpp>
#include <miopen/maskedfill/invoke_params.hpp>
#include <miopen/maskedfill/solvers.hpp>

#include <miopen/find_solution.hpp>

namespace miopen {

miopenStatus_t MaskedFillForward(Handle& handle,

                                 TensorDescriptor const& inputDesc,
                                 ConstData_t const input,
                                 TensorDescriptor const& outputDesc,
                                 Data_t output,

                                 TensorDescriptor const& maskDesc,
                                 ConstData_t const mask,

                                 float const value)
{
    auto const problem =
        maskedfill ::ProblemDescription(inputDesc, outputDesc, maskDesc, MIOPEN_MASKEDFILL_FORWARD);
    auto const algo          = AlgorithmName{"MaskedFillForward"};
    auto const invoke_params = [&] {
        auto tmp = maskedfill ::InvokeParams{};
        tmp.type = InvokeType ::Run;

        tmp.inputDesc  = &inputDesc;
        tmp.input      = input;
        tmp.outputDesc = &outputDesc;
        tmp.output     = output;

        tmp.maskDesc = &maskDesc;
        tmp.mask     = mask;

        tmp.value = value;

        return tmp;
    }();
    auto const solvers = solver ::SolverContainer<solver ::maskedfill ::MaskedFill>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

miopenStatus_t MaskedFillBackward(Handle& handle,

                                  TensorDescriptor const& outputGradientDesc,
                                  ConstData_t const outputGradient,
                                  TensorDescriptor const& inputGradientDesc,
                                  Data_t inputGradient,

                                  TensorDescriptor const& maskDesc,
                                  ConstData_t const mask,

                                  float const value)
{
    auto const problem = maskedfill ::ProblemDescription(
        outputGradientDesc, inputGradientDesc, maskDesc, MIOPEN_MASKEDFILL_BACKWARD);
    auto const algo          = AlgorithmName{"MaskedFillBackward"};
    auto const invoke_params = [&] {
        auto tmp = maskedfill ::InvokeParams{};
        tmp.type = InvokeType ::Run;

        tmp.inputDesc  = &outputGradientDesc;
        tmp.input      = outputGradient;
        tmp.outputDesc = &inputGradientDesc;
        tmp.output     = inputGradient;

        tmp.maskDesc = &maskDesc;
        tmp.mask     = mask;

        tmp.value = value;

        return tmp;
    }();
    auto const solvers = solver ::SolverContainer<solver ::maskedfill ::MaskedFill>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

} // namespace miopen
