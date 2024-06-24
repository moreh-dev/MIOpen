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
#include <miopen/tensor.hpp>
#include <miopen/dropout.hpp>
#include <miopen/rrelu.hpp>
#include <miopen/rrelu/invoke_params.hpp>
#include <miopen/rrelu/solvers.hpp>

namespace miopen {

size_t GetRReLUStatesSize(Handle& handle)
{
    return std::min(size_t(MAX_PRNG_STATE), handle.GetImage3dMaxWidth()) * sizeof(prngStates);
}

miopenStatus_t
RReLUStatesInit(Handle& handle, Data_t states, size_t stateSizeInBytes, uint64_t seed)
{
    DropoutDescriptor{}.InitPRNGState(handle, states, stateSizeInBytes, seed);

    return miopenStatusSuccess;
}

miopenStatus_t RReLUForward(Handle& handle,
                            ConstData_t states,
                            size_t stateSizeInBytes,
                            const TensorDescriptor& inputDesc,
                            ConstData_t input,
                            const TensorDescriptor& outputDesc,
                            Data_t output,
                            const float lower,
                            const float upper)
{
    const auto problem = rrelu::ForwardProblemDescription{inputDesc, outputDesc};

    const auto invoke_params = [&]() {
        auto tmp       = rrelu::InvokeParams{};
        tmp.type       = InvokeType::Run;
        tmp.inputDesc  = &inputDesc;
        tmp.input      = input;
        tmp.outputDesc = &outputDesc;
        tmp.output     = output;
        tmp.lower      = lower;
        tmp.upper      = upper;
        tmp.states     = states;
        tmp.state_size = stateSizeInBytes;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"RReLUForward"};
    const auto solvers = solver::SolverContainer<solver::rrelu::ContiguouseForward,
                                                 solver::rrelu::nonContiguouseForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
