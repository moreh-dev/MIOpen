/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <miopen/onehot/solvers.hpp>
#include <miopen/onehot/invoke_params.hpp>
#include <miopen/onehot.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t OneHot(Handle& handle,
                      TensorDescriptor& inDesc,
                      const void* input,
                      long inputSize,
                      TensorDescriptor& outDesc,
                      void* output,
                      int numClasses)
{
    // If num classes is to -1(default), the number of classes will be inferred as one greater than
    // the largest class value in the input tensor.
    if(numClasses == -1)
    {
        for(auto i = 0; i < inputSize; ++i)
        {
            numClasses = std::max(numClasses, static_cast<const int*>(input)[i] + 1);
        }
    }

    const int* input_int = static_cast<const int*>(input);
    for(auto i = 0; i < inputSize; ++i)
    {
        if(input_int[i] < 0)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "OneHot: tensor values must be non-negative integers.");
        }
        if(input_int[i] >= numClasses)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "OneHot: tensor values must be less than numClasses.");
        }
    }

    const auto problem = onehot::ProblemDescription{inDesc, outDesc, inputSize, numClasses};

    const auto invoke_params = [&]() {
        auto tmp        = onehot::InvokeParams{};
        tmp.inDesc      = &inDesc;
        tmp.outDesc     = &outDesc;
        tmp.input       = input;
        tmp.output      = output;
        tmp.input_size  = inputSize;
        tmp.num_classes = numClasses;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"OneHot"};
    const auto solvers = solver::SolverContainer<solver::onehot::OneHot>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
