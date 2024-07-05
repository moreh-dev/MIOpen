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
#include <miopen/repeat/invoke_params.hpp>
#include <miopen/repeat/solvers.hpp>
#include <miopen/repeat.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t RepeatForward(Handle& handle,
                             const TensorDescriptor& xDesc,
                             ConstData_t x,
                             const int* sizes,
                             const int num_sizes,
                             const TensorDescriptor& yDesc,
                             Data_t y)
{
    if(x == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    const auto x_dims_size = xDesc.GetLengths().size();
    int32_t offset         = static_cast<int32_t>(num_sizes) - static_cast<int32_t>(x_dims_size);

    if(offset < 0)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "Number of dimensions of sizes can not be smaller than number of dimensions "
                     "of tensor.");
    }

    std::vector<int> sizes_vector(sizes, sizes + num_sizes);

    const auto problem       = repeat::ProblemDescription{xDesc, yDesc, offset, sizes_vector, true};
    const auto invoke_params = repeat::InvokeParams{xDesc, x, yDesc, y, offset};
    const auto algo          = AlgorithmName{"RepeatForward"};
    const auto solvers       = solver::SolverContainer<solver::repeat::RepeatForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t RepeatBackward(Handle& handle,
                              const TensorDescriptor& dyDesc,
                              ConstData_t dy,
                              const int* sizes,
                              const int num_sizes,
                              const TensorDescriptor& dxDesc,
                              Data_t dx)
{
    if(dx == nullptr || dy == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    const auto dx_dims_size = dxDesc.GetLengths().size();
    int32_t offset          = static_cast<int32_t>(num_sizes) - static_cast<int32_t>(dx_dims_size);

    if(offset < 0)
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "Number of dimensions of sizes can not be smaller than number of dimensions "
                     "of tensor.");
    }

    std::vector<int> sizes_vector(sizes, sizes + num_sizes);

    const auto problem = repeat::ProblemDescription{dyDesc, dxDesc, offset, sizes_vector, false};
    const auto invoke_params = repeat::InvokeParams{dyDesc, dy, dxDesc, dx, offset};
    const auto algo          = AlgorithmName{"RepeatBackward"};
    const auto solvers       = solver::SolverContainer<solver::repeat::RepeatBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
