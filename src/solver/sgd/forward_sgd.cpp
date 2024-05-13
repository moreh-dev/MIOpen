/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include "miopen/conv_solution.hpp"
#include "miopen/miopen.h"
#include <miopen/sgd/solvers.hpp>

#include <miopen/sgd/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/sgd.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {
namespace solver {
namespace SGD {

bool SGDForward::IsApplicable([[maybe_unused]] const ExecutionContext& constext,
                              const miopen::SGD::ProblemDescription& problem) const
{
    if(!(problem.IsRightLength() && problem.IsSameType() && problem.IsSameStrides()))
    {
        return false;
    }

    return true;
}

ConvSolution SGDForward::GetSolution([[maybe_unused]] const ExecutionContext& context,
                                     const miopen::SGD::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype = problem.GetParamInDesc().GetType();
    auto dims  = problem.GetParamInDesc().GetLengths();

    bool is_contiguous = problem.IsContiguous(problem.GetParamInDesc());

    size_t param_size = std::accumulate(dims.begin(), dims.end(), 1ULL, std::multiplies<size_t>());
    size_t n_dims     = dims.size();

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
    };

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = AlignUp(param_size, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel         = KernelInfo{};
    kernel.kernel_file  = "MIOpenSGD.cpp";
    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    if(is_contiguous)
    {
        kernel.kernel_name     = "SGDFwdContiguous";
        result.invoker_factory = [param_size](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::SGD::InvokeParams>();

                kernel(params.paramIn,
                       params.paramOut,
                       params.grad,
                       params.momentumBufferIn,
                       params.momentumBufferOut,
                       params.lr,
                       params.momentum,
                       params.dampening,
                       params.weightDecay,
                       params.nesterov,
                       params.momentum_initialized,
                       param_size);
            };
        };
    }
    else
    {
        kernel.kernel_name     = "SGDFwd";
        result.invoker_factory = [param_size, n_dims](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::SGD::InvokeParams>();

                kernel(params.paramIn,
                       params.paramOut,
                       params.grad,
                       params.momentumBufferIn,
                       params.momentumBufferOut,
                       params.lr,
                       params.momentum,
                       params.dampening,
                       params.weightDecay,
                       params.nesterov,
                       params.momentum_initialized,
                       param_size,
                       n_dims,
                       params.dims,
                       params.strides);
            };
        };
    }
    result.construction_params.push_back(kernel);
    return result;
}

} // namespace SGD
} // namespace solver
} // namespace miopen
