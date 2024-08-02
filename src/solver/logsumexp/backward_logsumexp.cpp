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
#include <miopen/kernel_build_params.hpp>
#include <miopen/logsumexp/invoke_params.hpp>
#include <miopen/logsumexp/solvers.hpp>
#include <miopen/logsumexp.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>
#include "../../kernels/dims_utils.hpp"

#define LOCAL_SIZE 1024

namespace miopen {

namespace solver {

namespace logsumexp {

bool LogsumexpBackward::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                                     const miopen::logsumexp::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsAllPacked())
        return false;
    return true;
}

ConvSolution
LogsumexpBackward::GetSolution([[maybe_unused]] const ExecutionContext& context,
                               const miopen::logsumexp::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype           = problem.GetInputDesc().GetType();
    auto input_grad_dims = problem.GetInputDesc().GetLengths();
    auto N               = std::accumulate(
        input_grad_dims.begin(), input_grad_dims.end(), 1ULL, std::multiplies<size_t>());

    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(N, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenLogsumexp.cpp";
        kernel.kernel_name = "LogsumexpBackward";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_BF16", static_cast<int>(dtype == miopenBFloat16)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        };

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params =
                raw_params.CastTo<miopen::logsumexp::LogsumexpBackwardInvokeParams>();

            auto input_dims       = params.inputDesc->GetLengths();
            auto input_grad_dims  = params.inputGradDesc->GetLengths();
            auto output_dims      = params.outputDesc->GetLengths();
            auto output_grad_dims = params.outputGradDesc->GetLengths();

            int64_t N = std::accumulate(
                input_dims.begin(), input_dims.end(), 1, std::multiplies<int64_t>());
            auto dims_vector = *(params.dims);

            dims_5d_t selection_info;
            for(auto dim : dims_vector)
            {
                selection_info.x[dim] = 1;
            }

            auto input_tv       = get_inner_expanded_tv<5>(*(params.inputDesc));
            auto input_grad_tv  = get_inner_expanded_tv<5>(*(params.inputGradDesc));
            auto output_tv      = get_inner_expanded_tv<5>(*(params.outputDesc));
            auto output_grad_tv = get_inner_expanded_tv<5>(*(params.outputGradDesc));

            kernel(params.input,
                   params.inputGrad,
                   params.output,
                   params.outputGrad,
                   selection_info,
                   N,
                   input_tv,
                   input_grad_tv,
                   output_tv,
                   output_grad_tv);
        };
    };

    return result;
}

} // namespace logsumexp

} // namespace solver

} // namespace miopen
