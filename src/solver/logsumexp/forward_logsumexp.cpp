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
#define LOCAL_SIZE_64 64
#define LIMIT_SMALL_K 16

namespace miopen {

namespace solver {

namespace logsumexp {

std::size_t sizeof_kernel_FLOAT_ACCUM(const miopen::logsumexp::ProblemDescription& problem)
{
    const auto datatype = problem.GetInputDesc().GetType();
    return get_data_size(datatype);
}

std::size_t sizeof_local_memory(const miopen::logsumexp::ProblemDescription& problem)
{
    return LOCAL_SIZE_64 * sizeof_kernel_FLOAT_ACCUM(problem);
}

bool IsImprovementOverROCmForward(const miopen::logsumexp::ProblemDescription& problem)
{
    constexpr size_t max_input_numel = 1000000;
    constexpr size_t max_K           = 1024;

    auto input_grad_dims = problem.GetInputDesc().GetLengths();
    auto K               = 1;

    if(problem.GetInputDesc().GetElementSize() > max_input_numel)
        return false;

    for(auto dim : problem.GetDims())
        K *= input_grad_dims[dim];

    if(K > max_K)
        return false;

    return true;
}

bool LogsumexpForward::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                                    const miopen::logsumexp::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsAllPacked())
        return false;
    if(!IsImprovementOverROCmForward(problem))
        return false;
    if(!(sizeof_local_memory(problem) <= TargetProperties::GetMaxLocalMemorySize()))
        return false;
    return true;
}

ConvSolution
LogsumexpForward::GetSolution([[maybe_unused]] const ExecutionContext& context,
                              const miopen::logsumexp::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype            = problem.GetInputDesc().GetType();
    auto input_grad_dims  = problem.GetInputDesc().GetLengths();
    auto input_grad_numel = std::accumulate(
        input_grad_dims.begin(), input_grad_dims.end(), 1ULL, std::multiplies<size_t>());
    auto dims_vector = problem.GetDims();

    int64_t K = 1;
    for(auto dim : dims_vector)
    {
        K *= input_grad_dims[dim];
    }
    int64_t N = static_cast<int64_t>(input_grad_numel) / K;

    {
        size_t xlocalsize;
        size_t xgridsize;
        size_t ylocalsize;
        size_t ygridsize;
        size_t zlocalsize;
        size_t zgridsize;

        if(K > LIMIT_SMALL_K)
        {
            xlocalsize = LOCAL_SIZE_64;
            xgridsize  = N * LOCAL_SIZE_64;
            ylocalsize = 1;
            ygridsize  = 1;
            zlocalsize = 1;
            zgridsize  = 1;
        }
        else
        {
            xlocalsize = LOCAL_SIZE;
            xgridsize  = AlignUp(N, xlocalsize);
            ylocalsize = 1;
            ygridsize  = 1;
            zlocalsize = 1;
            zgridsize  = 1;
        }

        auto kernel = KernelInfo{};

        if(K > LIMIT_SMALL_K)
        {
            kernel.kernel_file = "MIOpenLogsumexp.cpp";
            kernel.kernel_name = "LogsumexpLargeKForward";
        }
        else
        {
            kernel.kernel_file = "MIOpenLogsumexp.cpp";
            kernel.kernel_name = "LogsumexpSmallKForward";
        }

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
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
                raw_params.CastTo<miopen::logsumexp::LogsumexpForwardInvokeParams>();

            auto input_dims  = params.inputDesc->GetLengths();
            auto output_dims = params.outputDesc->GetLengths();
            auto dims_vector = *(params.dims);

            auto input_grad_numel = std::accumulate(
                input_dims.begin(), input_dims.end(), 1ULL, std::multiplies<size_t>());
            int64_t K = 1;
            for(auto dim : dims_vector)
            {
                K *= input_dims[dim];
            }
            int64_t N = static_cast<int64_t>(input_grad_numel) / K;

            auto input_tv  = get_inner_expanded_tv<5>(*(params.inputDesc));
            auto output_tv = get_inner_expanded_tv<5>(*(params.outputDesc));

            kernel(params.input, params.output, N, K, input_tv, output_tv);
        };
    };

    return result;
}

} // namespace logsumexp

} // namespace solver

} // namespace miopen
