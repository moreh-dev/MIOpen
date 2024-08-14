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
#include <miopen/repeat/invoke_params.hpp>
#include <miopen/repeat/solvers.hpp>
#include <miopen/repeat.hpp>
#include <miopen/target_properties.hpp>
#include <hip/hip_runtime.h>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE 1024
#define LOCAL_SIZE_128 128
#define LIMIT_SMALL_K 16

namespace miopen {

namespace solver {

namespace repeat {

bool RepeatBackward::IsApplicable(const ExecutionContext& context,
                                  const miopen::repeat::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
    {
        return false;
    }
    return true;
}

ConvSolution RepeatBackward::GetSolution(const ExecutionContext& context,
                                         const miopen::repeat::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype  = problem.GetDyDesc().GetType();
    auto dydims = problem.GetDyDesc().GetLengths();
    auto dxdims = problem.GetDxDesc().GetLengths();

    auto N = std::accumulate(dxdims.begin(), dxdims.end(), 1ULL, std::multiplies<size_t>{});
    auto K = std::accumulate(dydims.begin(), dydims.end(), 1ULL, std::multiplies<size_t>{}) / N;

    {
        size_t xlocalsize;
        size_t xgridsize;
        size_t ylocalsize;
        size_t ygridsize;
        size_t zlocalsize;
        size_t zgridsize;

        if(K > LIMIT_SMALL_K)
        {
            xlocalsize = LOCAL_SIZE_128;
            xgridsize  = N * LOCAL_SIZE_128;
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
            kernel.kernel_file = "MIOpenRepeat.cpp";
            kernel.kernel_name = "RepeatLargeKBackward";
        }
        else
        {
            kernel.kernel_file = "MIOpenRepeat.cpp";
            kernel.kernel_name = "RepeatSmallKBackward";
        }

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
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
            decltype(auto) params = raw_params.CastTo<miopen::repeat::InvokeParams>();

            auto dydims = params.xDyDesc->GetLengths();
            auto dxdims = params.yDxDesc->GetLengths();

            auto N = std::accumulate(dxdims.begin(), dxdims.end(), 1ULL, std::multiplies<size_t>{});
            auto K =
                std::accumulate(dydims.begin(), dydims.end(), 1ULL, std::multiplies<size_t>{}) / N;

            auto dy_tv = get_inner_expanded_tv<5>(*(params.xDyDesc));
            auto dx_tv = get_inner_expanded_tv<5>(*(params.yDxDesc));

            hipMemset(params.yDx, 0, N * GetTypeSize(params.yDxDesc->GetType()));

            kernel(params.xDy,
                   params.yDx,
                   static_cast<uint64_t>(N),
                   static_cast<uint64_t>(K),
                   dy_tv,
                   dx_tv);
        };
    };

    return result;
}

} // namespace repeat

} // namespace solver

} // namespace miopen
