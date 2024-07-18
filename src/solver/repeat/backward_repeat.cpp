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
#include "../../kernels/tensor_utils.hpp"

#define LOCAL_SIZE 1024

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

    auto output_size =
        std::accumulate(dydims.begin(), dydims.end(), 1ULL, std::multiplies<size_t>{});

    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(output_size, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenRepeat.cpp";
        kernel.kernel_name = "RepeatBackward";

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

            auto dydims    = params.xDyDesc->GetLengths();
            auto dxdims    = params.yDxDesc->GetLengths();
            auto dystrides = params.xDyDesc->GetStrides();
            auto dxstrides = params.yDxDesc->GetStrides();
            auto offset    = params.offset;

            auto inout_size =
                std::accumulate(dydims.begin(), dydims.end(), 1ULL, std::multiplies<size_t>{});

            auto dx_size =
                std::accumulate(dxdims.begin(), dxdims.end(), 1ULL, std::multiplies<size_t>{});

            tensor_view output_grad_tv;
            tensor_view input_grad_tv;

            for(int i = 0; i < dydims.size(); ++i)
            {
                output_grad_tv.dimensions[i] = dydims[i];
                output_grad_tv.strides[i]    = dystrides[i];
            }

            for(int i = 0; i < dxdims.size(); ++i)
            {
                input_grad_tv.dimensions[i] = dxdims[i];
                input_grad_tv.strides[i]    = dxstrides[i];
            }

            hipMemset(params.yDx, 0, dx_size * GetTypeSize(params.yDxDesc->GetType()));

            kernel(params.xDy, params.yDx, inout_size, offset, output_grad_tv, input_grad_tv);
        };
    };

    return result;
}

} // namespace repeat

} // namespace solver

} // namespace miopen
