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

#include "miopen/instancenorm/problem_description.hpp"
#include "miopen/miopen.h"
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/instancenorm/invoke_params.hpp>
#include <miopen/instancenorm/solvers.hpp>
#include <miopen/instance_norm.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace instancenorm {

bool InstanceNormBwd::IsApplicable(
    [[maybe_unused]] const ExecutionContext& /*context*/,
    const miopen::instancenorm::InstanceNormBwdProblemDescription& problem) const
{
    return true;
}

ConvSolution InstanceNormBwd::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::instancenorm::InstanceNormBwdProblemDescription& problem) const
{
    std::ignore = context;
    auto result = ConvSolution{miopenStatusSuccess};

    auto in_dtype    = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto dtype       = problem.GetDoutputDesc().GetType();
    auto input_dims  = problem.GetInputDesc().GetLengths();
    auto target_size = input_dims[1];
    {
        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenInstanceNorm.cpp";
        kernel.kernel_name = "InstanceNormBwd";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"IN_OUT_TYPE", in_dtype == "bfloat16" ? "ushort" : in_dtype},
            {"LOCAL_SIZE", LOCAL_SIZE},
        };
        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = LOCAL_SIZE * target_size;
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;
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
            decltype(auto) params = raw_params.CastTo<miopen::instancenorm::InvokeParams>();

            auto x_tv          = get_inner_expanded_tv<5>(deref(params.inputDesc));
            auto dy_tv         = get_inner_expanded_tv<5>(deref(params.doutputDesc));
            auto scale_tv      = get_inner_expanded_tv<1>(deref(params.weightDesc));
            auto mean_var_tv   = get_inner_expanded_tv<2>(deref(params.meanVarDesc));
            auto dx_tv         = get_inner_expanded_tv<5>(deref(params.dinputDesc));
            auto scale_grad_tv = get_inner_expanded_tv<1>(deref(params.scaleGradDesc));
            auto bias_grad_tv  = get_inner_expanded_tv<1>(deref(params.biasGradDesc));
            auto input_dims    = deref(params.inputDesc).GetLengths();
            auto outer_size    = input_dims[0];
            auto inner_size    = std::accumulate(
                input_dims.begin() + 2, input_dims.end(), 1UL, std::multiplies<size_t>());

            kernel(params.input,
                   params.doutput,
                   params.weight,
                   params.meanVar,
                   params.dinput,
                   params.scaleGrad,
                   params.biasGrad,
                   outer_size,
                   inner_size,
                   x_tv,
                   dy_tv,
                   scale_tv,
                   mean_var_tv,
                   dx_tv,
                   scale_grad_tv,
                   bias_grad_tv);
        };
    };

    return result;
}

} // namespace instancenorm

} // namespace solver

} // namespace miopen
