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

#include "miopen/image_transform/adjust_brightness/invoke_params.hpp"
#include "miopen/image_transform/solvers.hpp"
#include "miopen/kernel_build_params.hpp"
#include "miopen/kernel_info.hpp"
#include "miopen/miopen.h"
#include "miopen/solver.hpp"
#include "miopen/tensor_view.hpp"

namespace miopen {
namespace solver {
namespace image_transform {
namespace adjust_brightness {

bool ImageAdjustBrightness::IsApplicable(
    const ExecutionContext& /* context */,
    const miopen::image_transform::adjust_brightness::ProblemDescription& problem) const
{
    return true;
}

ConvSolution ImageAdjustBrightness::GetSolution(
    const ExecutionContext& /* context */,
    const miopen::image_transform::adjust_brightness::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype = problem.GetInputTensorDesc().GetType();
    auto numel = problem.GetInputTensorDesc().GetElementSize();

    size_t xlocalsize = 256;
    size_t xgridsize  = AlignUp(numel, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenImageAdjustBrightness.cpp";
    if(problem.GetInputTensorDesc().IsContiguous() && problem.GetOutputTensorDesc().IsContiguous())
        kernel.kernel_name = "ImageAdjustBrightnessContiguous";
    else
        kernel.kernel_name = "ImageAdjustBrightness";

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
                              {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
                              {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)}};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params =
                raw_params.CastTo<miopen::image_transform::adjust_brightness::InvokeParams>();

            auto xdesc = miopen::deref(params.inputTensorDesc);
            auto ydesc = miopen::deref(params.outputTensorDesc);

            auto x_tv = get_inner_expanded_4d_tv(xdesc);
            auto y_tv = get_inner_expanded_4d_tv(ydesc);

            size_t N = xdesc.GetElementSize();

            if(kernel.name == "ImageAdjustBrightness")
            {
                kernel(
                    params.input_buf, params.output_buf, x_tv, y_tv, N, params.brightness_factor);
            }
            else
            {
                kernel(params.input_buf,
                       params.output_buf,
                       x_tv.offset,
                       y_tv.offset,
                       N,
                       params.brightness_factor);
            }
        };
    };

    return result;
}

} // namespace adjust_brightness
} // namespace image_transform
} // namespace solver
} // namespace miopen
