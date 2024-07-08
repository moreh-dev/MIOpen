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
#include <miopen/kernel_build_params.hpp>
#include <miopen/repeat/invoke_params.hpp>
#include <miopen/repeat/solvers.hpp>
#include <miopen/repeat.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace repeat {

bool RepeatForward::IsApplicable(const ExecutionContext& context,
                                 const miopen::repeat::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
    {
        return false;
    }
    return true;
}

ConvSolution RepeatForward::GetSolution(const ExecutionContext& context,
                                        const miopen::repeat::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype = problem.GetXDesc().GetType();
    auto xdims = problem.GetXDesc().GetLengths();
    auto ydims = problem.GetYDesc().GetLengths();

    auto output_size = std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>{});

    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(output_size, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenRepeat.cpp";
        kernel.kernel_name = "RepeatForward";

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

            auto xdims  = params.xDyDesc->GetLengths();
            auto ydims  = params.yDxDesc->GetLengths();
            auto offset = params.offset;

            auto inout_size =
                std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>{});

            std::vector<uint64_t> input_dimensions(5, 1);
            std::vector<uint64_t> output_dimensions(5, 1);

            for(int i = 0; i < xdims.size(); ++i)
            {
                input_dimensions[i] = xdims[i];
            }

            for(int i = 0; i < ydims.size(); ++i)
            {
                output_dimensions[i] = ydims[i];
            }

            kernel(params.xDy,
                   params.yDx,
                   inout_size,
                   offset,
                   input_dimensions[0],
                   input_dimensions[1],
                   input_dimensions[2],
                   input_dimensions[3],
                   input_dimensions[4],
                   output_dimensions[0],
                   output_dimensions[1],
                   output_dimensions[2],
                   output_dimensions[3],
                   output_dimensions[4]);
        };
    };

    return result;
}

} // namespace repeat

} // namespace solver

} // namespace miopen
