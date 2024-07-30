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

#include <miopen/indexselect/solvers.hpp>

#include <miopen/indexselect/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/indexselect.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>

#include <iostream>

namespace miopen {

namespace solver {

namespace indexselect {

static bool
IsImprovementOverROCm([[maybe_unused]] const miopen::indexselect::ProblemDescription& problem)
{
    return true;
}

bool IndexSelectForward::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::indexselect::ProblemDescription& problem) const
{
    if(!problem.IsAllPacked())
        return false;
    if(!IsImprovementOverROCm(problem))
        return false;
    return true;
}

ConvSolution
IndexSelectForward::GetSolution([[maybe_unused]] const ExecutionContext& context,
                                const miopen::indexselect::ProblemDescription& problem) const
{
    static const size_t LOCAL_SIZE = 256;
    auto result                    = ConvSolution{miopenStatusSuccess};

    auto dtype = problem.GetXDesc().GetType();
    auto xdims = problem.GetXDesc().GetLengths();
    auto ydims = problem.GetYDesc().GetLengths();
    auto dim   = problem.GetDim();

    size_t xlocalsize = LOCAL_SIZE;
    size_t ylocalsize = 1;
    size_t zlocalsize = 1;

    size_t output_size = 1;
    for(size_t i = 0; i < ydims.size(); i++)
    {
        output_size *= ydims[i];
    }

    size_t xgridsize = output_size;
    if(xgridsize % LOCAL_SIZE != 0)
    {
        xgridsize = (xgridsize / LOCAL_SIZE + 1) * LOCAL_SIZE;
    }
    size_t ygridsize = 1;
    size_t zgridsize = 1;

    auto kernel        = KernelInfo();
    kernel.kernel_file = "MIOpenIndexSelect.cpp";
    if(problem.GetXDesc().IsContiguous())
    {
        kernel.kernel_name = "IndexSelectForwardContiguous";
    }
    else
    {
        kernel.kernel_name = "IndexSelectForward";
    }

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
    };

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
            decltype(auto) params = raw_params.CastTo<miopen::indexselect::InvokeParamsForward>();

            auto xlens    = params.xDesc.GetLengths();
            auto ylens    = params.yDesc.GetLengths();
            auto xstrides = params.xDesc.GetStrides();
            auto ystrides = params.yDesc.GetStrides();
            auto dim      = params.dim;

            kernel(params.x,
                   params.y,
                   (int)xlens[0],
                   (int)xlens[1],
                   (int)xlens[2],
                   (int)xlens[3],
                   (int)ylens[0],
                   (int)ylens[1],
                   (int)ylens[2],
                   (int)ylens[3],
                   (int)xstrides[0],
                   (int)xstrides[1],
                   (int)xstrides[2],
                   (int)xstrides[3],
                   (int)ystrides[0],
                   (int)ystrides[1],
                   (int)ystrides[2],
                   (int)ystrides[3],
                   (int)dim,
                   params.indices);
        };
    };
    return result;
}

std::size_t IndexSelectForward::GetWorkspaceSize(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::indexselect::ProblemDescription& problem) const
{
    return 0;
}

} // namespace indexselect

} // namespace solver

} // namespace miopen
