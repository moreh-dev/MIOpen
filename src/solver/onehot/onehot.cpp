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
#include <miopen/onehot/invoke_params.hpp>
#include <miopen/onehot/solvers.hpp>
#include <miopen/onehot.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace onehot {

size_t get_reqd_work_item_cnt(const ExecutionContext& context)
{
    // At least 4 WGs per one CU
    return static_cast<size_t>(LOCAL_SIZE * context.GetStream().GetMaxComputeUnits() * 4);
}

size_t get_reqd_work_item_cnt(const Handle& handle)
{
    // At least 4 WGs per one CU
    return static_cast<size_t>(LOCAL_SIZE * handle.GetMaxComputeUnits() * 4);
}

size_t get_parallelism_size(size_t reqd_work_item_cnt, size_t output_numel, size_t reduce_size)
{
    size_t parallelism_size = 1ULL;
    while(parallelism_size * output_numel < reqd_work_item_cnt &&
          parallelism_size < std::sqrt(reduce_size))
    {
        parallelism_size *= 2ULL;
    }
    return parallelism_size;
}

bool is_parallelism(size_t reqd_work_item_cnt, size_t output_numel, size_t reduce_size)
{
    return !(output_numel > reqd_work_item_cnt) &&
           (output_numel * reduce_size > reqd_work_item_cnt);
}

bool OneHot::IsApplicable(const ExecutionContext& context,
                          const miopen::onehot::ProblemDescription& problem) const
{
    return true;
}

ConvSolution OneHot::GetSolution(const ExecutionContext& context,
                                 const miopen::onehot::ProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto dtype = problem.GetInDesc().GetType();

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(problem.getInputSize(), xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenOneHot.cpp";
        kernel.kernel_name = "OneHotContiguous";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"MIOPEN_USE_INT32", static_cast<int>(dtype == miopenInt32)},
            {"LOCAL_SIZE", LOCAL_SIZE},
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
            decltype(auto) params = raw_params.CastTo<miopen::onehot::InvokeParams>();

            kernel(params.input, params.output, params.input_size, params.num_classes);
        };
    };

    return result;
}

std::size_t OneHot::GetWorkspaceSize(const ExecutionContext& context,
                                     const miopen::onehot::ProblemDescription& problem) const
{
    return 0;
}

} // namespace onehot

} // namespace solver

} // namespace miopen
