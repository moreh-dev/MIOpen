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
#include <miopen/onehot/invoke_params.hpp>
#include <miopen/onehot/solvers.hpp>
#include <miopen/onehot.hpp>
#include <miopen/target_properties.hpp>

// TODO - trungbui: find optimal local_size
#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace onehot {

bool OneHot::IsApplicable(const ExecutionContext& /*context*/,
                          const miopen::onehot::ProblemDescription& problem) const
{
    if(!problem.IsShapeMatch())
        return false;
    if(!problem.IsNumClassesValid())
        return false;
    if(!problem.IsAllPacked())
        return false;
    return true;
}

ConvSolution OneHot::GetSolution(const ExecutionContext& context,
                                 const miopen::onehot::ProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto input_dtype  = miopen::GetDataType(problem.GetInDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutDesc().GetType());
    auto err_dtype    = miopen::GetDataType(problem.GetErrDesc().GetType());

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
        {"INPUT_TYPE", input_dtype},
        {"OUTPUT_TYPE", output_dtype},
        {"ERR_TYPE", err_dtype},
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

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::onehot::InvokeParams>();

            kernel(params.input, params.output, params.err, params.input_size, params.num_classes);
        };
    };

    return result;
}

} // namespace onehot

} // namespace solver

} // namespace miopen
