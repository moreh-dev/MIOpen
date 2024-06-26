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
#include <miopen/dropout.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/rrelu/invoke_params.hpp>
#include <miopen/rrelu/solvers.hpp>
#include <miopen/rrelu/utils.hpp>

#define VIEW_DIMS 5

#define LOCAL_SIZE_BWD_NONCONTIGUOUS 256

namespace miopen {

namespace solver {

namespace rrelu {

bool nonContiguousBackward::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::rrelu::BackwardProblemDescription& problem) const
{
    if(problem.GetdInputDesc().GetVectorLength() > VIEW_DIMS)
        return false;
    return true;
}

ConvSolution
nonContiguousBackward::GetSolution(const ExecutionContext& /*context*/,
                                   const miopen::rrelu::BackwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto dtype        = problem.GetdInputDesc().GetType();
        auto input_dtype  = miopen::GetDataType(problem.GetdInputDesc().GetType());
        auto output_dtype = miopen::GetDataType(problem.GetdOutputDesc().GetType());

        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
            {"VIEW_DIMS", VIEW_DIMS},
        };

        auto size = problem.GetdInputDesc().GetElementSize();
        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_BWD_NONCONTIGUOUS},
                                                             {size},
                                                             "MIOpenRReLU.cpp",
                                                             "RReLUBackwardNd",
                                                             build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            auto params = raw_params.CastTo<miopen::rrelu::BackwardInvokeParams>();

            {
                auto doutput_tv = get_inner_expanded_tv<VIEW_DIMS>(deref(params.doutputDesc));
                auto dinput_tv  = get_inner_expanded_tv<VIEW_DIMS>(deref(params.dinputDesc));

                auto kernel = handle_.Run(kernels.front());
                kernel(params.noise, params.doutput, params.dinput, doutput_tv, dinput_tv);
            }
        };
    };

    return result;
}

} // namespace rrelu
} // namespace solver
} // namespace miopen
