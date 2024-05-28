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

#include "miopen/loss/problem_description.hpp"
#include "miopen/miopen.h"
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/loss/invoke_params.hpp>
#include <miopen/loss/solvers.hpp>
#include <miopen/hinge_embedding_loss.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/loss/utils.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace loss {

bool HingeEmbeddingLossBwd::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::loss::HingeEmbeddingLossBwdProblemDescription& problem) const
{
    if(problem.GetInputDesc().GetSize() > 5)
        return false;
    return true;
}

ConvSolution HingeEmbeddingLossBwd::GetSolution(
    const ExecutionContext& context,
    const miopen::loss::HingeEmbeddingLossBwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto in_dtype     = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto dtype        = problem.GetDinputDesc().GetType();
    auto target_dtype = miopen::GetDataType(problem.GetTargetDesc().GetType());

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IN_OUT_TYPE", in_dtype == "bfloat16" ? "ushort" : in_dtype},
        {"TARGET_TYPE", target_dtype},
        {"LOCAL_SIZE", LOCAL_SIZE},
    };

    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE},
                                                         {problem.GetInputDesc().GetElementSize()},
                                                         "MIOpenHingeEmbeddingLoss.cpp",
                                                         "HingeEmbeddingLossBwd",
                                                         build_params));

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::loss::BwdInvokeParams>();
            auto input_tv         = get_inner_expanded_tv(deref(params.inputDesc));
            auto target_tv        = get_inner_expanded_tv(deref(params.targetDesc));
            auto doutput_tv       = get_inner_expanded_tv(deref(params.doutputDesc));
            float divisor         = 1;
            if(params.reduction == MIOPEN_LOSS_REDUCTION_MEAN)
            {
                divisor = deref(params.inputDesc).GetElementSize();
            }

            kernel(params.input,
                   params.target,
                   params.doutput,
                   params.dinput,
                   params.margin,
                   divisor,
                   input_tv,
                   target_tv,
                   doutput_tv);
        };
    };

    return result;
}

} // namespace loss

} // namespace solver

} // namespace miopen
