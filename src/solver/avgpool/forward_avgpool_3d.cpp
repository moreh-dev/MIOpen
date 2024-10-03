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

#include "miopen/conv_solution.hpp"
#include "miopen/execution_context.hpp"
#include "miopen/invoke_params.hpp"
#include "miopen/tensor_view_utils.hpp"
#include <miopen/avgpool/solvers.hpp>

#include <miopen/avgpool/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/avgpool.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE_FWD_3D 256

namespace miopen {

namespace solver {

namespace avgpool {

bool IsOverRocmFwd3d(const miopen::avgpool::FwdProblemDescription& problem)
{
    auto dtype      = problem.GetOutputDesc().GetType();
    auto in_nelems  = problem.GetInputDesc().GetElementSize();
    auto out_nelems = problem.GetOutputDesc().GetElementSize();
    auto mul_nc = problem.GetOutputDesc().GetLengths()[0] * problem.GetOutputDesc().GetLengths()[1];
    auto N      = problem.GetOutputDesc().GetLengths()[0];
    auto in_over_out = static_cast<float>(in_nelems) / out_nelems;

    if(dtype == miopenFloat)
    {
        if(in_over_out < 2 || in_over_out >= 262144 || (out_nelems >= 10125000 && N > 4))
        {
            return true;
        }
    }
    else if(dtype == miopenHalf)
    {
        if(in_nelems >= 201326592 || (in_over_out < 2 && mul_nc < 8192))
        {
            return true;
        }
    }
    else if(dtype == miopenBFloat16)
    {
        if((out_nelems >= 5971968 && in_over_out < 2) || out_nelems >= 74088000)
        {
            return true;
        }
    }
    return false;
}

bool AvgPoolForward3d::IsApplicable(const ExecutionContext&,
                                    const miopen::avgpool::FwdProblemDescription& problem) const
{
    if(problem.GetInputDesc().GetNumDims() != 5 || problem.GetOutputDesc().GetNumDims() != 5)
    {
        return false;
    }
    if(!IsOverRocmFwd3d(problem))
    {
        return false;
    }
    return true;
}

ConvSolution
AvgPoolForward3d::GetSolution(const ExecutionContext& context,
                              const miopen::avgpool::FwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result       = ConvSolution{miopenStatusSuccess};
    auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());
    auto dtype        = problem.GetOutputDesc().GetType();
    size_t N_total    = problem.GetNtotal();

    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype}};

    result.construction_params.push_back(make_hip_kernel(
        {LOCAL_SIZE_FWD_3D}, {N_total}, "MIOpenAvgPool.cpp", "AvgPoolForward3d", build_params));

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::avgpool::FwdInvokeParams>();

            decltype(auto) kernel = handle_.Run(kernels.front());

            auto input_tv  = get_inner_expanded_tv<5>(deref(params.inputDesc));
            auto output_tv = get_inner_expanded_tv<5>(deref(params.outputDesc));

            auto N  = deref(params.inputDesc).GetLengths()[0];
            auto C  = deref(params.inputDesc).GetLengths()[1];
            auto D  = deref(params.inputDesc).GetLengths()[2];
            auto H  = deref(params.inputDesc).GetLengths()[3];
            auto W  = deref(params.inputDesc).GetLengths()[4];
            auto OD = deref(params.outputDesc).GetLengths()[2];
            auto OH = deref(params.outputDesc).GetLengths()[3];
            auto OW = deref(params.outputDesc).GetLengths()[4];

            kernel(params.input,
                   params.output,
                   N,
                   C,
                   D,
                   H,
                   W,
                   OD,
                   OH,
                   OW,
                   params.KD,
                   params.KH,
                   params.KW,
                   params.SD,
                   params.SH,
                   params.SW,
                   params.PD,
                   params.PH,
                   params.PW,
                   params.count_include_pad,
                   params.divisor_override,
                   input_tv,
                   output_tv);
        };
    };

    return result;
}

} // namespace avgpool

} // namespace solver

} // namespace miopen
