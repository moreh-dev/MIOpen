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

#include "miopen/miopen.h"
#include <cstddef>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/pad_reflection/invoke_params.hpp>
#include <miopen/pad_reflection/solvers.hpp>
#include <miopen/pad_reflection.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace pad_reflection {

bool PadReflectionFwd::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::pad_reflection::PadReflectionFwdProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsRightNumPadding())
        return false;
    return true;
}

ConvSolution PadReflectionFwd::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::pad_reflection::PadReflectionFwdProblemDescription& problem) const
{
    auto result       = ConvSolution{miopenStatusSuccess};
    auto input_dtype  = miopen::GetDataType(problem.GetXDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetYDesc().GetType());
    auto xdims        = problem.GetXDesc().GetLengths();
    auto ydims        = problem.GetYDesc().GetLengths();
    auto dtype        = problem.GetXDesc().GetType();

    {
        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenPadReflection.cpp";
        kernel.kernel_name =
            problem.IsContiguous() ? "PadReflection1dFwdContiguous" : "PadReflection1dFwd";
        auto output_numel =
            std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>());

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        };

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(output_numel, xlocalsize);
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

    if(problem.IsContiguous())
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels[0]);
                decltype(auto) params = raw_params.CastTo<miopen::pad_reflection::InvokeParams>();

                auto xdims = params.xDesc->GetLengths();
                auto ydims = params.yDesc->GetLengths();

                auto xstrides = params.xDesc->GetStrides();

                auto output_size =
                    std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>());

                auto padding = params.padding;
                long padding_l;
                padding_l = padding[0];

                size_t in_W           = xdims[2];
                size_t output_size_1  = ydims[1];
                size_t output_size_2  = ydims[2];
                size_t input_stride_0 = xstrides[0];
                size_t input_stride_1 = xstrides[1];
                size_t input_stride_2 = xstrides[2];
                kernel(params.x,
                       params.y,
                       output_size,
                       padding_l,
                       in_W,
                       output_size_1,
                       output_size_2,
                       input_stride_0,
                       input_stride_1,
                       input_stride_2);
            };
        };
    }
    else
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels[0]);
                decltype(auto) params = raw_params.CastTo<miopen::pad_reflection::InvokeParams>();

                auto xdims = params.xDesc->GetLengths();
                auto ydims = params.yDesc->GetLengths();

                auto xstrides = params.xDesc->GetStrides();
                auto ystrides = params.yDesc->GetStrides();

                auto output_size =
                    std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>());

                auto padding = params.padding;
                long padding_l;
                padding_l = padding[0];

                size_t in_W            = xdims[2];
                size_t output_size_1   = ydims[1];
                size_t output_size_2   = ydims[2];
                size_t output_stride_0 = ystrides[0];
                size_t output_stride_1 = ystrides[1];
                size_t output_stride_2 = ystrides[2];
                size_t input_stride_0  = xstrides[0];
                size_t input_stride_1  = xstrides[1];
                size_t input_stride_2  = xstrides[2];
                kernel(params.x,
                       params.y,
                       output_size,
                       padding_l,
                       in_W,
                       output_size_1,
                       output_size_2,
                       output_stride_0,
                       output_stride_1,
                       output_stride_2,
                       input_stride_0,
                       input_stride_1,
                       input_stride_2);
            };
        };
    }
    return result;
}

bool PadReflectionBwd::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::pad_reflection::PadReflectionBwdProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsRightNumPadding())
        return false;
    return true;
}

ConvSolution PadReflectionBwd::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::pad_reflection::PadReflectionBwdProblemDescription& problem) const
{
    auto result       = ConvSolution{miopenStatusSuccess};
    auto input_dtype  = miopen::GetDataType(problem.GetXDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetYDesc().GetType());
    auto xdims        = problem.GetXDesc().GetLengths();
    auto ydims        = problem.GetYDesc().GetLengths();
    auto dtype        = problem.GetXDesc().GetType();

    {
        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenPadReflection.cpp";
        kernel.kernel_name =
            problem.IsContiguous() ? "PadReflection1dBwdContiguous" : "PadReflection1dBwd";
        auto output_numel =
            std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>());

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        };

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(output_numel, xlocalsize);
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

    if(problem.IsContiguous())
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels[0]);
                decltype(auto) params = raw_params.CastTo<miopen::pad_reflection::InvokeParams>();

                auto xdims = params.xDesc->GetLengths();
                auto ydims = params.yDesc->GetLengths();

                auto xstrides = params.xDesc->GetStrides();

                auto output_size =
                    std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>());

                auto padding = params.padding;
                long padding_l;
                padding_l = padding[0];

                size_t in_W           = xdims[2];
                size_t output_size_1  = ydims[1];
                size_t output_size_2  = ydims[2];
                size_t input_stride_0 = xstrides[0];
                size_t input_stride_1 = xstrides[1];
                size_t input_stride_2 = xstrides[2];
                kernel(params.x,
                       params.y,
                       output_size,
                       padding_l,
                       in_W,
                       output_size_1,
                       output_size_2,
                       input_stride_0,
                       input_stride_1,
                       input_stride_2);
            };
        };
    }
    else
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels[0]);
                decltype(auto) params = raw_params.CastTo<miopen::pad_reflection::InvokeParams>();

                auto xdims = params.xDesc->GetLengths();
                auto ydims = params.yDesc->GetLengths();

                auto xstrides = params.xDesc->GetStrides();
                auto ystrides = params.yDesc->GetStrides();

                auto output_size =
                    std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>());

                auto padding = params.padding;
                long padding_l;
                padding_l = padding[0];

                size_t in_W            = xdims[2];
                size_t output_size_1   = ydims[1];
                size_t output_size_2   = ydims[2];
                size_t output_stride_0 = ystrides[0];
                size_t output_stride_1 = ystrides[1];
                size_t output_stride_2 = ystrides[2];
                size_t input_stride_0  = xstrides[0];
                size_t input_stride_1  = xstrides[1];
                size_t input_stride_2  = xstrides[2];
                kernel(params.x,
                       params.y,
                       output_size,
                       padding_l,
                       in_W,
                       output_size_1,
                       output_size_2,
                       output_stride_0,
                       output_stride_1,
                       output_stride_2,
                       input_stride_0,
                       input_stride_1,
                       input_stride_2);
            };
        };
    }
    return result;
}

} // namespace pad_reflection

} // namespace solver

} // namespace miopen
