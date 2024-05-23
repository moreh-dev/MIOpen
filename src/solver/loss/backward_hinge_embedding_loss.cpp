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
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/loss/invoke_params.hpp>
#include <miopen/loss/solvers.hpp>
#include <miopen/hinge_embedding_loss.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace loss {

using tensor_view_5d_t = struct
{
    uint64_t stride[5];
    uint64_t size[5];
};

inline tensor_view_5d_t get_inner_expanded_tv(const miopen::TensorDescriptor Desc)
{
    auto dims    = Desc.GetLengths();
    auto strides = Desc.GetStrides();

    tensor_view_5d_t tv_5d;
    for(size_t i = 0; i < strides.size(); ++i)
    {
        tv_5d.stride[i] = strides[i];
        tv_5d.size[i]   = dims[i];
    }
    auto rest = strides.size();
    for(size_t j = rest; j < 5; ++j)
    {
        tv_5d.stride[j] = (rest == 0 ? 1 : strides[rest - 1]);
        tv_5d.size[j]   = 1;
    }
    return tv_5d;
}

const auto make_hip_kernel = [](std::vector<size_t> localsize,
                                std::vector<size_t> gridsize,
                                std::string kernel_file,
                                std::string kernel_name,
                                KernelBuildParameters build_params) {
    while(localsize.size() < 3)
        localsize.push_back(1);
    while(gridsize.size() < 3)
        gridsize.push_back(1);
    for(int i = 0; i < localsize.size(); ++i)
        gridsize[i] = AlignUp(gridsize[i], localsize[i]);
    return KernelInfo{
        build_params.GenerateFor(kbp::HIP{}), localsize, gridsize, kernel_file, kernel_name};
};

bool HingeEmbeddingLossBwd::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::loss::HingeEmbeddingLossBwdProblemDescription& problem) const
{
    if(problem.GetIDesc().GetSize() > 5)
        return false;
    return true;
}

ConvSolution HingeEmbeddingLossBwd::GetSolution(
    const ExecutionContext& context,
    const miopen::loss::HingeEmbeddingLossBwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto in_dtype     = miopen::GetDataType(problem.GetIDesc().GetType());
    auto dtype        = problem.GetdIDesc().GetType();
    auto target_dtype = miopen::GetDataType(problem.GetTDesc().GetType());

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IN_OUT_TYPE", in_dtype == "bfloat16" ? "ushort" : in_dtype},
        {"TARGET_TYPE", target_dtype},
        {"LOCAL_SIZE", LOCAL_SIZE},
    };

    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE},
                                                         {problem.GetIDesc().GetElementSize()},
                                                         "MIOpenHingeEmbeddingLoss.cpp",
                                                         "HingeEmbeddingLossBwd",
                                                         build_params));

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::loss::BwdInvokeParams>();
            auto I_tv             = get_inner_expanded_tv(deref(params.iDesc));
            auto T_tv             = get_inner_expanded_tv(deref(params.tDesc));
            auto dO_tv            = get_inner_expanded_tv(deref(params.dODesc));

            kernel(params.i,
                   params.t,
                   params.dO,
                   params.dI,
                   params.margin,
                   params.divisor,
                   I_tv,
                   T_tv,
                   dO_tv);
        };
    };

    return result;
}

bool HingeEmbeddingLossUnreducedBwd::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::loss::HingeEmbeddingLossUnreducedBwdProblemDescription& problem) const
{
    if(problem.GetIDesc().GetSize() > 5)
        return false;
    return true;
}

ConvSolution HingeEmbeddingLossUnreducedBwd::GetSolution(
    const ExecutionContext& context,
    const miopen::loss::HingeEmbeddingLossUnreducedBwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto in_dtype     = miopen::GetDataType(problem.GetIDesc().GetType());
    auto dtype        = problem.GetdIDesc().GetType();
    auto target_dtype = miopen::GetDataType(problem.GetTDesc().GetType());

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IN_OUT_TYPE", in_dtype == "bfloat16" ? "ushort" : in_dtype},
        {"TARGET_TYPE", target_dtype},
        {"LOCAL_SIZE", LOCAL_SIZE},
    };

    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE},
                                                         {problem.GetIDesc().GetElementSize()},
                                                         "MIOpenHingeEmbeddingLoss.cpp",
                                                         "HingeEmbeddingLossUnreducedBwd",
                                                         build_params));

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::loss::UnreducedBwdInvokeParams>();
            auto I_tv             = get_inner_expanded_tv(deref(params.iDesc));
            auto T_tv             = get_inner_expanded_tv(deref(params.tDesc));
            auto dO_tv            = get_inner_expanded_tv(deref(params.dODesc));

            kernel(params.i, params.t, params.dO, params.dI, params.margin, I_tv, T_tv, dO_tv);
        };
    };

    return result;
}

} // namespace loss

} // namespace solver

} // namespace miopen
