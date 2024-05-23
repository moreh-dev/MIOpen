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
#define LOCAL_SIZE_REDUCE_FWD 256

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

bool HingeEmbeddingLossFwd::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::loss::HingeEmbeddingLossFwdProblemDescription& problem) const
{
    if(problem.GetInputDesc().GetSize() > 5)
        return false;
    return true;
}

ConvSolution HingeEmbeddingLossFwd::GetSolution(
    const ExecutionContext& context,
    const miopen::loss::HingeEmbeddingLossFwdProblemDescription& problem) const
{
    std::ignore = context;
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetOutputDesc().GetType();
    auto in_dtype     = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto target_dtype = miopen::GetDataType(problem.GetTargetDesc().GetType());
    auto size         = problem.GetInputDesc().GetElementSize();

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IN_OUT_TYPE", in_dtype == "bfloat16" ? "ushort" : in_dtype},
        {"TARGET_TYPE", target_dtype},
        {"LOCAL_SIZE", LOCAL_SIZE},
    };

    /* Phase 1: Add loss kernel */
    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE},
                                                         {size},
                                                         "MIOpenHingeEmbeddingLoss.cpp",
                                                         "HingeEmbeddingLossFwd",
                                                         build_params));

    /* Phase 2: Add reduce kernels */
    auto _size = size;
    do
    {
        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_REDUCE_FWD},
                                                             {_size},
                                                             "MIOpenHingeEmbeddingLoss.cpp",
                                                             "LossSum",
                                                             build_params));
        _size = AlignUp(_size, LOCAL_SIZE_REDUCE_FWD) / LOCAL_SIZE_REDUCE_FWD;
    } while(_size > 1);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::loss::FwdInvokeParams>();
            auto elapsed          = 0.f;

            /* Phase 1: Calc loss for each element. */
            {
                decltype(auto) kernel = handle_.Run(kernels.front());
                auto input_tv         = get_inner_expanded_tv(deref(params.inputDesc));
                auto target_tv        = get_inner_expanded_tv(deref(params.targetDesc));
                kernel(params.input,
                       params.target,
                       params.workspace,
                       params.margin,
                       params.divisor,
                       input_tv,
                       target_tv);
            }
            if(handle_.IsProfilingEnabled())
            {
                elapsed = handle_.GetKernelTime();
            }

            /* Phase 2: Reduce */
            auto reduce_in = params.workspace;
            auto reduce_out =
                static_cast<Data_t>(static_cast<char*>(params.workspace) +
                                    deref(params.inputDesc).GetElementSize() *
                                        get_data_size(deref(params.outputDesc).GetType()));
            auto size = deref(params.inputDesc).GetElementSize();
            for(int i = 1; i < kernels.size(); ++i)
            {
                decltype(auto) kernel = handle_.Run(kernels[i]);
                if(i + 1 != kernels.size())
                {
                    kernel(reduce_in, reduce_out, size);
                    std::swap(reduce_in, reduce_out);
                }
                else
                {
                    kernel(reduce_in, params.output, size);
                }
                size = AlignUp(size, LOCAL_SIZE_REDUCE_FWD) / LOCAL_SIZE_REDUCE_FWD;
            }

            if(handle_.IsProfilingEnabled())
            {
                handle_.ResetKernelTime();
                handle_.AccumKernelTime(elapsed);
            };
        };
    };

    return result;
}

std::size_t HingeEmbeddingLossFwd::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::loss::HingeEmbeddingLossFwdProblemDescription& problem) const
{
    size_t inputElements  = problem.GetInputDesc().GetElementSize();
    size_t reduceElements = (inputElements + LOCAL_SIZE_REDUCE_FWD - 1) / LOCAL_SIZE_REDUCE_FWD;
    size_t res =
        (inputElements + reduceElements) * get_data_size(problem.GetOutputDesc().GetType());

    return res;
}

bool HingeEmbeddingLossUnreducedFwd::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::loss::HingeEmbeddingLossUnreducedFwdProblemDescription& problem) const
{
    if(problem.GetInputDesc().GetSize() > 5)
        return false;
    return true;
}

ConvSolution HingeEmbeddingLossUnreducedFwd::GetSolution(
    const ExecutionContext& context,
    const miopen::loss::HingeEmbeddingLossUnreducedFwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto in_dtype     = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto dtype        = problem.GetOutputDesc().GetType();
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
                                                         "HingeEmbeddingLossUnreducedFwd",
                                                         build_params));

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::loss::UnreducedFwdInvokeParams>();
            auto input_tv         = get_inner_expanded_tv(deref(params.inputDesc));
            auto target_tv        = get_inner_expanded_tv(deref(params.targetDesc));

            kernel(params.input, params.target, params.output, params.margin, input_tv, target_tv);
        };
    };

    return result;
}

} // namespace loss

} // namespace solver

} // namespace miopen
