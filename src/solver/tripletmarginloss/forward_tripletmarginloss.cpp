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

#include "miopen/kernel.hpp"
#include "miopen/miopen.h"
#include <cstddef>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tripletmarginloss.hpp>
#include <miopen/tripletmarginloss/invoke_params.hpp>
#include <miopen/tripletmarginloss/solvers.hpp>
#include <miopen/tripletmarginloss/utils.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

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

namespace tripletmarginloss {

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

bool UnreducedForward2d::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::tripletmarginloss::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsRightLength())
        return false;
    return true;
}

ConvSolution
UnreducedForward2d::GetSolution(const ExecutionContext& context,
                                const miopen::tripletmarginloss::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetODesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetADesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetODesc().GetType());
    auto input_size   = problem.GetADesc().GetElementSize();

    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
    };

    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE},
                                                         {input_size},
                                                         "MIOpenTripletMarginLoss.cpp",
                                                         "TripletMarginLossForward2d_1",
                                                         build_params));

    auto reduce_size  = problem.GetADesc().GetSize() == 2 ? problem.GetADesc().GetLengths()[1] : 1;
    auto output_numel = problem.GetADesc().GetLengths()[0] * 3;
    auto reqd_work_item_cnt = get_reqd_work_item_cnt(context);

    if(is_parallelism(reqd_work_item_cnt, output_numel, reduce_size))
    {
        auto parallelism_size = get_parallelism_size(reqd_work_item_cnt, output_numel, reduce_size);
        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE},
                                                             {parallelism_size * output_numel},
                                                             "MIOpenSum.cpp",
                                                             "SumParallelFwdContiguous",
                                                             build_params));
    }

    result.construction_params.push_back(make_hip_kernel(
        {LOCAL_SIZE}, {output_numel}, "MIOpenSum.cpp", "SumFwdContiguous", build_params));

    auto output_size = problem.GetADesc().GetLengths()[0];
    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE},
                                                         {output_size},
                                                         "MIOpenTripletMarginLoss.cpp",
                                                         "TripletMarginLossUnreducedForward2d_2",
                                                         build_params));

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::tripletmarginloss::InvokeParams>();

            float elapsed = 0.0f;
            int kernelCnt = 0;

            auto work_a = params.workspace;
            auto work_b = reinterpret_cast<Data_t>(reinterpret_cast<char*>(params.workspace) +
                                                   params.aDesc->GetElementSize() *
                                                       get_data_size(params.oDesc->GetType()) * 3);

            {
                auto A_tv   = get_inner_expanded_tv<2>(deref(params.aDesc));
                auto P_tv   = get_inner_expanded_tv<2>(deref(params.pDesc));
                auto N_tv   = get_inner_expanded_tv<2>(deref(params.nDesc));
                auto kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(params.anchor,
                       params.positive,
                       params.negative,
                       work_a,
                       params.p,
                       params.eps,
                       A_tv,
                       P_tv,
                       N_tv);
                if(handle_.IsProfilingEnabled())
                    elapsed += handle_.GetKernelTime();
            }

            auto reduce_size  = params.aDesc->GetSize() == 2 ? params.aDesc->GetLengths()[1] : 1;
            auto output_numel = params.aDesc->GetLengths()[0] * 3;
            auto reqd_work_item_cnt = get_reqd_work_item_cnt(handle_);

            if(is_parallelism(reqd_work_item_cnt, output_numel, reduce_size))
            {
                auto parallelism_size =
                    get_parallelism_size(reqd_work_item_cnt, output_numel, reduce_size);
                auto parallel_kernel = handle_.Run(kernels[kernelCnt++]);
                parallel_kernel(work_a,
                                work_b,
                                (uint64_t)output_numel,
                                (uint64_t)reduce_size,
                                (uint64_t)parallelism_size,
                                (uint64_t)1,
                                false);
                if(handle_.IsProfilingEnabled())
                    elapsed += handle_.GetKernelTime();

                auto kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(work_b,
                       work_a,
                       (uint64_t)output_numel,
                       (uint64_t)parallelism_size,
                       (uint64_t)1,
                       false);
                if(handle_.IsProfilingEnabled())
                    elapsed += handle_.GetKernelTime();
            }
            else
            {
                auto kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(work_a,
                       work_b,
                       (uint64_t)output_numel,
                       (uint64_t)reduce_size,
                       (uint64_t)1,
                       false);
                if(handle_.IsProfilingEnabled())
                    elapsed += handle_.GetKernelTime();
                std::swap(work_a, work_b);
            }

            {
                auto O_tv   = get_inner_expanded_tv<1>(deref(params.oDesc));
                auto kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(work_a, params.o, params.margin, params.p, params.swap, O_tv);
                if(handle_.IsProfilingEnabled())
                    elapsed += handle_.GetKernelTime();
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

std::size_t UnreducedForward2d::GetWorkspaceSize(
    const ExecutionContext& context,
    const miopen::tripletmarginloss::ProblemDescription& problem) const
{
    std::size_t size =
        problem.GetADesc().GetElementSize() * get_data_size(problem.GetODesc().GetType()) * 3;

    auto reduce_size  = problem.GetADesc().GetSize() == 2 ? problem.GetADesc().GetLengths()[1] : 1;
    auto output_numel = problem.GetADesc().GetLengths()[0] * 3;
    auto reqd_work_item_cnt = get_reqd_work_item_cnt(context);
    if(is_parallelism(reqd_work_item_cnt, output_numel, reduce_size))
    {
        auto parallelism_size = get_parallelism_size(reqd_work_item_cnt, output_numel, reduce_size);
        size += parallelism_size * output_numel * get_data_size(problem.GetODesc().GetType());
    }
    else
    {
        size += output_numel * get_data_size(problem.GetODesc().GetType());
    }

    return size;
}

} // namespace tripletmarginloss

} // namespace solver

} // namespace miopen
