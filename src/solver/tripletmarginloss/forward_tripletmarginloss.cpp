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
#include <miopen/tripletmarginloss/invoke_params.hpp>
#include <miopen/tripletmarginloss/solvers.hpp>
#include <miopen/tripletmarginloss/utils.hpp>

#define LOCAL_SIZE_DIST 256
#define LOCAL_SIZE_DIST_REDUCE 256
#define LOCAL_SIZE_LOSS_FWD 256
#define LOCAL_SIZE_LOSS_FWD_REDUCE 256

namespace miopen {

namespace solver {

namespace tripletmarginloss {

inline void ConstructDistParams(const ExecutionContext& context,
                                const miopen::tripletmarginloss::ForwardProblemDescription& problem,
                                ConvSolution& result,
                                const KernelBuildParameters& build_params)
{
    auto input_size = problem.GetADesc().GetElementSize();
    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_DIST},
                                                         {input_size},
                                                         "MIOpenTripletMarginLoss.cpp",
                                                         "TripletMarginLossDist2d",
                                                         build_params));

    auto reduce_size        = problem.GetADesc().GetLengths()[1];
    auto output_numel       = problem.GetADesc().GetLengths()[0] * 3;
    auto reqd_work_item_cnt = get_reqd_work_item_cnt(context, LOCAL_SIZE_DIST_REDUCE);
    if(is_parallelism(reqd_work_item_cnt, output_numel, reduce_size))
    {
        auto parallelism_size = get_parallelism_size(reqd_work_item_cnt, output_numel, reduce_size);
        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_DIST_REDUCE},
                                                             {parallelism_size * output_numel},
                                                             "MIOpenSum.cpp",
                                                             "SumParallelFwdContiguous",
                                                             build_params));
    }
    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_DIST_REDUCE},
                                                         {output_numel},
                                                         "MIOpenTripletMarginLoss.cpp",
                                                         "TripletMarginLossDistSumPow2d",
                                                         build_params));
}

inline void RunDistKernels(const std::vector<Kernel>& kernels,
                           const Handle& handle_,
                           const AnyInvokeParams& raw_params,
                           float& elapsed,
                           int& kernelCnt,
                           Data_t& work_a,
                           Data_t& work_b)
{
    auto params = raw_params.CastTo<miopen::tripletmarginloss::InvokeParams>();

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

    auto reduce_size        = params.aDesc->GetSize() == 2 ? params.aDesc->GetLengths()[1] : 1;
    auto output_numel       = params.aDesc->GetLengths()[0] * 3;
    auto reqd_work_item_cnt = get_reqd_work_item_cnt(handle_, LOCAL_SIZE_DIST_REDUCE);

    if(is_parallelism(reqd_work_item_cnt, output_numel, reduce_size))
    {
        auto parallelism_size = get_parallelism_size(reqd_work_item_cnt, output_numel, reduce_size);
        auto parallel_kernel  = handle_.Run(kernels[kernelCnt++]);
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
        kernel(
            work_b, work_a, (size_t)output_numel, (size_t)parallelism_size, params.p, params.eps);
        if(handle_.IsProfilingEnabled())
            elapsed += handle_.GetKernelTime();
    }
    else
    {
        auto kernel = handle_.Run(kernels[kernelCnt++]);
        kernel(work_a, work_b, (size_t)output_numel, (size_t)reduce_size, params.p, params.eps);
        if(handle_.IsProfilingEnabled())
            elapsed += handle_.GetKernelTime();
        std::swap(work_a, work_b);
    }
}

bool IsImprovementOverROCm(const ExecutionContext& /*context*/,
                           const miopen::tripletmarginloss::ForwardProblemDescription& problem)
{
    if(problem.GetADesc().GetLengths()[1] > LOCAL_SIZE_DIST_REDUCE)
        return false;
    return true;
}

bool Forward2d::IsApplicable(
    const ExecutionContext& context,
    const miopen::tripletmarginloss::ForwardProblemDescription& problem) const
{
    if(!IsImprovementOverROCm(context, problem))
        return false;
    return true;
}

std::size_t Forward2d::GetWorkspaceSize(
    const ExecutionContext& context,
    const miopen::tripletmarginloss::ForwardProblemDescription& problem) const
{
    std::size_t size =
        problem.GetADesc().GetElementSize() * get_data_size(problem.GetODesc().GetType()) * 3;

    auto reduce_size        = problem.GetADesc().GetLengths()[1];
    auto output_numel       = problem.GetADesc().GetLengths()[0] * 3;
    auto reqd_work_item_cnt = get_reqd_work_item_cnt(context, LOCAL_SIZE_DIST_REDUCE);
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

bool UnreducedForward2d::IsApplicable(
    const ExecutionContext& context,
    const miopen::tripletmarginloss::ForwardProblemDescription& problem) const
{
    if(problem.IsReduced())
        return false;
    if(!Forward2d::IsApplicable(context, problem))
        return false;
    return true;
}

ConvSolution UnreducedForward2d::GetSolution(
    const ExecutionContext& context,
    const miopen::tripletmarginloss::ForwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetODesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetADesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetODesc().GetType());

    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
    };

    /* Phase 1: Calc distance for each vector. */
    ConstructDistParams(context, problem, result, build_params);

    /* Phase 2: Calc loss for each vector. */
    {
        auto output_size = problem.GetADesc().GetLengths()[0];
        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_LOSS_FWD},
                                                             {output_size},
                                                             "MIOpenTripletMarginLoss.cpp",
                                                             "TripletMarginLossUnreducedForward2d",
                                                             build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::tripletmarginloss::InvokeParams>();

            float elapsed = 0.0f;
            int kernelCnt = 0;

            auto work_a = params.workspace;
            auto work_b = reinterpret_cast<Data_t>(reinterpret_cast<char*>(params.workspace) +
                                                   params.aDesc->GetElementSize() *
                                                       get_data_size(params.oDesc->GetType()) * 3);

            /* Phase 1: Calc distance for each vector. */
            RunDistKernels(kernels, handle_, raw_params, elapsed, kernelCnt, work_a, work_b);

            /* Phase 2: Calc loss for each vector. */
            {
                auto O_tv   = get_inner_expanded_tv<1>(deref(params.oDesc));
                auto kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(work_a, params.o, params.margin, params.eps, params.swap, O_tv);
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

bool ReducedForward2d::IsApplicable(
    const ExecutionContext& context,
    const miopen::tripletmarginloss::ForwardProblemDescription& problem) const
{
    if(!problem.IsReduced())
        return false;
    if(!Forward2d::IsApplicable(context, problem))
        return false;
    return true;
}

ConvSolution ReducedForward2d::GetSolution(
    const ExecutionContext& context,
    const miopen::tripletmarginloss::ForwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetODesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetADesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetODesc().GetType());

    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"REDUCE_SIZE", LOCAL_SIZE_LOSS_FWD_REDUCE},
    };

    /* Phase 1: Calc distance for each vector. */
    ConstructDistParams(context, problem, result, build_params);

    /* Phase 2: Calc loss for each vector. */
    {
        auto output_size = problem.GetADesc().GetLengths()[0];
        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_LOSS_FWD},
                                                             {output_size},
                                                             "MIOpenTripletMarginLoss.cpp",
                                                             "TripletMarginLossForward2d",
                                                             build_params));
    }

    /* Phase 3: Reduce */
    {
        auto size = problem.GetADesc().GetLengths()[0];
        do
        {
            result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_LOSS_FWD_REDUCE},
                                                                 {size},
                                                                 "MIOpenLossReduce.cpp",
                                                                 "LossSum",
                                                                 build_params));
            size = AlignUp(size, LOCAL_SIZE_LOSS_FWD_REDUCE) / LOCAL_SIZE_LOSS_FWD_REDUCE;
        } while(size > 1);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            auto params = raw_params.CastTo<miopen::tripletmarginloss::InvokeParams>();

            float elapsed = 0.0f;
            int kernelCnt = 0;

            auto work_a = params.workspace;
            auto work_b = reinterpret_cast<Data_t>(reinterpret_cast<char*>(params.workspace) +
                                                   params.aDesc->GetElementSize() *
                                                       get_data_size(params.oDesc->GetType()) * 3);

            /* Phase 1: Calc distance for each vector. */
            RunDistKernels(kernels, handle_, raw_params, elapsed, kernelCnt, work_a, work_b);

            /* Phase 2: Calc loss for each vector. */
            {
                auto kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(work_a,
                       work_b,
                       params.margin,
                       params.eps,
                       params.swap,
                       params.divisor,
                       params.aDesc->GetLengths()[0]);
                if(handle_.IsProfilingEnabled())
                    elapsed += handle_.GetKernelTime();
                std::swap(work_a, work_b);
            }

            /* Phase 3: Reduce */
            {
                size_t size = params.aDesc->GetLengths()[0];
                while(kernelCnt < kernels.size())
                {
                    auto kernel = handle_.Run(kernels[kernelCnt++]);
                    if(kernelCnt < kernels.size())
                    {
                        kernel(work_a, work_b, size);
                        std::swap(work_a, work_b);
                    }
                    else
                    {
                        kernel(work_a, params.o, size);
                    }
                    size = AlignUp(size, LOCAL_SIZE_LOSS_FWD_REDUCE) / LOCAL_SIZE_LOSS_FWD_REDUCE;
                    if(handle_.IsProfilingEnabled())
                        elapsed += handle_.GetKernelTime();
                }
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

} // namespace tripletmarginloss

} // namespace solver

} // namespace miopen
