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

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace tripletmarginloss {

inline void
ConstructDistParams(const ExecutionContext& context,
                    const miopen::tripletmarginloss::BackwardProblemDescription& problem,
                    ConvSolution& result,
                    const KernelBuildParameters& build_params)
{
    auto input_size = problem.GetADesc().GetElementSize();
    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE},
                                                         {input_size},
                                                         "MIOpenTripletMarginLoss.cpp",
                                                         "TripletMarginLossDist2d",
                                                         build_params));

    auto reduce_size        = problem.GetADesc().GetLengths()[1];
    auto output_numel       = problem.GetADesc().GetLengths()[0] * 3;
    auto reqd_work_item_cnt = get_reqd_work_item_cnt(context, LOCAL_SIZE);
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
    auto reqd_work_item_cnt = get_reqd_work_item_cnt(handle_, LOCAL_SIZE);

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
            work_b, work_a, (uint64_t)output_numel, (uint64_t)parallelism_size, (uint64_t)1, false);
        if(handle_.IsProfilingEnabled())
            elapsed += handle_.GetKernelTime();
    }
    else
    {
        auto kernel = handle_.Run(kernels[kernelCnt++]);
        kernel(work_a, work_b, (uint64_t)output_numel, (uint64_t)reduce_size, (uint64_t)1, false);
        if(handle_.IsProfilingEnabled())
            elapsed += handle_.GetKernelTime();
        std::swap(work_a, work_b);
    }
}

bool Backward2d::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::tripletmarginloss::BackwardProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsRightLength())
        return false;
    return true;
}

std::size_t Backward2d::GetWorkspaceSize(
    const ExecutionContext& context,
    const miopen::tripletmarginloss::BackwardProblemDescription& problem) const
{
    std::size_t size =
        problem.GetADesc().GetElementSize() * get_data_size(problem.GetdADesc().GetType()) * 3;

    auto reduce_size        = problem.GetADesc().GetLengths()[1];
    auto output_numel       = problem.GetADesc().GetLengths()[0] * 3;
    auto reqd_work_item_cnt = get_reqd_work_item_cnt(context, LOCAL_SIZE);
    if(is_parallelism(reqd_work_item_cnt, output_numel, reduce_size))
    {
        auto parallelism_size = get_parallelism_size(reqd_work_item_cnt, output_numel, reduce_size);
        size += parallelism_size * output_numel * get_data_size(problem.GetdADesc().GetType());
    }
    else
    {
        size += output_numel * get_data_size(problem.GetdADesc().GetType());
    }

    return size;
}

bool UnreducedBackward2d::IsApplicable(
    const ExecutionContext& context,
    const miopen::tripletmarginloss::BackwardProblemDescription& problem) const
{
    if(!problem.IsUnreduced())
        return false;
    if(!Backward2d::IsApplicable(context, problem))
        return false;
    return true;
}

ConvSolution UnreducedBackward2d::GetSolution(
    const ExecutionContext& context,
    const miopen::tripletmarginloss::BackwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetdADesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetdADesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetdODesc().GetType());

    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"D_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
    };

    /* Phase 1: Calc distance for each vector. */
    ConstructDistParams(context, problem, result, build_params);

    /* Phase 2: Calc gradient for each vector. */
    {
        auto size = problem.GetdADesc().GetElementSize();
        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE},
                                                             {size},
                                                             "MIOpenTripletMarginLoss.cpp",
                                                             "TripletMarginLossUnreducedBackward2d",
                                                             build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            auto params = raw_params.CastTo<miopen::tripletmarginloss::InvokeParams>();

            float elapsed = 0.0f;
            int kernelCnt = 0;

            auto work_a = params.workspace;
            auto work_b = reinterpret_cast<Data_t>(reinterpret_cast<char*>(params.workspace) +
                                                   params.aDesc->GetElementSize() *
                                                       get_data_size(params.dADesc->GetType()) * 3);

            /* Phase 1: Calc distance for each vector. */
            RunDistKernels(kernels, handle_, raw_params, elapsed, kernelCnt, work_a, work_b);

            /* Phase 2: Calc gradient for each vector. */
            {
                auto A_tv  = get_inner_expanded_tv<2>(deref(params.aDesc));
                auto P_tv  = get_inner_expanded_tv<2>(deref(params.pDesc));
                auto N_tv  = get_inner_expanded_tv<2>(deref(params.nDesc));
                auto dO_tv = get_inner_expanded_tv<1>(deref(params.dODesc));
                auto dA_tv = get_inner_expanded_tv<2>(deref(params.dADesc));
                auto dP_tv = get_inner_expanded_tv<2>(deref(params.dPDesc));
                auto dN_tv = get_inner_expanded_tv<2>(deref(params.dNDesc));

                auto kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(work_a,
                       params.anchor,
                       params.positive,
                       params.negative,
                       params.dO,
                       params.dA,
                       params.dP,
                       params.dN,
                       params.margin,
                       params.p,
                       params.eps,
                       params.swap,
                       A_tv,
                       P_tv,
                       N_tv,
                       dO_tv,
                       dA_tv,
                       dP_tv,
                       dN_tv);
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

bool ReducedBackward2d::IsApplicable(
    const ExecutionContext& context,
    const miopen::tripletmarginloss::BackwardProblemDescription& problem) const
{
    if(!problem.IsReduced())
        return false;
    if(!Backward2d::IsApplicable(context, problem))
        return false;
    return true;
}

ConvSolution ReducedBackward2d::GetSolution(
    const ExecutionContext& context,
    const miopen::tripletmarginloss::BackwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetdADesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetdADesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetdODesc().GetType());

    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"D_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
    };

    /* Phase 1: Calc distance for each vector. */
    ConstructDistParams(context, problem, result, build_params);

    /* Phase 2: Calc gradient for each vector. */
    {
        auto output_size = problem.GetdADesc().GetElementSize();
        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE},
                                                             {output_size},
                                                             "MIOpenTripletMarginLoss.cpp",
                                                             "TripletMarginLossBackward2d",
                                                             build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            auto params = raw_params.CastTo<miopen::tripletmarginloss::InvokeParams>();

            float elapsed = 0.0f;
            int kernelCnt = 0;

            auto work_a = params.workspace;
            auto work_b = reinterpret_cast<Data_t>(reinterpret_cast<char*>(params.workspace) +
                                                   params.aDesc->GetElementSize() *
                                                       get_data_size(params.dADesc->GetType()) * 3);

            /* Phase 1: Calc distance for each vector. */
            RunDistKernels(kernels, handle_, raw_params, elapsed, kernelCnt, work_a, work_b);

            /* Phase 2: Calc gradient for each vector. */
            {
                auto A_tv  = get_inner_expanded_tv<2>(deref(params.aDesc));
                auto P_tv  = get_inner_expanded_tv<2>(deref(params.pDesc));
                auto N_tv  = get_inner_expanded_tv<2>(deref(params.nDesc));
                auto dA_tv = get_inner_expanded_tv<2>(deref(params.dADesc));
                auto dP_tv = get_inner_expanded_tv<2>(deref(params.dPDesc));
                auto dN_tv = get_inner_expanded_tv<2>(deref(params.dNDesc));

                auto kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(work_a,
                       params.anchor,
                       params.positive,
                       params.negative,
                       params.dO,
                       params.dA,
                       params.dP,
                       params.dN,
                       params.margin,
                       params.p,
                       params.eps,
                       params.swap,
                       params.divisor,
                       A_tv,
                       P_tv,
                       N_tv,
                       dA_tv,
                       dP_tv,
                       dN_tv);
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

} // namespace tripletmarginloss

} // namespace solver

} // namespace miopen
