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

#define MAX_DIMS 5

#define LOCAL_SIZE_PRNG_STATE 256
#define LOCAL_MAX_PRNG_STATE (LOCAL_SIZE_PRNG_STATE * 128)

#define LOCAL_SIZE_CONTIGUOUS 256

namespace miopen {

namespace solver {

namespace rrelu {

size_t GetNumStates(const ExecutionContext& context,
                    const miopen::rrelu::ForwardProblemDescription& problem)
{
    return std::min({size_t(LOCAL_MAX_PRNG_STATE),
                     context.GetStream().GetImage3dMaxWidth(),
                     problem.GetInputDesc().GetElementSize()});
}

size_t GetNumThreads(const ExecutionContext& context,
                     const miopen::rrelu::ForwardProblemDescription& problem)
{
    size_t size = problem.GetInputDesc().GetElementSize();
    if(size == GetNumStates(context, problem))
        return size;
    size_t divisor = 1;
    while((1ul << divisor) * divisor <= size)
        ++divisor;
    --divisor;
    return (1ul << divisor);
}

void InitPRNGState(const ExecutionContext& context,
                   const miopen::rrelu::ForwardProblemDescription& problem,
                   ConvSolution& result)
{
    auto num_states   = GetNumStates(context, problem);
    auto build_params = KernelBuildParameters{
        {"RUN_INIT_PRNG", 1},
    };
    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_PRNG_STATE},
                                                         {num_states},
                                                         "MIOpenDropout.cl",
                                                         "InitKernelState",
                                                         build_params));
}

size_t Forward::GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::rrelu::ForwardProblemDescription& problem) const
{
    return GetNumStates(context, problem) * sizeof(prngStates);
}

bool ContiguouseForward::IsApplicable(const ExecutionContext& /*context*/,
                                      const miopen::rrelu::ForwardProblemDescription& problem) const
{
    if(!problem.IsAllPacked())
        return false;
    if(!problem.IsSameStride())
        return false;
    return true;
}

ConvSolution
ContiguouseForward::GetSolution(const ExecutionContext& context,
                                const miopen::rrelu::ForwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    /* Phase 1: Init Pseudo random number generator states */
    {
        InitPRNGState(context, problem, result);
    }

    /* Phase 2: RReLU */
    {
        auto dtype        = problem.GetOutputDesc().GetType();
        auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
        auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());

        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
            {"MAX_DIMS", MAX_DIMS},
        };

        auto nthreads = GetNumThreads(context, problem);
        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_CONTIGUOUS},
                                                             {nthreads},
                                                             "MIOpenRReLU.cpp",
                                                             "RReLUContiguous",
                                                             build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            auto params = raw_params.CastTo<miopen::rrelu::InvokeParams>();

            auto elapsed = 0.0f;
            HipEventPtr start;
            HipEventPtr stop;

            bool reset_profiling_state = false;
            if(handle_.IsProfilingEnabled())
            {
                reset_profiling_state = true;
                handle_.EnableProfiling(false);
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }

            int kernelCnt     = 0;
            auto prng_state   = params.workspace;
            size_t num_states = params.workspace_size / sizeof(prngStates);

            /* Phase 1: Init Pseudo random number generator states */
            {
                auto kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(prng_state, (ulong)0, (ulong)num_states);
            }

            /* Phase 2: RReLU */
            {
                auto size   = deref(params.inputDesc).GetElementSize();
                auto kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(params.input,
                       params.output,
                       params.noise,
                       params.lower,
                       params.upper,
                       size,
                       prng_state,
                       num_states);
            }

            if(reset_profiling_state)
                handle_.EnableProfiling(true);
            if(handle_.IsProfilingEnabled())
            {
                hipEventRecord(stop.get(), handle_.GetStream());
                hipEventSynchronize(stop.get());
                hipEventElapsedTime(&elapsed, start.get(), stop.get());

                // Clean up
                hipEventDestroy(start.get());
                hipEventDestroy(stop.get());
                handle_.ResetKernelTime();
                handle_.AccumKernelTime(elapsed);
            };
        };
    };

    return result;
}

bool nonContiguouseForward::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::rrelu::ForwardProblemDescription& problem) const
{
    if(problem.GetInputDesc().GetVectorLength() > MAX_DIMS)
        return false;
    return true;
}

ConvSolution
nonContiguouseForward::GetSolution(const ExecutionContext& context,
                                   const miopen::rrelu::ForwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    /* Phase 1: Init Pseudo random number generator states */
    {
        InitPRNGState(context, problem, result);
    }

    /* Phase 2: RReLU */
    {
        auto dtype        = problem.GetOutputDesc().GetType();
        auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
        auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());

        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
            {"MAX_DIMS", MAX_DIMS},
        };

        auto nthreads = GetNumThreads(context, problem);
        result.construction_params.push_back(make_hip_kernel(
            {LOCAL_SIZE_CONTIGUOUS}, {nthreads}, "MIOpenRReLU.cpp", "RReLU", build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            auto params = raw_params.CastTo<miopen::rrelu::InvokeParams>();

            auto elapsed = 0.0f;
            HipEventPtr start;
            HipEventPtr stop;

            bool reset_profiling_state = false;
            if(handle_.IsProfilingEnabled())
            {
                reset_profiling_state = true;
                handle_.EnableProfiling(false);
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }

            int kernelCnt     = 0;
            auto prng_state   = params.workspace;
            size_t num_states = params.workspace_size / sizeof(prngStates);

            /* Phase 1: Init Pseudo random number generator states */
            {
                auto kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(prng_state, 0, num_states);
            }

            /* Phase 2: RReLU */
            {
                auto input_tv  = get_inner_expanded_tv<MAX_DIMS>(deref(params.inputDesc));
                auto output_tv = get_inner_expanded_tv<MAX_DIMS>(deref(params.outputDesc));

                auto size   = deref(params.inputDesc).GetElementSize();
                auto kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(params.input,
                       params.output,
                       params.noise,
                       params.lower,
                       params.upper,
                       size,
                       input_tv,
                       output_tv,
                       prng_state,
                       num_states);
            }

            if(reset_profiling_state)
                handle_.EnableProfiling(true);
            if(handle_.IsProfilingEnabled())
            {
                hipEventRecord(stop.get(), handle_.GetStream());
                hipEventSynchronize(stop.get());
                hipEventElapsedTime(&elapsed, start.get(), stop.get());

                // Clean up
                hipEventDestroy(start.get());
                hipEventDestroy(stop.get());
                handle_.ResetKernelTime();
                handle_.AccumKernelTime(elapsed);
            };
        };
    };

    return result;
}

} // namespace rrelu
} // namespace solver
} // namespace miopen
