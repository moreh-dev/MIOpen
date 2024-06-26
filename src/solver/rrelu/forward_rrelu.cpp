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

#define LOCAL_SIZE_FWD_CONTIGUOUS 256
#define LOCAL_SIZE_DATA_ASSIGN 256

namespace miopen {

namespace solver {

namespace rrelu {

size_t Forward::GetWorkspaceSize(const ExecutionContext& /*context*/,
                                 const miopen::rrelu::ForwardProblemDescription& problem) const
{
    size_t size = 0;
    if(!::miopen::rrelu::checkContiguous(problem.GetInputDesc()))
        size +=
            problem.GetInputDesc().GetElementSize() * GetTypeSize(problem.GetInputDesc().GetType());
    if(!::miopen::rrelu::checkContiguous(problem.GetOutputDesc()))
        size += problem.GetOutputDesc().GetElementSize() *
                GetTypeSize(problem.GetOutputDesc().GetType());
    return size;
}

bool Forward::IsApplicable(const ExecutionContext& /*context*/,
                           const miopen::rrelu::ForwardProblemDescription& problem) const
{
    if(!problem.IsAllContiguous() && problem.GetInputDesc().GetSize() > VIEW_DIMS)
        return false;
    return true;
}

ConvSolution Forward::GetSolution(const ExecutionContext& context,
                                  const miopen::rrelu::ForwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    if(!::miopen::rrelu::checkContiguous(problem.GetInputDesc()))
    {
        auto dtype     = problem.GetInputDesc().GetType();
        auto data_type = GetDataType(problem.GetInputDesc().GetType());
        auto size      = problem.GetInputDesc().GetElementSize();

        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"DTYPE", data_type == "bfloat16" ? "ushort" : data_type},
            {"VIEW_DIMS", VIEW_DIMS},
        };

        result.construction_params.push_back(make_hip_kernel(
            {LOCAL_SIZE_DATA_ASSIGN}, {size}, "MIOpenAssign.cpp", "Assign", build_params));
    }

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
            {"VIEW_DIMS", VIEW_DIMS},
        };

        auto nthreads = GetNumThreads(context, problem.GetInputDesc().GetElementSize());
        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_FWD_CONTIGUOUS},
                                                             {nthreads},
                                                             "MIOpenRReLU.cpp",
                                                             "RReLUForward",
                                                             build_params));
    }

    if(!::miopen::rrelu::checkContiguous(problem.GetOutputDesc()))
    {
        auto dtype     = problem.GetOutputDesc().GetType();
        auto data_type = GetDataType(problem.GetOutputDesc().GetType());
        auto size      = problem.GetOutputDesc().GetElementSize();

        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"DTYPE", data_type == "bfloat16" ? "ushort" : data_type},
            {"VIEW_DIMS", VIEW_DIMS},
        };

        result.construction_params.push_back(make_hip_kernel(
            {LOCAL_SIZE_DATA_ASSIGN}, {size}, "MIOpenAssign.cpp", "Assign", build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            auto params = raw_params.CastTo<miopen::rrelu::ForwardInvokeParams>();

            auto elapsed = 0.0f;
            HipEventPtr start;
            HipEventPtr stop;

            bool reset_profiling_state = false;
            if(kernels.size() > 1)
            {
                if(handle_.IsProfilingEnabled())
                {
                    reset_profiling_state = true;
                    handle_.EnableProfiling(false);
                    start = miopen::make_hip_event();
                    stop  = miopen::make_hip_event();
                    hipEventRecord(start.get(), handle_.GetStream());
                }
            }

            int kernelCnt = 0;
            auto input    = params.input;
            auto output   = params.output;

            if(params.workspace != nullptr)
            {
                if(!::miopen::rrelu::checkContiguous(deref(params.inputDesc)))
                {
                    input = params.workspace;
                    if(!::miopen::rrelu::checkContiguous(deref(params.outputDesc)))
                        output =
                            static_cast<Data_t>(static_cast<char*>(params.workspace) +
                                                deref(params.inputDesc).GetElementSize() *
                                                    GetTypeSize(deref(params.inputDesc).GetType()));
                }
                else
                    output = params.workspace;
            }

            if(input != params.input)
            {
                auto input_tv = get_inner_expanded_tv<VIEW_DIMS>(deref(params.inputDesc));
                auto kernel   = handle_.Run(kernels[kernelCnt++]);
                kernel(params.input, input, input_tv, false);
            }

            {
                auto prng_states  = params.states;
                size_t num_states = params.state_size / sizeof(prngStates);
                auto size         = deref(params.inputDesc).GetElementSize();
                auto kernel       = handle_.Run(kernels[kernelCnt++]);
                kernel(prng_states,
                       num_states,
                       input,
                       output,
                       params.noise,
                       params.lower,
                       params.upper,
                       size);
            }

            if(output != params.output)
            {
                auto output_tv = get_inner_expanded_tv<VIEW_DIMS>(deref(params.outputDesc));
                auto kernel    = handle_.Run(kernels[kernelCnt++]);
                kernel(params.output, output, output_tv, true);
            }

            if(kernels.size() > 1)
            {
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
            }
        };
    };

    return result;
}

} // namespace rrelu
} // namespace solver
} // namespace miopen
