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

#include <cstddef>

#include <miopen/datatype.hpp>
#include <miopen/gather.hpp>
#include <miopen/gather/invoke_params.hpp>
#include <miopen/gather/solvers.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace gather {

bool GatherV2Backward::IsApplicable(const ExecutionContext& /*context*/,
                                    const miopen::gather::BwdProblemDescription& problem) const
{
    auto paramGradDesc = problem.GetParamGradDesc();
    if(paramGradDesc.GetType() != miopenFloat && paramGradDesc.GetType() != miopenHalf &&
       paramGradDesc.GetType() != miopenBFloat16)
    {
        return false;
    }
    if(problem.GetGatherDesc().getMode() != MIOPEN_GATHER_V2)
    {
        return false;
    }
    return true;
}

ConvSolution
GatherV2Backward::GetSolution(const ExecutionContext& context,
                              const miopen::gather::BwdProblemDescription& problem) const
{
    std::ignore = context;
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetParamGradDesc().GetType();
    auto in_out_dtype = miopen::GetDataType(dtype);
    auto indices_type = miopen::GetDataType(problem.GetIndicesDesc().GetType());

    auto gatherDesc      = problem.GetGatherDesc();
    auto batch_dims      = gatherDesc.getBatchDims();
    auto param_grad_len  = problem.GetParamGradDesc().GetLengths();
    auto outGrad_numel   = problem.GetOutputGradDesc().GetElementSize();
    auto paramGrad_numel = problem.GetParamGradDesc().GetElementSize();
    auto dim             = gatherDesc.getDim();

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IO_TYPE", in_out_dtype == "bfloat16" ? "ushort" : in_out_dtype},
        {"INDEX_TYPE", indices_type},
    };

    /* Phase 1: Fill param grad with zeros */
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(paramGrad_numel, xlocalsize);

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenFill.cpp";
        kernel.kernel_name = "FillZero";

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);
        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(1);
        kernel.g_wk.push_back(1);

        result.construction_params.push_back(kernel);
    }

    /* Phase 2: Gather backward */
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(outGrad_numel, xlocalsize);

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenGatherV2.cpp";

        if(batch_dims > 0)
        {
            kernel.kernel_name = "BatchedGatherV2Backward";
        }
        else
        {
            kernel.kernel_name = "GatherV2Backward";
        }

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);
        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(1);
        kernel.g_wk.push_back(1);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory =
        [paramGrad_numel, batch_dims, dim, outGrad_numel](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) params = raw_params.CastTo<miopen::gather::BwdInvokeParams>();

                float elapsed = 0.0f;
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

                /* Phase 1: Fill param grad with zeros */
                {
                    decltype(auto) kernel = handle_.Run(kernels.front());
                    kernel(params.paramGrad, paramGrad_numel);
                }

                auto param_grad_len = deref(params.paramGradDesc).GetLengths();
                size_t batch_size   = 1;
                size_t outer_size   = 1;
                size_t inner_size   = 1;

                for(uint32_t i = 0; i < batch_dims; ++i)
                {
                    batch_size *= param_grad_len[i];
                }
                for(uint32_t i = batch_dims; i < dim; ++i)
                {
                    outer_size *= param_grad_len[i];
                }
                for(uint32_t i = dim + 1; i < param_grad_len.size(); ++i)
                {
                    inner_size *= param_grad_len[i];
                }

                size_t gather_dim_size = param_grad_len[dim];
                size_t indices_numel   = deref(params.indicesDesc).GetElementSize() / batch_size;

                /* Phase 2: Gather backward */
                {
                    decltype(auto) kernel         = handle_.Run(kernels.back());
                    const bool is_axis_zero       = (outer_size == 1);
                    const bool is_batch_dims_zero = (batch_size == 1);

                    if(batch_dims > 0)
                    {
                        auto output_grad_tv = miopen::gather::reshape<4>(
                            miopen::deref(params.outputGradDesc),
                            {batch_size, outer_size, indices_numel, inner_size});
                        kernel(params.outputGrad,
                               params.indices,
                               params.paramGrad,
                               output_grad_tv,
                               outer_size,
                               gather_dim_size,
                               indices_numel,
                               inner_size,
                               outGrad_numel,
                               is_batch_dims_zero);
                    }
                    else
                    {
                        auto output_grad_tv =
                            miopen::gather::reshape<3>(miopen::deref(params.outputGradDesc),
                                                       {outer_size, indices_numel, inner_size});
                        kernel(params.outputGrad,
                               params.indices,
                               params.paramGrad,
                               output_grad_tv,
                               gather_dim_size,
                               indices_numel,
                               inner_size,
                               outGrad_numel,
                               is_axis_zero);
                    }
                }

                if(reset_profiling_state)
                {
                    handle_.EnableProfiling(true);
                }
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
                }
            };
        };

    return result;
}

} // namespace gather

} // namespace solver

} // namespace miopen
