/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <miopen/datatype.hpp>
#include <miopen/tensor.hpp>
#include <miopen/sumtosize.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/reduce/invoke_params.hpp>
#include <miopen/reduce/solvers.hpp>
#include <miopen/buffer_info.hpp>

namespace miopen {

namespace sumtosize {

std::vector<int> GetReduceDim(const std::vector<size_t>& inputDim,
                              const std::vector<size_t>& outputDim)
{
    std::vector<int> reduce_dims;
    for(int i = inputDim.size() - 1; i >= outputDim.size(); i--)
        if(inputDim[i] != 1)
            reduce_dims.push_back(i);
    for(int i = outputDim.size() - 1; i >= 0; i--)
        if(inputDim[i] != 1 && outputDim[i] == 1)
            reduce_dims.push_back(i);
    if(reduce_dims.empty())
        reduce_dims.push_back(inputDim.size() - 1);
    return reduce_dims;
}

namespace {
void CheckValidDim(const std::vector<size_t>& inputDim, const std::vector<size_t>& outputDim)
{
    if(outputDim.size() > inputDim.size())
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "Output tensor num dims should not be greater than input tensor num dims.");
    }
    for(auto i = 0; i < outputDim.size(); i++)
        if(outputDim[i] != inputDim[i] && outputDim[i] != 1)
            MIOPEN_THROW(miopenStatusBadParm, "Cannot reduce input dim to output dim.");
}

MultiBufferWorkspaceTraits GetMultiBufferWorkspaceTraits(const TensorDescriptor& inputDesc,
                                                         const std::size_t ws_size,
                                                         const std::vector<int>& reduce_dims)
{
    // If reduce 1 dim, no need additional workspace
    // If reduce 2-3 dims, need 1-2 additional workspaces
    // If reduce > 3 dims, need 2 additional workspaces and then swap between them in each reduction
    auto data_size = get_data_size(inputDesc.GetType());
    if(reduce_dims.size() == 1)
        return MultiBufferWorkspaceTraits{ws_size};
    else if(reduce_dims.size() == 2)
    {
        return MultiBufferWorkspaceTraits{ws_size,
                                          data_size * inputDesc.GetElementSize() /
                                              inputDesc.GetLengths()[reduce_dims[0]]};
    }
    else
    {
        return MultiBufferWorkspaceTraits{
            ws_size,
            data_size * inputDesc.GetElementSize() / inputDesc.GetLengths()[reduce_dims[0]],
            data_size * inputDesc.GetElementSize() / inputDesc.GetLengths()[reduce_dims[0]] /
                inputDesc.GetLengths()[reduce_dims[1]]};
    }
}
} // namespace

std::size_t GetWorkspaceReductionAllDims(ExecutionContext ctx,
                                         const TensorDescriptor& inputDesc,
                                         const std::vector<int>& reduce_dims)
{
    size_t ws_size = 0;
    std::vector<size_t> lens(inputDesc.GetLengths());
    auto data_type = inputDesc.GetType();
    for(int reduce_dim : reduce_dims)
    {
        TensorDescriptor xDesc = TensorDescriptor(data_type, lens);
        lens[reduce_dim]       = 1;
        // reduce::ProblemDescriptionCalculation require yDesc_ to remove reduce_dim
        std::vector<size_t> lens_without_red(lens);
        lens_without_red.erase(lens_without_red.begin() + reduce_dim);
        TensorDescriptor yDesc = TensorDescriptor(data_type, lens_without_red);
        // Tensor is always contiguous, so we don't care about stride when solving problem => don't
        // need to pass the same lens and strides
        const auto problem =
            reduce::ProblemDescriptionCalculation{MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN,
                                                  xDesc,
                                                  yDesc,
                                                  reduce_dim,
                                                  MIOPEN_REDUCE_CALCULATION_SUM};
        const auto solvers    = solver::SolverContainer<solver::reduce::SumForward>{};
        auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);
        if(pair_size_vector.empty())
        {
            return static_cast<size_t>(-1);
        }
        else
            ws_size = std::max(ws_size, pair_size_vector.front().second);
    }
    return ws_size;
}

} // namespace sumtosize

std::size_t GetSumToSizeForwardWorkspaceSize(Handle& handle,
                                             const TensorDescriptor& inputDesc,
                                             const TensorDescriptor& outputDesc)
{
    sumtosize::CheckValidDim(inputDesc.GetLengths(), outputDesc.GetLengths());
    auto ctx = ExecutionContext{&handle};
    std::vector<int> reduce_dims =
        sumtosize::GetReduceDim(inputDesc.GetLengths(), outputDesc.GetLengths());

    // we have multiple problems, each problem will reduce each dim
    // => workspace required to solve all problems = max workspace among all problems
    auto ws_size = sumtosize::GetWorkspaceReductionAllDims(ctx, inputDesc, reduce_dims);

    // workspace required = workspace required to solve all problems + workspace to store temporary
    // output tensor
    return sumtosize::GetMultiBufferWorkspaceTraits(inputDesc, ws_size, reduce_dims).GetSize();
}

miopenStatus_t SumToSizeForward(Handle& handle,
                                const TensorDescriptor& inputDesc,
                                ConstData_t input,
                                const TensorDescriptor& outputDesc,
                                Data_t output,
                                Data_t workspace,
                                size_t /*workspaceSizeInBytes*/)
{
    sumtosize::CheckValidDim(inputDesc.GetLengths(), outputDesc.GetLengths());

    // Get MultiBufferWorkspaceTraits. Currently we have no ways to pass MultiBufferWorkspaceTraits
    // from GetSumToSizeForwardWorkspaceSize to SumToSizeForward so below code will be duplicate
    // code of GetSumToSizeForwardWorkspaceSize
    auto ctx = ExecutionContext{&handle};
    std::vector<int> reduce_dims =
        sumtosize::GetReduceDim(inputDesc.GetLengths(), outputDesc.GetLengths());
    auto ws_size = sumtosize::GetWorkspaceReductionAllDims(ctx, inputDesc, reduce_dims);
    auto wt      = sumtosize::GetMultiBufferWorkspaceTraits(inputDesc, ws_size, reduce_dims);

    // Reduce each dim inside reduce_dims with solver::reduce::SumForward
    std::vector<size_t> lens(inputDesc.GetLengths());
    auto data_type = inputDesc.GetType();
    Data_t reduce_in, reduce_out;
    if(reduce_dims.size() == 2)
    {
        reduce_out = static_cast<Data_t>(static_cast<std::byte*>(workspace) + wt.GetOffset(1));
    }
    if(reduce_dims.size() >= 3)
    {
        reduce_out = static_cast<Data_t>(static_cast<std::byte*>(workspace) + wt.GetOffset(1));
        reduce_in  = static_cast<Data_t>(static_cast<std::byte*>(workspace) + wt.GetOffset(2));
    }

    // Start profiling
    // TODO: this profiling method will count time to init problem, invoke_params, ...
    float elapsed = 0.0f;
    HipEventPtr start;
    HipEventPtr stop;
    const bool profiling = handle.IsProfilingEnabled();
    if(profiling)
    {
        handle.EnableProfiling(false);
        start = miopen::make_hip_event();
        stop  = miopen::make_hip_event();
        hipEventRecord(start.get(), handle.GetStream());
    }

    for(int i = 0; i < reduce_dims.size(); i++)
    {
        TensorDescriptor xDesc = TensorDescriptor(data_type, lens);
        lens[reduce_dims[i]]   = 1;
        // reduce::ProblemDescriptionCalculation require yDesc_ to remove reduce_dim
        std::vector<size_t> lens_without_red(lens);
        lens_without_red.erase(lens_without_red.begin() + reduce_dims[i]);
        TensorDescriptor yDesc = TensorDescriptor(data_type, lens_without_red);
        // Tensor is always contiguous, so we don't care about stride when solving problem => don't
        // need to pass the same lens and strides
        const auto problem =
            reduce::ProblemDescriptionCalculation{MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN,
                                                  xDesc,
                                                  yDesc,
                                                  reduce_dims[i],
                                                  MIOPEN_REDUCE_CALCULATION_SUM};

        const auto invoke_params = [&]() {
            auto tmp           = reduce::CalculationInvokeParams{};
            tmp.type           = InvokeType::Run;
            tmp.xDesc          = &xDesc;
            tmp.yDesc          = &yDesc;
            tmp.x              = (i == 0) ? input : reduce_in;
            tmp.y              = (i == reduce_dims.size() - 1) ? output : reduce_out;
            tmp.workspace      = workspace;
            tmp.workspace_size = wt.v_offset[1];
            tmp.nanPropagation = MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN;
            tmp.dim            = reduce_dims[i];
            return tmp;
        }();

        const auto algo    = AlgorithmName{"SumForward"};
        const auto solvers = solver::SolverContainer<solver::reduce::SumForward>{};

        solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

        std::swap(reduce_in, reduce_out);
    }

    // End of profiling
    if(profiling)
    {
        hipEventRecord(stop.get(), handle.GetStream());
        hipEventSynchronize(stop.get());
        hipEventElapsedTime(&elapsed, start.get(), stop.get());

        // Clean up
        hipEventDestroy(start.get());
        hipEventDestroy(stop.get());
        handle.ResetKernelTime();
        handle.AccumKernelTime(elapsed);

        handle.EnableProfiling(true);
    };
    return miopenStatusSuccess;
}

} // namespace miopen
