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
#pragma once

#include "InputFlags.hpp"
#include "driver.hpp"
#include "miopen/miopen.h"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <../test/verify.hpp>
#include <miopen/sumtosize.hpp>
#include <miopen/tensor_view_utils.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloSumToSizeForwardRunHost(const miopen::TensorDescriptor& inputDesc,
                                   const miopen::TensorDescriptor& outputDesc,
                                   const Tgpu* input,
                                   Tcheck* output)
{
    std::vector<int> reduce_dims =
        miopen::sumtosize::GetReduceDim(inputDesc.GetLengths(), outputDesc.GetLengths());
    size_t reduce_size = 1;
    for(auto reduce_dim : reduce_dims)
        reduce_size *= inputDesc.GetLengths()[reduce_dim];
    auto input_tv  = miopen::get_inner_expanded_tv<5>(inputDesc);
    auto output_tv = miopen::get_inner_expanded_tv<5>(outputDesc);

    par_ford(outputDesc.GetElementSize())([&](size_t gid) {
        tensor_layout_t<5> idx_output(output_tv, gid);
        for(size_t i = 0; i < reduce_size; i++)
        {
            // construct idx_input
            tensor_layout_t<5> idx_input;
            size_t reduce_idx = i;

            int cur_reduce = 0;
            for(int dim = inputDesc.GetNumDims() - 1; dim >= 0; dim--)
            {
                if(cur_reduce < reduce_dims.size() && dim == reduce_dims[cur_reduce])
                {
                    // This is reduce dim
                    cur_reduce++;
                    idx_input.layout[dim] = reduce_idx % input_tv.size[dim];
                    reduce_idx /= input_tv.size[dim];
                }
                else
                {
                    // This is NOT reduce dim
                    idx_input.layout[dim] = idx_output.layout[dim];
                }
            }
            output[gid] += static_cast<Tcheck>(input[input_tv.get_tensor_view_idx(idx_input)]);
        }
    });
    return miopenStatusSuccess;
};

template <typename Tgpu, typename Tref>
class SumToSizeDriver : public Driver
{
public:
    SumToSizeDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunBackwardGPU() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~SumToSizeDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;

    // forw = 0 -> run both fw, bw, = 1 -> run only fw, = 2 -> run only bw
    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> output;
    std::vector<Tref> ref_output;

    size_t ws_sizeInBytes;
};

template <typename Tgpu, typename Tref>
int SumToSizeDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "1",
                         "Run Forward or Backward. 0 to run both Fw and Bw, 1 to run only Fw, 2 to "
                         "run only Bw (Default=1)",
                         "int");
    inflags.AddInputFlag(
        "input", 'I', "8x104x1", "Shape of input tensor (Default=8x104x1)", "tensor");
    inflags.AddInputFlag(
        "output", 'O', "1x104x1", "Shape of output tensor (Default=1x104x1)", "tensor");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time Each Layer (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SumToSizeDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    forw = inflags.GetValueInt("forw");
    if(forw != 1)
    {
        MIOPEN_THROW("Only support forward mode");
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SumToSizeDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len = inflags.GetValueTensor("input").lengths;
    if(SetTensorNd(inputDesc, in_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("SetTensorNd: Invalid input tensor shape.");
    std::vector<int> out_len = inflags.GetValueTensor("output").lengths;
    if(SetTensorNd(outputDesc, out_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("SetTensorNd: Invalid output tensor shape.");
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SumToSizeDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;

    size_t i_sz = GetTensorSpace(inputDesc);
    input_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, i_sz, sizeof(Tgpu)));
    input       = std::vector<Tgpu>(i_sz);

    for(size_t i = 0; i < i_sz; i++)
    {
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0), static_cast<Tgpu>(1));
    }
    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
    {
        std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }
    miopenGetSumToSizeForwardWorkSpaceSize(GetHandle(), inputDesc, outputDesc, &ws_sizeInBytes);
    if(ws_sizeInBytes == static_cast<size_t>(-1))
    {
        return miopenStatusAllocFailed;
    }
    workspace_dev = std::make_unique<GPUMem>(ctx, ws_sizeInBytes, sizeof(std::byte));

    size_t o_sz = GetTensorSpace(outputDesc);
    output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, o_sz, sizeof(Tgpu)));
    output      = std::vector<Tgpu>(o_sz);
    ref_output  = std::vector<Tref>(o_sz);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SumToSizeDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenStatus_t status = miopenSumToSizeForward(GetHandle(),
                                                       inputDesc,
                                                       input_dev->GetMem(),
                                                       outputDesc,
                                                       output_dev->GetMem(),
                                                       workspace_dev->GetMem(),
                                                       ws_sizeInBytes);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in Forward SumToSize");

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Forward SumToSize Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward SumToSize Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (input_grad) from GPU, size: " << output_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SumToSizeDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SumToSizeDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloSumToSizeForwardRunHost(
        miopen::deref(inputDesc), miopen::deref(outputDesc), input.data(), ref_output.data());
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref SumToSizeDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int SumToSizeDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(output, ref_output);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward SumToSize FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward SumToSize Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SumToSizeDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}
