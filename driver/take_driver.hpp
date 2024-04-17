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
#ifndef GUARD_MIOPEN_TAKE_DRIVER_HPP
#define GUARD_MIOPEN_TAKE_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#ifndef MLO_SUMMHOST_H_
#define MLO_SUMMHOST_H_

template <typename Tgpu, typename Tcheck>
int32_t mloTakeForwardRunHost(miopenTensorDescriptor_t inputDesc,
                              miopenTensorDescriptor_t indexDesc,
                              miopenTensorDescriptor_t outputDesc,
                              Tgpu* input,
                              const int32_t* index,
                              Tcheck* outputhost)
{
    auto input_dims  = miopen::deref(inputDesc).GetLengths();
    auto output_dims = miopen::deref(outputDesc).GetLengths();

    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());
    auto input_numel =
        std::accumulate(input_dims.begin(), input_dims.end(), 1L, std::multiplies<int64_t>());

    int32_t ret = 0;

    for(size_t o = 0; o < output_numel; o++)
    {
        int32_t index_v = index[o];
        if(index_v < -input_numel || index_v >= input_numel)
            continue;
        index_v += input_numel * (index_v < 0);
        outputhost[o] = input[index_v];
    }
    return ret;
}
#endif

template <typename Tgpu, typename Tref>
class TakeDriver : public Driver
{
public:
    TakeDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&indexDesc);
        miopenCreateTensorDescriptor(&outputDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetOutputTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~TakeDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(indexDesc);
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t indexDesc;
    miopenTensorDescriptor_t outputDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> index_dev;
    std::unique_ptr<GPUMem> out_dev;

    std::vector<Tgpu> in;
    std::vector<int32_t> index;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;
};

template <typename Tgpu, typename Tref>
int TakeDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Take (Default=1)", "int");
    inflags.AddInputFlag("in_n", 'n', "3", "Number of input N (Default=3)", "int");
    inflags.AddInputFlag("in_c", 'c', "2", "Number of Input Channels (Default=2)", "int");
    inflags.AddInputFlag("in_d", 'd', "0", "Input Depth (Default=0)", "int");
    inflags.AddInputFlag("in_h", 'm', "0", "Input Height (Default=0)", "int");
    inflags.AddInputFlag("in_w", 'w', "2", "Input Width (Default=2)", "int");
    inflags.AddInputFlag("out_n", 'N', "3", "Number of output N (Default=3)", "int");
    inflags.AddInputFlag("out_c", 'C', "0", "Number of Output Channels (Default=0)", "int");
    inflags.AddInputFlag("out_d", 'D', "0", "Output Depth (Default=0)", "int");
    inflags.AddInputFlag("out_h", 'H', "0", "Output Height (Default=0)", "int");
    inflags.AddInputFlag("out_w", 'W', "0", "Output Width (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "1", "Number of Iterations (Default=1)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'x', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TakeDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TakeDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();
    SetTensorNd(inputDesc, in_len, data_type);
    std::vector<int> out_len = GetOutputTensorLengthsFromCmdLine();
    SetTensorNd(outputDesc, out_len, data_type);
    SetTensorNd(indexDesc, out_len, data_type);

    return 0;
}

template <typename Tgpu, typename Tref>
std::vector<int> TakeDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("in_n");
    int in_c = inflags.GetValueInt("in_c");
    int in_w = inflags.GetValueInt("in_w");
    int in_h = inflags.GetValueInt("in_h");
    // int in_h = 32;
    int in_d = inflags.GetValueInt("in_d");

    if((in_n != 0) && (in_c != 0) && (in_d != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_d, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_w});
    }
    else if((in_n != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_w});
    }
    else if(in_n != 0)
    {
        return std::vector<int>({in_n});
    }
    else
    {
        std::cerr << "Error Input Tensor Lengths\n" << std::endl;
        return std::vector<int>({0});
    }
}

template <typename Tgpu, typename Tref>
std::vector<int> TakeDriver<Tgpu, Tref>::GetOutputTensorLengthsFromCmdLine()
{
    int out_n = inflags.GetValueInt("out_n");
    int out_c = inflags.GetValueInt("out_c");
    int out_w = inflags.GetValueInt("out_w");
    int out_h = inflags.GetValueInt("out_h");
    int out_d = inflags.GetValueInt("out_d");

    if((out_n != 0) && (out_c != 0) && (out_d != 0) && (out_h != 0) && (out_w != 0))
    {
        return std::vector<int>({out_n, out_c, out_d, out_h, out_w});
    }
    else if((out_n != 0) && (out_c != 0) && (out_h != 0) && (out_w != 0))
    {
        return std::vector<int>({out_n, out_c, out_h, out_w});
    }
    else if((out_n != 0) && (out_c != 0) && (out_w != 0))
    {
        return std::vector<int>({out_n, out_c, out_w});
    }
    else if((out_n != 0) && (out_w != 0))
    {
        return std::vector<int>({out_n, out_w});
    }
    else if(out_n != 0)
    {
        return std::vector<int>({out_n});
    }
    else
    {
        std::cerr << "Error Output Tensor Lengths\n" << std::endl;
        return std::vector<int>({0});
    }
}

template <typename Tgpu, typename Tref>
int TakeDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz = GetTensorSize(inputDesc);
    // index_sz = out_sz
    size_t out_sz = GetTensorSize(outputDesc);

    uint32_t ctx = 0;

    in_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    index_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    out_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    in      = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    index   = std::vector<int32_t>(out_sz, static_cast<Tgpu>(0));
    out     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-1), static_cast<Tgpu>(1));
    }
    for(int i = 0; i < out_sz; i++)
    {
        // Generate random index from [-in_sz, insz)
        index[i] = prng::gen_A_to_B<int32_t>(static_cast<Tgpu>(-10), static_cast<Tgpu>(10));
    }

    if(in_dev->ToGPU(GetStream(), in.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;

    if(index_dev->ToGPU(GetStream(), index.data()) != 0)
        std::cerr << "Error copying (index) to GPU, size: " << in_dev->GetSize() << std::endl;

    if(out_dev->ToGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TakeDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenTakeForward(GetHandle(),
                          inputDesc,
                          in_dev->GetMem(),
                          indexDesc,
                          index_dev->GetMem(),
                          outputDesc,
                          out_dev->GetMem());

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
            std::cout << "Wall-clock Time Forward Sum Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward Sum Elapsed: " << kernel_average_time << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TakeDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloTakeForwardRunHost<Tgpu, Tref>(
        inputDesc, indexDesc, outputDesc, in.data(), index.data(), outhost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TakeDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

// TODO: check pull request
template <typename Tgpu, typename Tref>
Tref TakeDriver<Tgpu, Tref>::GetTolerance()
{
    // Computation error of fp16 is ~2^13 (=8192) bigger than
    // the one of fp32 because mantissa is shorter by 13 bits.
    auto tolerance = std::is_same<Tgpu, float>::value ? 1.5e-6 : 8.2e-3;

    // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
    if(std::is_same<Tgpu, bfloat16>::value)
        tolerance *= 8.0;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int TakeDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Take FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Take Verifies OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TakeDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_TAKE_DRIVER_HPP
