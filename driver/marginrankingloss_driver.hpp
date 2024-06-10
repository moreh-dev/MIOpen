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
#ifndef GUARD_MIOPEN_MARGINRANKINGLOSS_DRIVER_HPP
#define GUARD_MIOPEN_MARGINRANKINGLOSS_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "mloMarginRakningLossHost.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "util_driver.hpp"

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <cstdio>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <vector>

template <typename Tgpu, typename Tref>
class MarginRankingLossDriver : public Driver
{
public:
    MarginRankingLossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&input1Desc);
        miopenCreateTensorDescriptor(&input2Desc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&outGradDesc);
        miopenCreateTensorDescriptor(&in1GradDesc);
        miopenCreateTensorDescriptor(&in2GradDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    std::vector<int> GetTensorDimsFromCmd();
    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~MarginRankingLossDriver() override
    {
        miopenDestroyTensorDescriptor(input1Desc);
        miopenDestroyTensorDescriptor(input2Desc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(outGradDesc);
        miopenDestroyTensorDescriptor(in1GradDesc);
        miopenDestroyTensorDescriptor(in2GradDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t input1Desc;
    miopenTensorDescriptor_t input2Desc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t outGradDesc;
    miopenTensorDescriptor_t in1GradDesc;
    miopenTensorDescriptor_t in2GradDesc;

    std::unique_ptr<GPUMem> input1_dev;
    std::unique_ptr<GPUMem> input2_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> outGrad_dev;
    std::unique_ptr<GPUMem> in1Grad_dev;
    std::unique_ptr<GPUMem> in2Grad_dev;

    std::vector<Tgpu> input1;
    std::vector<Tgpu> input2;
    std::vector<Tgpu> target;
    std::vector<Tgpu> output;
    std::vector<Tgpu> outGrad;
    std::vector<Tgpu> in1Grad;
    std::vector<Tgpu> in2Grad;

    std::vector<Tref> out_host;
    std::vector<Tref> in1Grad_host;
    std::vector<Tref> in2Grad_host;

    std::vector<int> dims;
    float margin;
    float divisor;
    bool is_forward;
    miopenMarginRakningLossReductionMode_t reduction_mode;
};

template <typename Tgpu, typename Tref>
int MarginRankingLossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> MarginRankingLossDriver<Tgpu, Tref>::GetTensorDimsFromCmd()
{
    std::string lengthsStr = inflags.GetValueStr("dims");

    std::vector<int> lengths;
    std::size_t pos = 0;
    std::size_t new_pos;

    new_pos = lengthsStr.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = lengthsStr.substr(pos, new_pos - pos);

        int len = std::stoi(sliceStr);

        lengths.push_back(len);

        pos     = new_pos + 1;
        new_pos = lengthsStr.find(',', pos);
    };

    std::string sliceStr = lengthsStr.substr(pos);
    int len              = std::stoi(sliceStr);

    lengths.push_back(len);

    while (lengths.size() < 5)
    {
        lengths.push_back(1);
    }

    return (lengths);
}

template <typename Tgpu, typename Tref>
int MarginRankingLossDriver<Tgpu, Tref>::GetandSetData()
{
    dims = GetTensorDimsFromCmd();
    SetTensorNd(input1Desc, dims, data_type);
    SetTensorNd(input2Desc, dims, data_type);
    SetTensorNd(targetDesc, dims, data_type);

    auto reduction_mode_string = inflags.GetValueStr("reduction");
    if(reduction_mode_string == "none")
    {
        reduction_mode = MIOPEN_MARGINRANKINGLOSS_REDUCTION_NONE;
        divisor        = 0.0f;
    }
    else if(reduction_mode_string == "sum")
    {
        reduction_mode = MIOPEN_MARGINRANKINGLOSS_REDUCTION_SUM;
        divisor        = 1.0f;
    }
    else if(reduction_mode_string == "mean")
    {
        reduction_mode = MIOPEN_MARGINRANKINGLOSS_REDUCTION_MEAN;
        divisor        = static_cast<float>(miopen::deref(input1Desc).GetElementSize());
    }
    else
    {
        return miopenStatusInvalidValue;
    }

    margin     = inflags.GetValueDouble("margin");
    is_forward = static_cast<bool>(inflags.GetValueInt("forw"));

    if(is_forward)
    {
        SetTensorNd(outputDesc, dims, data_type);
    }
    else
    {
        SetTensorNd(outGradDesc, dims, data_type);
        SetTensorNd(in1GradDesc, dims, data_type);
        SetTensorNd(in2GradDesc, dims, data_type);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MarginRankingLossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "MarginRankingLoss direction (Default=1)", "int");
    inflags.AddInputFlag(
        "dims",
        'D',
        "16,3,64,64,2",
        "The dimensional lengths of the input tensor: N,C,H,W,D (Default=16,3,64,64,2)",
        "string");
    inflags.AddInputFlag(
        "reduction",
        'R',
        "none",
        "Specifies the reduction to apply to the output ('none'|'mean'|'sum') (Default=none)",
        "string");
    inflags.AddInputFlag("margin", 'M', "0", "Margin value (Default=0)", "string");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MarginRankingLossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t element_size = miopen::deref(input1Desc).GetElementSize();

    uint32_t ctx = 0;

    input1_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    input2_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    target_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    outGrad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    in1Grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));
    in2Grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, element_size, sizeof(Tgpu)));

    input1  = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    input2  = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    target  = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    output  = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    outGrad = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    in1Grad = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));
    in2Grad = std::vector<Tgpu>(element_size, static_cast<Tgpu>(0));

    out_host     = std::vector<Tref>(element_size, static_cast<Tref>(0));
    in1Grad_host = std::vector<Tref>(element_size, static_cast<Tref>(0));
    in2Grad_host = std::vector<Tref>(element_size, static_cast<Tref>(0));

    for(int i = 0; i < element_size; i++)
    {
        input1[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        input2[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        target[i] = static_cast<Tgpu>(prng::gen_A_to_B<int>(0, 2) * 2 - 1); // 1 or -1
    }
    if(input1_dev->ToGPU(GetStream(), input1.data()) != 0)
        std::cerr << "Error copying (input1) to GPU, size: " << input1_dev->GetSize() << std::endl;
    if(input2_dev->ToGPU(GetStream(), input2.data()) != 0)
        std::cerr << "Error copying (input2) to GPU, size: " << input2_dev->GetSize() << std::endl;
    if(target_dev->ToGPU(GetStream(), target.data()) != 0)
        std::cerr << "Error copying (target) to GPU, size: " << target_dev->GetSize() << std::endl;

    if(is_forward)
    {
        fill(output.begin(), output.end(), static_cast<Tgpu>(0));
        if(output_dev->ToGPU(GetStream(), output.data()) != 0)
            std::cerr << "Error copying (out) to GPU, size: " << output_dev->GetSize() << std::endl;
    }
    else
    {
        for(int i = 0; i < element_size; i++)
        {
            outGrad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }
        fill(in1Grad.begin(), in1Grad.end(), static_cast<Tgpu>(0));
        fill(in2Grad.begin(), in2Grad.end(), static_cast<Tgpu>(0));
        if(outGrad_dev->ToGPU(GetStream(), outGrad.data()) != 0)
            std::cerr << "Error copying (outGrad) to GPU, size: " << outGrad_dev->GetSize()
                      << std::endl;
        if(in1Grad_dev->ToGPU(GetStream(), in1Grad.data()) != 0)
            std::cerr << "Error copying (in1Grad) to GPU, size: " << in1Grad_dev->GetSize()
                      << std::endl;
        if(in2Grad_dev->ToGPU(GetStream(), in2Grad.data()) != 0)
            std::cerr << "Error copying (in2Grad) to GPU, size: " << in2Grad_dev->GetSize()
                      << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MarginRankingLossDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenMarginRankingLossForward(GetHandle(),
                                       input1Desc,
                                       input1_dev->GetMem(),
                                       input2Desc,
                                       input2_dev->GetMem(),
                                       targetDesc,
                                       target_dev->GetMem(),
                                       outputDesc,
                                       output_dev->GetMem(),
                                       margin,
                                       reduction_mode);
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
            printf("Wall-clock Time Forward MarginRankingLoss Elapsed: %f ms\n",
                   t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Forward MarginRankingLoss Elapsed: %f ms\n", kernel_average_time);
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (output_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MarginRankingLossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenMarginRankingLossBackward(GetHandle(),
                                        input1Desc,
                                        input1_dev->GetMem(),
                                        input2Desc,
                                        input2_dev->GetMem(),
                                        targetDesc,
                                        target_dev->GetMem(),
                                        outGradDesc,
                                        outGrad_dev->GetMem(),
                                        in1GradDesc,
                                        in1Grad_dev->GetMem(),
                                        in2GradDesc,
                                        in2Grad_dev->GetMem(),
                                        margin,
                                        reduction_mode);
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
            printf("Wall-clock Time Backward MarginRankingLoss Elapsed: %f ms\n",
                   t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Backward MarginRankingLoss Elapsed: %f ms\n", kernel_average_time);
    }

    in1Grad_dev->FromGPU(GetStream(), in1Grad.data());
    in2Grad_dev->FromGPU(GetStream(), in2Grad.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MarginRankingLossDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(reduction_mode != MIOPEN_MARGINRANKINGLOSS_REDUCTION_NONE)
    {
        mloMarginRankingLossReducedForwardRunHost<Tgpu, Tref>(input1Desc,
                                                              input1.data(),
                                                              input2Desc,
                                                              input2.data(),
                                                              targetDesc,
                                                              target.data(),
                                                              outputDesc,
                                                              out_host.data(),
                                                              margin,
                                                              divisor);
    }
    else
    {
        mloMarginRankingLossUnreducedForwardRunHost<Tgpu, Tref>(input1Desc,
                                                                input1.data(),
                                                                input2Desc,
                                                                input2.data(),
                                                                targetDesc,
                                                                target.data(),
                                                                outputDesc,
                                                                out_host.data(),
                                                                margin);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MarginRankingLossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    if(reduction_mode != MIOPEN_MARGINRANKINGLOSS_REDUCTION_NONE)
    {
        mloMarginRankingLossReducedBackwardRunHost<Tgpu, Tref>(input1Desc,
                                                               input1.data(),
                                                               input2Desc,
                                                               input2.data(),
                                                               targetDesc,
                                                               target.data(),
                                                               outGradDesc,
                                                               outGrad.data(),
                                                               in1GradDesc,
                                                               in1Grad_host.data(),
                                                               in2GradDesc,
                                                               in2Grad_host.data(),
                                                               margin,
                                                               divisor);
    }
    else
    {
        mloMarginRankingLossUnreducedBackwardRunHost<Tgpu, Tref>(input1Desc,
                                                                 input1.data(),
                                                                 input2Desc,
                                                                 input2.data(),
                                                                 targetDesc,
                                                                 target.data(),
                                                                 outGradDesc,
                                                                 outGrad.data(),
                                                                 in1GradDesc,
                                                                 in1Grad_host.data(),
                                                                 in2GradDesc,
                                                                 in2Grad_host.data(),
                                                                 margin);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref MarginRankingLossDriver<Tgpu, Tref>::GetTolerance()
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
int MarginRankingLossDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(out_host, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward MarginRankingLoss FAILED: error=" << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        printf("Forward MarginRankingLoss Verifies on CPU and GPU (err=%f)\n", error);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int MarginRankingLossDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto in1Grad_error   = miopen::rms_range(in1Grad_host, in1Grad);
    auto in2Grad_error   = miopen::rms_range(in2Grad_host, in2Grad);

    if(!std::isfinite(in1Grad_error) || in1Grad_error > tolerance)
    {
        std::cout << "Backward MarginRankingLoss (in1Grad) FAILED: " << in1Grad_error << std::endl;
        return EC_VerifyFwd;
    }
    else if(!std::isfinite(in2Grad_error) || in2Grad_error > tolerance)
    {
        std::cout << "Backward MarginRankingLoss (in2Grad) FAILED: " << in2Grad_error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        printf("Backward MarginRankingLoss Verifies on CPU and GPU (in1Grad_error=%f, "
               "in2Grad_error=%f)\n",
               in1Grad_error,
               in2Grad_error);
    }
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_MARGINRANKINGLOSS_DRIVER_HPP
