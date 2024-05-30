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

#pragma once
#include "InputFlags.hpp"
#include "driver.hpp"
#include "miopen/errors.hpp"
#include "miopen/sigmoidfocalloss/utils.hpp"
#include "miopen/miopen.h"
#include "tensor_driver.hpp"
#include "tensor_view_5d.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

// #define DEBUGGING

template <typename TIO>
void mloSigmoidFocalLossUnreducedFwdRunHost(TIO* input,
                                            miopenTensorDescriptor_t inputDesc,
                                            TIO* target,
                                            miopenTensorDescriptor_t targetDesc,
                                            TIO* outputHost,
                                            float alpha = 0.25,
                                            float gamma = 2)
{
    tensor_view_5d_t input_tv  = get_inner_expanded_tv(miopen::deref(inputDesc));
    tensor_view_5d_t target_tv = get_inner_expanded_tv(miopen::deref(targetDesc));
    size_t inputSize           = miopen::deref(inputDesc).GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < inputSize; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, input_tv);

        float i = static_cast<float>(TV_5D_AT(input, n[0], n[1], n[2], n[3], n[4]));
        float t = static_cast<float>(TV_5D_AT(target, n[0], n[1], n[2], n[3], n[4]));

        float sig    = 1 / (1 + exp(-i));
        float ceLoss = -(t * log(sig) + (1 - t) * log(1 - sig));
        float sigT   = sig * t + (1 - sig) * (1 - t);
        float loss   = ceLoss * pow(1 - sigT, gamma);

        if(alpha >= 0)
        {
            float alphaT = alpha * t + (1 - alpha) * (1 - t);
            loss         = alphaT * loss;
        }

        outputHost[idx] = static_cast<TIO>(loss);
    }
}

template <typename TIO>
class SigmoidFocalLossDriver : public Driver
{
public:
    SigmoidFocalLossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&doutputDesc);
        miopenCreateTensorDescriptor(&dinputDesc);

        data_type = miopen_type<TIO>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetTensorStride(std::vector<int> dim);

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    TIO GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~SigmoidFocalLossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(doutputDesc);
        miopenDestroyTensorDescriptor(dinputDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t doutputDesc;
    miopenTensorDescriptor_t dinputDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> doutput_dev;
    std::unique_ptr<GPUMem> dinput_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<TIO> input;
    std::vector<TIO> target;
    std::vector<TIO> output;
    std::vector<TIO> outputHost;
    std::vector<TIO> doutput;
    std::vector<TIO> dinput;
    std::vector<TIO> dinputHost;
    std::vector<TIO> workspace;

    float alpha;
    float gamma;
    float divisor;
    bool isContiguous;
    miopenLossReductionMode_t reduction;

    size_t workSpaceSizeInBytes;
};

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::GetandSetData()
{
    std::vector<int> inDim = GetInputTensorLengthsFromCmdLine();
    alpha                  = inflags.GetValueDouble("alpha");
    gamma                  = inflags.GetValueDouble("gamma");
    isContiguous           = inflags.GetValueInt("is-contiguous") == 1 ? true : false;
    reduction = static_cast<miopenLossReductionMode_t>(inflags.GetValueInt("reduction"));

    std::vector<int> inStride = GetTensorStride(inDim);
    if(!isContiguous)
    {
        std::swap(inDim.front(), inDim.back());
    }

    SetTensorNd(inputDesc, inDim, inStride, data_type);
    SetTensorNd(targetDesc, inDim, inStride, data_type);
    SetTensorNd(doutputDesc, inDim, data_type);
    SetTensorNd(dinputDesc, inDim, data_type);

    if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    {
        SetTensorNd(outputDesc, inDim, data_type);
    }
    else
    {
        std::vector<int> outDim(1);
        outDim[0] = 1;
        SetTensorNd(outputDesc, outDim, data_type);
        divisor = 1;
        if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
        {
            divisor = miopen::deref(inputDesc).GetElementSize();
        }
    }

    return 0;
}

template <typename TIO>
std::vector<int> SigmoidFocalLossDriver<TIO>::GetTensorStride(std::vector<int> dim)
{
    std::vector<int> strides(dim.size(), 1);
    for(int i = dim.size() - 2; i >= 0; --i)
    {
        strides[i] = dim[i + 1] * strides[i + 1];
    }

    if(!isContiguous)
    {
        std::swap(strides.front(), strides.back());
    }

    return strides;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward (Default=1)", "int");
    inflags.AddInputFlag("DimLengths",
                         'D',
                         "256,4,1,1,8723",
                         "The dimensional lengths of the input tensor",
                         "string");
    inflags.AddInputFlag("is-contiguous", 'c', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag(
        "reduction", 'R', "0", "reduction mode: 0(default) - unreduced, 1 - sum, 2 -mean", "int");
    inflags.AddInputFlag("alpha", 'A', "0.25", "Alpha (Default=0.25)", "float");
    inflags.AddInputFlag("gamma", 'G', "2", "Gamma (Default=2)", "float");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename TIO>
std::vector<int> SigmoidFocalLossDriver<TIO>::GetInputTensorLengthsFromCmdLine()
{
    std::string lengthsStr = inflags.GetValueStr("DimLengths");

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

    return (lengths);
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::AllocateBuffersAndCopy()
{
    size_t in_sz     = miopen::deref(inputDesc).GetElementSize();
    size_t target_sz = miopen::deref(targetDesc).GetElementSize();
    size_t out_sz    = miopen::deref(outputDesc).GetElementSize();
    size_t dO_sz     = miopen::deref(doutputDesc).GetElementSize();
    size_t dI_sz     = miopen::deref(dinputDesc).GetElementSize();

    uint32_t ctx = 0;

    input_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(TIO)));
    target_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(TIO)));
    output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(TIO)));
    doutput_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, dO_sz, sizeof(TIO)));
    dinput_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, dI_sz, sizeof(TIO)));

    miopenGetSigmoidFocalLossForwardWorkspaceSize(
        handle, inputDesc, targetDesc, outputDesc, reduction, &workSpaceSizeInBytes);
    workspace_dev =
        std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceSizeInBytes / sizeof(TIO), sizeof(TIO)));

    input      = std::vector<TIO>(in_sz, static_cast<TIO>(0));
    target     = std::vector<TIO>(target_sz, static_cast<TIO>(0));
    output     = std::vector<TIO>(out_sz, static_cast<TIO>(0));
    outputHost = std::vector<TIO>(out_sz, static_cast<TIO>(0));
    doutput    = std::vector<TIO>(dO_sz, static_cast<TIO>(0));
    dinput     = std::vector<TIO>(dI_sz, static_cast<TIO>(0));
    dinputHost = std::vector<TIO>(dI_sz, static_cast<TIO>(0));
    workspace  = std::vector<TIO>(workSpaceSizeInBytes / sizeof(TIO), static_cast<TIO>(0));

    for(int i = 0; i < in_sz; i++)
    {
        input[i]  = prng::gen_A_to_B<TIO>(static_cast<TIO>(-2), static_cast<TIO>(2));
        target[i] = prng::gen_A_to_B<TIO>(static_cast<TIO>(-2), static_cast<TIO>(2));
    }

#ifdef DEBUGGING
    float input_arr[12] = {0.4525,
                           -1.1261,
                           -2.2234,
                           -0.7377,
                           0.2541,
                           -2.2175,
                           2.5682,
                           -2.2930,
                           -0.5593,
                           1.9189,
                           -2.5782,
                           -0.4460};

    float target_arr[12] = {1.0245,
                            0.4950,
                            1.7240,
                            -1.1395,
                            2.1006,
                            -0.2978,
                            -1.1221,
                            4.5647,
                            2.3071,
                            -0.6812,
                            -3.5125,
                            -0.4301};

    for(int i = 0; i < in_sz; ++i)
    {
        input[i]  = input_arr[i];
        target[i] = target_arr[i];
    }
#endif

    for(int i = 0; i < dO_sz; ++i)
    {
        doutput[i] = prng::gen_A_to_B<TIO>(static_cast<TIO>(-2), static_cast<TIO>(2));
    }

    fill(output.begin(), output.end(), static_cast<TIO>(0));
    fill(dinput.begin(), dinput.end(), static_cast<TIO>(0));

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << input_dev->GetSize() << std::endl;

    if(target_dev->ToGPU(GetStream(), target.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << target_dev->GetSize() << std::endl;

    if(output_dev->ToGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << output_dev->GetSize() << std::endl;

    if(doutput_dev->ToGPU(GetStream(), doutput.data()) != 0)
        std::cerr << "Error copying (dO) to GPU, size: " << doutput_dev->GetSize() << std::endl;

    if(dinput_dev->ToGPU(GetStream(), dinput.data()) != 0)
        std::cerr << "Error copying (dI) to GPU, size: " << dinput_dev->GetSize() << std::endl;

    if(workspace_dev->ToGPU(GetStream(), workspace.data()) != 0)
        std::cerr << "Error copying (dI) to GPU, size: " << workspace_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenSigmoidFocalLossForward(GetHandle(),
                                      workspace_dev->GetMem(),
                                      workSpaceSizeInBytes,
                                      inputDesc,
                                      input_dev->GetMem(),
                                      targetDesc,
                                      target_dev->GetMem(),
                                      outputDesc,
                                      output_dev->GetMem(),
                                      alpha,
                                      gamma,
                                      reduction);
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
            std::cout << "Wall-clock Time Sigmoid Focal Loss Unreduced Fwd Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Sigmoid Focal Loss Unreduced Fwd Elapsed: "
                  << kernel_average_time << " ms" << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::RunForwardCPU()
{
    if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    {
        mloSigmoidFocalLossUnreducedFwdRunHost<TIO>(
            input.data(), inputDesc, target.data(), targetDesc, outputHost.data(), alpha, gamma);
    }

    return miopenStatusSuccess;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::RunBackwardGPU()
{
    // float kernel_total_time = 0;
    // float kernel_first_time = 0;

    // Timer t;
    // START_TIME

    // for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    // {

    //     miopenSigmoidFocalLossBackward(GetHandle(),
    //                                    inputDesc,
    //                                    input_dev->GetMem(),
    //                                    targetDesc,
    //                                    target_dev->GetMem(),
    //                                    doutputDesc,
    //                                    doutput_dev->GetMem(),
    //                                    dinputDesc,
    //                                    dinput_dev->GetMem(),
    //                                    alpha,
    //                                    gamma,
    //                                    reduction);

    //     float time = 0.0;
    //     miopenGetKernelTime(GetHandle(), &time);
    //     kernel_total_time += time;
    //     if(i == 0)
    //         kernel_first_time = time;
    // }

    // if(inflags.GetValueInt("time") == 1)
    // {
    //     STOP_TIME
    //     int iter = inflags.GetValueInt("iter");
    //     if(WALL_CLOCK)
    //         std::cout << "Wall-clock Time Sigmoid Focal Loss Unreduced Bwd Elapsed: "
    //                   << t.gettime_ms() / iter << " ms" << std::endl;

    //     float kernel_average_time =
    //         iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
    //     std::cout << "GPU Kernel Time Sigmoid Focal Loss Unreduced Bwd Elapsed: "
    //               << kernel_average_time << " ms" << std::endl;
    // }

    // if(dinput_dev->FromGPU(GetStream(), dinput.data()) != 0)
    //     std::cerr << "Error copying (dI_dev) from GPU, size: " << dinput_dev->GetSize()
    //               << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::RunBackwardCPU()
{
    // if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
    // {

    //     mloSigmoidFocalLossUnreducedBwdRunHost<TIO>(input.data(),
    //                                                 inputDesc,
    //                                                 target.data(),
    //                                                 targetDesc,
    //                                                 doutput.data(),
    //                                                 doutputDesc,
    //                                                 dinputHost.data(),
    //                                                 alpha,
    //                                                 gamma);
    // }
    // else
    // {
    //     mloSigmoidFocalLossBwdRunHost<TIO>(input.data(),
    //                                        inputDesc,
    //                                        target.data(),
    //                                        targetDesc,
    //                                        doutput.data(),
    //                                        doutputDesc,
    //                                        dinputHost.data(),
    //                                        margin,
    //                                        divisor);
    // }

    return miopenStatusSuccess;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::VerifyForward()
{
    RunForwardCPU();
#ifdef DEBUGGING
    for(int i = 0; i < miopen::deref(inputDesc).GetElementSize(); ++i)
    {
        std::cout << output.data()[i] << " " << outputHost.data()[i] << std::endl;
    }
#endif
    double tolerance = std::numeric_limits<TIO>::epsilon() * 10;
    auto error       = miopen::rms_range(outputHost, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward " << reduction << " Sigmoid Focal Loss FAILED: " << error << " > "
                  << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward " << reduction << " Sigmoid Focal Loss Verifies OK on CPU reference ("
                  << error << "< " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename TIO>
int SigmoidFocalLossDriver<TIO>::VerifyBackward()
{
    RunBackwardCPU();
    double tolerance = std::numeric_limits<TIO>::epsilon() * 10;
    auto error       = miopen::rms_range(dinputHost, dinput);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward " << reduction << " Sigmoid Focal Loss FAILED: " << error << " > "
                  << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backward " << reduction
                  << " Sigmoid Focal Loss Verifies OK on CPU reference (" << error << "< "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
