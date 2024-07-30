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
#ifndef GUARD_MIOPEN_INDEXSELECT_DRIVER_HPP
#define GUARD_MIOPEN_INDEXSELECT_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloIndexSelectForwardRunHost(miopenTensorDescriptor_t inputDesc,
                                     miopenTensorDescriptor_t indicesDesc,
                                     miopenTensorDescriptor_t outputDesc,
                                     Tgpu* input,
                                     int* indices,
                                     Tgpu* output,
                                     size_t dim,
                                     size_t numOfIndices,
                                     Tcheck* outputhost)
{
    int32_t ret = 0;

    auto input_dims  = miopen::deref(inputDesc).GetLengths();
    auto output_dims = miopen::deref(outputDesc).GetLengths();

    auto input_strides  = miopen::deref(inputDesc).GetStrides();
    auto output_strides = miopen::deref(outputDesc).GetStrides();

    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1LL, std::multiplies<int64_t>());

    size_t n[4], n012, n01;

    for(size_t i = 0; i < output_numel; i++)
    {
        n[3] = i % output_dims[3];
        n012 = i / output_dims[3];
        n[2] = n012 % output_dims[2];
        n01  = n012 / output_dims[2];
        n[1] = n01 % output_dims[1];
        n[0] = n01 / output_dims[1];

        size_t output_idx = n[0] * output_strides[0] + n[1] * output_strides[1] +
                            n[2] * output_strides[2] + n[3] * output_strides[3];

        n[dim] = indices[n[dim]];

        size_t input_idx = n[0] * input_strides[0] + n[1] * input_strides[1] +
                           n[2] * input_strides[2] + n[3] * input_strides[3];

        outputhost[output_idx] = input[input_idx];
    }
    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloIndexSelectBackwardRunHost(miopenTensorDescriptor_t outputGradDesc,
                                      miopenTensorDescriptor_t inputGradDesc,
                                      Tgpu* outputGrad,
                                      int* indices,
                                      Tgpu* inputGrad,
                                      size_t dim,
                                      size_t numOfIndices,
                                      Tcheck* inputGradhost)
{
    int32_t ret = 0;

    auto inputGrad_dims     = miopen::deref(inputGradDesc).GetLengths();
    auto outputGrad_dims    = miopen::deref(outputGradDesc).GetLengths();
    auto inputGrad_strides  = miopen::deref(inputGradDesc).GetStrides();
    auto outputGrad_strides = miopen::deref(outputGradDesc).GetStrides();

    size_t oK = outputGrad_dims[dim];

    size_t st = 1;
    for(size_t i = dim + 1; i < inputGrad_dims.size(); i++)
    {
        st *= inputGrad_dims[i];
    }
    size_t N = 1;
    for(size_t i = 0; i < inputGrad_dims.size(); i++)
    {
        if(i != dim)
        {
            N *= inputGrad_dims[i];
        }
    }

    for(size_t i = 0; i < N; i++)
    {
        size_t output_grad_base_idx = (i / st) * st * oK + i % st;
        size_t n[4], n012, n01;
        n[3] = output_grad_base_idx % outputGrad_dims[3];
        n012 = output_grad_base_idx / outputGrad_dims[3];
        n[2] = n012 % outputGrad_dims[2];
        n01  = n012 / outputGrad_dims[2];
        n[1] = n01 % outputGrad_dims[1];
        n[0] = n01 / outputGrad_dims[1];

        for(int j = 0; j < oK; ++j)
        {
            n[dim]                 = j;
            size_t idx             = indices[j];
            size_t output_grad_idx = n[0] * outputGrad_strides[0] + n[1] * outputGrad_strides[1] +
                                     n[2] * outputGrad_strides[2] + n[3] * outputGrad_strides[3];
            n[dim]                = idx;
            size_t input_grad_idx = n[0] * inputGrad_strides[0] + n[1] * inputGrad_strides[1] +
                                    n[2] * inputGrad_strides[2] + n[3] * inputGrad_strides[3];

            Tcheck input_grad_v           = inputGradhost[input_grad_idx];
            Tcheck output_grad_v          = outputGrad[output_grad_idx];
            inputGradhost[input_grad_idx] = input_grad_v + output_grad_v;
        }
    }
    return ret;
}

template <typename Tgpu, typename Tref>
class IndexSelectDriver : public Driver
{
public:
    IndexSelectDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&indicesDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&outputGradDesc);

        data_type = miopen_type<Tgpu>{};
    }
    std::vector<int> ComputeInputStrides();
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
    int RunBackwardCPU();

    Tref GetTolerance();

    int VerifyForward() override;
    int VerifyBackward() override;
    ~IndexSelectDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(indicesDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t indicesDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t outputGradDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> indices_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> inputGrad_dev;
    std::unique_ptr<GPUMem> outputGrad_dev;

    std::vector<Tgpu> input;
    std::vector<int> indices;
    std::vector<Tgpu> output;
    std::vector<Tgpu> inputGrad;
    std::vector<Tgpu> outputGrad;

    std::vector<Tref> outputhost;
    std::vector<Tref> inputGradhost;

    size_t dim;
    size_t numOfIndices;

    bool isContiguous;
};

template <typename Tgpu, typename Tref>
std::vector<int> IndexSelectDriver<Tgpu, Tref>::ComputeInputStrides()
{
    std::vector<int> in_lens = GetInputTensorLengthsFromCmdLine();
    if(!isContiguous)
        std::swap(in_lens.front(), in_lens.back());
    std::vector<int> strides(in_lens.size());
    strides.back() = 1;
    for(int i = in_lens.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * in_lens[i + 1];
    if(!isContiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len      = GetInputTensorLengthsFromCmdLine();
    dim                          = inflags.GetValueInt("DimToReduce");
    std::vector<int> out_len     = GetOutputTensorLengthsFromCmdLine();
    numOfIndices                 = inflags.GetValueInt("NumOfIndices");
    std::vector<int> indices_len = std::vector<int>({numOfIndices});
    auto inputStrides            = ComputeInputStrides();

    SetTensorNd(inputDesc, in_len, inputStrides, data_type);
    SetTensorNd(indicesDesc, indices_len, miopenInt32);
    SetTensorNd(outputDesc, out_len, data_type);
    SetTensorNd(inputGradDesc, in_len, data_type);
    SetTensorNd(outputGradDesc, out_len, data_type);

    return 0;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("batchsize", 'n', "32", "Mini-batch size (Default=32)", "int");

    inflags.AddInputFlag("in_channels", 'c', "32", "Number of Input Channels (Default=32)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");

    inflags.AddInputFlag("contiguous", 'C', "1", "Tensor is contiguous or not", "int");

    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    inflags.AddInputFlag(
        "DimToReduce", 'R', "1", "The indice of the dimensions to be reduced(Default=1)", "int");

    inflags.AddInputFlag("NumOfIndices",
                         'I',
                         "4",
                         "The number of indice of the dimension to be reduced(Default=1)",
                         "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Outer (Default=1)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> IndexSelectDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");

    if((in_n != 0) && (in_c != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_h, in_w});
    }
    else
    {
        std::cerr << "Error Input Tensor Lengths\n" << std::endl;
        return std::vector<int>({0});
    }
}

template <typename Tgpu, typename Tref>
std::vector<int> IndexSelectDriver<Tgpu, Tref>::GetOutputTensorLengthsFromCmdLine()
{
    int in_n     = inflags.GetValueInt("batchsize");
    int in_c     = inflags.GetValueInt("in_channels");
    int in_h     = inflags.GetValueInt("in_h");
    int in_w     = inflags.GetValueInt("in_w");
    dim          = inflags.GetValueInt("DimToReduce");
    numOfIndices = inflags.GetValueInt("NumOfIndices");
    if(inflags.GetValueInt("contiguous") == 1)
    {
        isContiguous = true;
    }
    else
    {
        isContiguous = false;
    }

    if(dim == 0)
        in_n = numOfIndices;
    else if(dim == 1)
        in_c = numOfIndices;
    else if(dim == 2)
        in_h = numOfIndices;
    else if(dim == 3)
        in_w = numOfIndices;

    if((in_n != 0) && (in_c != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_h, in_w});
    }
    else
    {
        std::cerr << "Error Input Tensor Lengths\n" << std::endl;
        return std::vector<int>({0});
    }
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz  = GetTensorSize(inputDesc);
    size_t out_sz = GetTensorSize(outputDesc);

    std::vector<int> in_len  = GetInputTensorLengthsFromCmdLine();
    std::vector<int> out_len = GetOutputTensorLengthsFromCmdLine();

    uint32_t ctx = 0;

    input_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    indices_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, numOfIndices, sizeof(int)));
    output_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    inputGrad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    outputGrad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    input      = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    indices    = std::vector<int>(numOfIndices, static_cast<int>(0));
    output     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    inputGrad  = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    outputGrad = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

    inputGradhost = std::vector<Tref>(in_sz, static_cast<Tref>(0));
    outputhost    = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    for(size_t i = 0; i < in_sz; i++)
    {
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    for(size_t i = 0; i < out_sz; i++)
    {
        outputGrad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    std::vector<int> numbers(in_len[dim]);
    std::iota(numbers.begin(), numbers.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(numbers.begin(), numbers.end(), gen);

    for(size_t i = 0; i < numOfIndices; i++)
    {
        indices[i] = numbers[i];
    }

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize() << std::endl;

    if(output_dev->ToGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (output) to GPU, size: " << output_dev->GetSize() << std::endl;

    if(indices_dev->ToGPU(GetStream(), indices.data()) != 0)
        std::cerr << "Error copying (indices) to GPU, size: " << indices_dev->GetSize()
                  << std::endl;

    if(inputGrad_dev->ToGPU(GetStream(), inputGrad.data()) != 0)
        std::cerr << "Error copying (inputGrad) to GPU, size: " << inputGrad_dev->GetSize()
                  << std::endl;

    if(outputGrad_dev->ToGPU(GetStream(), outputGrad.data()) != 0)
        std::cerr << "Error copying (outputGrad) to GPU, size: " << outputGrad_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME
    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenIndexSelectForward(GetHandle(),
                                 inputDesc,
                                 input_dev->GetMem(),
                                 indicesDesc,
                                 indices_dev->GetMem(),
                                 outputDesc,
                                 output_dev->GetMem(),
                                 dim);
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
            std::cout << "Wall-clock Time Forward IndexSelect Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward IndexSelect Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (output_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloIndexSelectForwardRunHost<Tgpu, Tref>(inputDesc,
                                             indicesDesc,
                                             outputDesc,
                                             input.data(),
                                             indices.data(),
                                             output.data(),
                                             dim,
                                             numOfIndices,
                                             outputhost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::RunBackwardGPU()
{
    size_t in_sz = GetTensorSize(inputDesc);

    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenIndexSelectBackward(GetHandle(),
                                  inputGradDesc,
                                  inputGrad_dev->GetMem(),
                                  indicesDesc,
                                  indices_dev->GetMem(),
                                  outputGradDesc,
                                  outputGrad_dev->GetMem(),
                                  dim);
        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;

        if(i != inflags.GetValueInt("iter") - 1)
        {
            inputGrad = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
            if(inputGrad_dev->ToGPU(GetStream(), inputGrad.data()) != 0)
                std::cerr << "Error copying (inputGrad) to GPU, size: " << inputGrad_dev->GetSize()
                          << std::endl;
        }
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Backward IndexSelect Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward IndexSelect Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(inputGrad_dev->FromGPU(GetStream(), inputGrad.data()) != 0)
        std::cerr << "Error copying (inputGrad_dev) from GPU, size: " << inputGrad_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloIndexSelectBackwardRunHost<Tgpu, Tref>(outputGradDesc,
                                              inputGradDesc,
                                              outputGrad.data(),
                                              indices.data(),
                                              inputGrad.data(),
                                              dim,
                                              numOfIndices,
                                              inputGradhost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref IndexSelectDriver<Tgpu, Tref>::GetTolerance()
{
    auto tolerance = std::is_same<Tgpu, float>::value ? 1.5e-6 : 8.2e-3;

    if(std::is_same<Tgpu, bfloat16>::value)
        tolerance *= 8.0;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();

    const Tref tolerance = GetTolerance();

    auto error = miopen::rms_range(outputhost, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward IndexSelect FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward IndexSelect Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int IndexSelectDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();

    const Tref tolerance = GetTolerance();

    auto error = miopen::rms_range(inputGradhost, inputGrad);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward IndexSelect FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backward IndexSelect Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_INDEXSELECT_DRIVER_HPP
