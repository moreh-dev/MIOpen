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
#include <iostream>
#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include "tensor_driver.hpp"
#include <rocfft/rocfft.h>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>
#include <cmath>

#define ROCFFT_CHECK(status)                                  \
    if(status != rocfft_status_success)                       \
    {                                                         \
        std::cerr << "rocFFT error: " << status << std::endl; \
        exit(status);                                         \
    }

#define HIP_CHECK(status)                                                     \
    if(status != hipSuccess)                                                  \
    {                                                                         \
        std::cerr << "HIP error: " << hipGetErrorString(status) << std::endl; \
        exit(status);                                                         \
    }

template <typename TI, typename TO, typename Tcheck>
class RfftDriver : public Driver
{
public:
    RfftDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&doutputDesc);
        miopenCreateTensorDescriptor(&dinputDesc);
        miopenCreateTensorDescriptor(&dtargetDesc);

        data_type = miopen_type<TI>{};
    }

    std::vector<size_t> ComputeStrides(std::vector<int> input);
    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tcheck GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~RfftDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(doutputDesc);
        miopenDestroyTensorDescriptor(dinputDesc);
        miopenDestroyTensorDescriptor(dtargetDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t doutputDesc;
    miopenTensorDescriptor_t dinputDesc;
    miopenTensorDescriptor_t dtargetDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> doutput_dev;
    std::unique_ptr<GPUMem> dinput_dev;
    std::unique_ptr<GPUMem> dtarget_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<TI> input;
    std::vector<TI> target;
    std::vector<TO> output;
    std::vector<int> outDims;
    std::vector<Tcheck> outputHost;
    std::vector<TI> doutput;
    std::vector<TI> dinput;
    std::vector<Tcheck> dinputHost;
    std::vector<TI> dtarget;
    std::vector<Tcheck> dtargetHost;
    std::vector<TI> workspace;
    std::vector<Tcheck> workspaceHost;

    bool isContiguous = true;
    int normalize;
    int dim;
};

template <typename TI, typename TO, typename Tcheck>
int RfftDriver<TI, TO, Tcheck>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    dim       = inflags.GetValueInt("dim");
    normalize = inflags.GetValueInt("norm");
    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename TI, typename TO, typename Tcheck>
int RfftDriver<TI, TO, Tcheck>::GetandSetData()
{
    auto inDims                  = inflags.GetValueTensor("dim-lengths").lengths;
    std::vector<size_t> inStride = ComputeStrides(inDims);
    if(dim < 0)
    {
        dim += inDims.size();
    }

    // TODO: pass strides to create non-cont tensor, but strides require vector<int>
    SetTensorNd(inputDesc, inDims, data_type);
    SetTensorNd(doutputDesc, inDims, data_type);
    SetTensorNd(dinputDesc, inDims, data_type);

    outDims      = inDims;
    outDims[dim] = outDims[dim] / 2 + 1;

    return 0;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename TI, typename TO, typename Tcheck>
std::vector<size_t> RfftDriver<TI, TO, Tcheck>::ComputeStrides(std::vector<int> inputDim)
{
    if(!isContiguous)
        std::swap(inputDim.front(), inputDim.back());
    std::vector<size_t> strides(inputDim.size());
    strides.back() = 1;
    for(int i = inputDim.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * inputDim[i + 1];
    if(!isContiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename TI, typename TO, typename Tcheck>
int RfftDriver<TI, TO, Tcheck>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward (Default=1)", "int");
    inflags.AddTensorFlag(
        "dim-lengths", 'D', "256x4x2", "The dimensional lengths of the input tensor");
    inflags.AddInputFlag("is-contiguous", 'c', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("dim", 'd', "-1", "dim (Default=-1)", "int");
    inflags.AddInputFlag("norm", 'n', "0", "Norm (Default=0 - backward)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename TI, typename TO, typename Tcheck>
int RfftDriver<TI, TO, Tcheck>::AllocateBuffersAndCopy()
{
    size_t in_sz     = miopen::deref(inputDesc).GetElementSize();
    size_t target_sz = miopen::deref(targetDesc).GetElementSize();
    size_t out_sz    = 1;
    for(auto i : outDims)
    {
        out_sz *= i;
    }
    size_t dO_sz = miopen::deref(doutputDesc).GetElementSize();
    size_t dI_sz = miopen::deref(dinputDesc).GetElementSize();
    size_t dT_sz = miopen::deref(dtargetDesc).GetElementSize();

    uint32_t ctx = 0;

    input_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(TI)));
    target_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(TI)));
    output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(TO)));
    doutput_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, dO_sz, sizeof(TI)));
    dinput_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, dI_sz, sizeof(TI)));
    dtarget_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, dT_sz, sizeof(TI)));

    input       = std::vector<TI>(in_sz, static_cast<TI>(0));
    target      = std::vector<TI>(target_sz, static_cast<TI>(0));
    output      = std::vector<TO>(out_sz, static_cast<TO>(0));
    outputHost  = std::vector<Tcheck>(out_sz, static_cast<Tcheck>(0));
    doutput     = std::vector<TI>(dO_sz, static_cast<TI>(0));
    dinput      = std::vector<TI>(dI_sz, static_cast<TI>(0));
    dinputHost  = std::vector<Tcheck>(dI_sz, static_cast<Tcheck>(0));
    dtarget     = std::vector<TI>(dT_sz, static_cast<TI>(0));
    dtargetHost = std::vector<Tcheck>(dT_sz, static_cast<Tcheck>(0));

    float randomBound = 2;
    // For half, the random bound is smaller to avoid half overflow
    for(int i = 0; i < in_sz; i++)
    {
        input[i] =
            prng::gen_A_to_B<TI>(static_cast<TI>(-randomBound), static_cast<TI>(randomBound));
        input[i] = (float)i / in_sz;
    }
    for(int i = 0; i < dO_sz; ++i)
    {
        doutput[i] =
            prng::gen_A_to_B<TI>(static_cast<TI>(-randomBound), static_cast<TI>(randomBound));
    }

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << input_dev->GetSize() << std::endl;

    if(output_dev->ToGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << output_dev->GetSize() << std::endl;

    if(doutput_dev->ToGPU(GetStream(), doutput.data()) != 0)
        std::cerr << "Error copying (dO) to GPU, size: " << doutput_dev->GetSize() << std::endl;

    if(dinput_dev->ToGPU(GetStream(), dinput.data()) != 0)
        std::cerr << "Error copying (dI) to GPU, size: " << dinput_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename TI, typename TO, typename Tcheck>
int RfftDriver<TI, TO, Tcheck>::RunForwardGPU()
{
    rocfft_setup();

    int totalDim = miopen::deref(inputDesc).GetNumDims();
    if(dim != 0 and dim != totalDim - 1)
    {
        return miopenStatusNotImplemented;
    }
    // auto inStrides = miopen::deref(inputDesc).GetStrides();
    auto selectedDimSize = miopen::deref(inputDesc).GetLengths()[dim];
    std::vector<size_t> inStrides;
    std::vector<size_t> outStrides;
    size_t inDistance;
    size_t outDistance;

    if(dim == totalDim - 1)
    {
        inStrides.push_back(1);
        outStrides.push_back(1);

        inDistance  = miopen::deref(inputDesc).GetLengths()[dim];
        outDistance = 1 + inDistance / 2;
    }
    else
    {
        inStrides.push_back(miopen::deref(inputDesc).GetStrides()[0]);
        outStrides.push_back(miopen::deref(inputDesc).GetStrides()[0]);

        inDistance  = 1;
        outDistance = 1;
    }

    auto inSz      = miopen::deref(inputDesc).GetElementSize();
    auto batchSize = inSz / selectedDimSize;

    // Description
    rocfft_plan_description desc = nullptr;
    rocfft_plan_description_create(&desc);
    float scaleFactor = 1.0;
    // Forward
    if(normalize == 1)
    {
        scaleFactor = 1.0 / selectedDimSize;
    }
    // Ortho
    else if(normalize == 2)
    {
        scaleFactor = 1.0 / sqrt(selectedDimSize);
    }
    rocfft_plan_description_set_scale_factor(desc, scaleFactor);
    rocfft_plan_description_set_data_layout(desc,
                                            rocfft_array_type_real,
                                            rocfft_array_type_hermitian_interleaved,
                                            0,
                                            0,
                                            inStrides.size(),
                                            inStrides.data(),
                                            inDistance,
                                            outStrides.size(),
                                            outStrides.data(),
                                            outDistance);

    // Create rocFFT plan
    rocfft_plan plan = nullptr;
    ROCFFT_CHECK(rocfft_plan_create(&plan,
                                    rocfft_placement_notinplace,
                                    rocfft_transform_type_real_forward,
                                    rocfft_precision_single,
                                    1,
                                    &selectedDimSize,
                                    batchSize,
                                    desc));

    // Check if the plan requires a work buffer
    size_t work_buf_size = 0;
    ROCFFT_CHECK(rocfft_plan_get_work_buffer_size(plan, &work_buf_size));
    void* work_buf             = nullptr;
    rocfft_execution_info info = nullptr;
    if(work_buf_size != 0u)
    {
        ROCFFT_CHECK(rocfft_execution_info_create(&info));
        HIP_CHECK(hipMalloc(&work_buf, work_buf_size));
        ROCFFT_CHECK(rocfft_execution_info_set_work_buffer(info, work_buf, work_buf_size));
    }

    void* input_dev_ptr  = input_dev->GetMem();
    void* output_dev_ptr = output_dev->GetMem();

    float totalTime = 0;

    for(int i = 0; i <= inflags.GetValueInt("iter"); i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        // Execute plan
        ROCFFT_CHECK(rocfft_execute(plan, &input_dev_ptr, &output_dev_ptr, info));
        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        if(i > 0)
        {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            totalTime += duration.count() / 1000.0; // Convert to milliseconds
        }
    }

    if(inflags.GetValueInt("time") == 1)
    {
        int iter                  = inflags.GetValueInt("iter");
        float kernel_average_time = totalTime / iter;
        std::cout << "GPU Kernel Time Rfft Fwd Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    // Clean up work buffer
    if(work_buf_size != 0u)
    {
        HIP_CHECK(hipFree(work_buf));
        ROCFFT_CHECK(rocfft_execution_info_destroy(info));
    }

    // Destroy plan
    ROCFFT_CHECK(rocfft_plan_destroy(plan));

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;

    // Print results
    // for(auto elem : output)
    // {
    //     std::cout << elem.x << "+" << elem.y << "i" << std::endl;
    // }

    ROCFFT_CHECK(rocfft_cleanup());
    return miopenStatusSuccess;
}

template <typename TI, typename TO, typename Tcheck>
int RfftDriver<TI, TO, Tcheck>::RunForwardCPU()
{
    return miopenStatusSuccess;
}

template <typename TI, typename TO, typename Tcheck>
int RfftDriver<TI, TO, Tcheck>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename TI, typename TO, typename Tcheck>
int RfftDriver<TI, TO, Tcheck>::RunBackwardCPU()
{
    return miopenStatusSuccess;
}

template <typename TI, typename TO, typename Tcheck>
Tcheck RfftDriver<TI, TO, Tcheck>::GetTolerance()
{
    // Computation error of fp16 is ~2^13 (=8192) bigger than
    // the one of fp32 because mantissa is shorter by 13 bits.
    auto tolerance = std::is_same<TI, float>::value ? 1.5e-6 : 8.2e-3;

    // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
    if(std::is_same<TI, bfloat16>::value)
        tolerance *= 8.0;
    return tolerance;
}

template <typename TI, typename TO, typename Tcheck>
int RfftDriver<TI, TO, Tcheck>::VerifyForward()
{
    return miopenStatusSuccess;
}

template <typename TI, typename TO, typename Tcheck>
int RfftDriver<TI, TO, Tcheck>::VerifyBackward()
{
    return miopenStatusSuccess;
}
