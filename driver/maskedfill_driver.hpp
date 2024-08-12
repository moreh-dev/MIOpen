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

#ifndef GUARD_MIOPEN_MASKEDFILL_DRIVER_HPP
#define GUARD_MIOPEN_MASKEDFILL_DRIVER_HPP

#include <miopen/miopen.h>

#include "driver.hpp"
#include "../test/tensor_holder.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "../test/verify.hpp"

#include "mloMaskedFillHost.hpp"

template <typename Tgpu, typename Tref = Tgpu>
class MaskedFillDriver : public Driver
{
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc, outputDesc, maskDesc;
    std ::vector<Tgpu> input, output;
    std ::vector<int8_t> mask;
    std ::unique_ptr<GPUMem> input_dev, output_dev, mask_dev;

    std ::vector<Tref> hostoutput;

    Tgpu value;

public:
    MaskedFillDriver() : Driver()
    {
        data_type = miopen_type<Tgpu>{};
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&maskDesc);
    }
    ~MaskedFillDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(maskDesc);
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; };
    int GetandSetData() override;
    int AllocateBuffersAndCopy() override;
    int RunForwardGPU() override;
    int VerifyForward() override;
    int RunBackwardGPU() override;
    int VerifyBackward() override;
};
template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward MaskedFill (Default=1)", "int");

    inflags.AddTensorFlag("input", 'I', "2x2", "");
    inflags.AddTensorFlag("mask", 'm', "", "Identical to the input tensor descriptor");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}
template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    if(inflags.GetValueInt("time") == 1)
        miopenEnableProfiling(GetHandle(), true);

    return miopenStatusSuccess;
}
template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::GetandSetData()
{
    auto const input = inflags.GetValueTensor("input");
    auto const mask  = inflags.GetValueTensor("mask").FillMissing(input);
    SetTensorNd(inputDesc, input.lengths, input.strides, data_type);
    SetTensorNd(outputDesc, input.lengths, input.strides, data_type);
    SetTensorNd(maskDesc, mask.lengths, mask.strides, miopenInt8);
    value = prng ::gen_0_to_B<Tgpu>(static_cast<Tgpu>(1));
    return 0;
}
template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    auto const inputsize  = GetTensorSize(inputDesc);
    auto const outputsize = GetTensorSize(outputDesc);
    auto const masksize   = GetTensorSize(maskDesc);

    uint32_t const ctx = 0;
    input_dev          = std ::unique_ptr<GPUMem>{new GPUMem{ctx, inputsize, sizeof(Tgpu)}};
    output_dev         = std ::unique_ptr<GPUMem>{new GPUMem{ctx, outputsize, sizeof(Tgpu)}};
    mask_dev           = std ::unique_ptr<GPUMem>{new GPUMem{ctx, masksize, sizeof(int8_t)}};
    input              = std ::vector<Tgpu>(
        inputsize,
        static_cast<Tgpu>(
            0)); // "Note that the presence of list-initializing constructor (10) means list
                              // initialization and direct initialization do different things[.]"
                              // (https://en.cppreference.com/w/cpp/container/vector/vector)
    output = std ::vector<Tgpu>(
        outputsize,
        static_cast<Tgpu>(
            0)); // "If a class has an initializer list constructor
                 // (TypeName(initializer_list<SomeType>);), then it takes priority over other forms
                 // of construction, provided that the initializer list conforms to the sequence
                 // constructor's type. The C++11 version of std::vector has an initializer list
                 // constructor for its template type. Thus this code: `std::vector<int>
                 // the_vec{4};` will call the initializer list constructor, not the constructor of
                 // std::vector that takes a single size parameter and creates the vector with that
                 // size. To access the latter constructor, the user will need to use the standard
                 // constructor syntax directly."
                 // (https://en.wikipedia.org/wiki/C%2B%2B11#Uniform_initialization)
    mask       = std ::vector<int8_t>(masksize, 0);
    hostoutput = std ::vector<Tref>(outputsize, static_cast<Tref>(0));

    for(auto i = 0; i < inputsize; ++i)
        input[i] = prng ::gen_0_to_B<Tgpu>(static_cast<Tgpu>(1));
    for(auto i = 0; i < masksize; ++i)
        mask[i] = prng ::gen_0_to_B<int8_t>(1);

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        std ::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize() << std ::endl;
    if(output_dev->ToGPU(GetStream(), output.data()) != 0)
        std ::cerr << "Error copying (output) to GPU, size: " << output_dev->GetSize()
                   << std ::endl;
    if(mask_dev->ToGPU(GetStream(), mask.data()) != 0)
        std ::cerr << "Error copying (mask) to GPU, size: " << mask_dev->GetSize() << std ::endl;

    return miopenStatusSuccess;
}
template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_first_time = 0, kernel_total_time = 0;
    Timer t;
    START_TIME
    for(auto i = 0; i < inflags.GetValueInt("iter"); ++i)
    {
        miopenMaskedFillForward(GetHandle(),

                                inputDesc,
                                input_dev->GetMem(),
                                outputDesc,
                                output_dev->GetMem(),

                                maskDesc,
                                mask_dev->GetMem(),

                                value);
        float time = 0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }
    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        auto iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std ::cout << "Wall-clock Time Forward MaskedFill Elapsed: " << t.gettime_ms() / iter
                       << " ms\n";
        auto kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward MaskedFill Elapsed: " << kernel_average_time
                  << " ms\n";
    }
    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std ::cerr << "Error copying (output_dev) from GPU, size: " << output_dev->GetSize()
                   << std ::endl;
    return miopenStatusSuccess;
}
template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::VerifyForward()
{
    mloMaskedFillForwardRunHost<Tgpu, Tref>(outputDesc,

                                            input.data(),
                                            hostoutput.data(),

                                            mask.data(),

                                            value);
    auto const error =
        miopen ::range_product(output, hostoutput, 0, miopen ::sum, miopen ::abs_diff);
    if(error == 0)
    {
        std ::cout << "Forward MaskedFill Verifies OK on CPU reference" << std ::endl;
    }
    else
    {
        std ::cout << "Forward MaskedFill FAILED: " << error << " > 0" << std ::endl;
        return EC_VerifyFwd;
    }
    return miopenStatusSuccess;
}
template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_first_time = 0, kernel_total_time = 0;
    Timer t;
    START_TIME
    for(auto i = 0; i < inflags.GetValueInt("iter"); ++i)
    {
        miopenMaskedFillBackward(GetHandle(),

                                 inputDesc,
                                 input_dev->GetMem(),
                                 outputDesc,
                                 output_dev->GetMem(),

                                 maskDesc,
                                 mask_dev->GetMem(),

                                 value);
        float time = 0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }
    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        auto iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std ::cout << "Wall-clock Time Backward MaskedFill Elapsed: " << t.gettime_ms() / iter
                       << " ms\n";
        auto kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward MaskedFill Elapsed: " << kernel_average_time
                  << " ms\n";
    }
    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std ::cerr << "Error copying (output_dev) from GPU, size: " << output_dev->GetSize()
                   << std ::endl;
    return miopenStatusSuccess;
}
template <typename Tgpu, typename Tref>
int MaskedFillDriver<Tgpu, Tref>::VerifyBackward()
{
    mloMaskedFillBackwardRunHost<Tgpu, Tref>(outputDesc,

                                             input.data(),
                                             hostoutput.data(),

                                             mask.data());
    auto const error =
        miopen ::range_product(output, hostoutput, 0, miopen ::sum, miopen ::abs_diff);
    if(error == 0)
    {
        std ::cout << "Backward MaskedFill Verifies OK on CPU reference" << std ::endl;
    }
    else
    {
        std ::cout << "Backward MaskedFill FAILED: " << error << " > 0" << std ::endl;
        return EC_VerifyBwd;
    }
    return miopenStatusSuccess;
}

#endif
