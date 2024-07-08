/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_REPEAT_DRIVER_HPP
#define GUARD_MIOPEN_REPEAT_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include "../test/verify.hpp"

#include <algorithm>
#include <cstdlib>
#include <cfloat>
#include <memory>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>

template <typename Tgpu, typename Tref>
class RepeatDriver : public Driver
{
public:
    RepeatDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetSizesFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyForward() override;
    int VerifyBackward() override;

    ~RepeatDriver()
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;

    std::vector<Tref> outhost;

    std::vector<int> sizes;
    int num_sizes;
};

template <typename Tgpu, typename Tref>
int RepeatDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    
    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RepeatDriver<Tgpu, Tref>::GetandSetData()
{
    sizes     = GetSizesFromCmdLine();
    num_sizes = sizes.size();

    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();
    std::vector<int> out_len;

    SetTensorNd(inputDesc, in_len, data_type);

    return 0;
}

template <typename Tgpu, typename Tref>
int RepeatDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward (Default=1)", "int");
    inflags.AddInputFlag("DimLengths",
                         'D',
                         "20,10",
                         "The dimensional lengths of the input tensor (Default=20,10)",
                         "string");

    inflags.AddInputFlag("repeat", 'r', "4,3", "Number of times to repeat each dimension (Default=4,3)", "string");

    inflags.AddInputFlag("iter", 'i', "10", "Number of iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'v', "1", "Verify Forward and Backward (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> RepeatDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
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

template <typename Tgpu, typename Tref>
std::vector<int> RepeatDriver<Tgpu, Tref>::GetSizesFromCmdLine()
{
    std::string sizesStr = inflags.GetValueStr("repeat");

    std::vector<int> sizes_;
    std::size_t pos = 0;
    std::size_t new_pos;

    new_pos = sizesStr.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = sizesStr.substr(pos, new_pos - pos);

        int len = std::stoi(sliceStr);

        sizes_.push_back(len);

        pos     = new_pos + 1;
        new_pos = sizesStr.find(',', pos);
    };

    std::string sliceStr = sizesStr.substr(pos);
    int len              = std::stoi(sliceStr);

    sizes_.push_back(len);

    return (sizes_);
}

template <typename Tgpu, typename Tref>
int RepeatDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz  = GetTensorSize(inputDesc);
    size_t out_sz = GetTensorSize(outputDesc);

    uint32_t ctx = 0;

    in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    in  = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

    int status;

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status = in_dev->ToGPU(q, in.data());

    status |= out_dev->ToGPU(q, out.data());

    if (status != 0)
    {
        std::cout << "Error copying data to GPU\n" << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RepeatDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenRepeatForward(GetHandle(),
                            inputDesc,
                            in_dev->GetMem(),
                            sizes.data(),
                            num_sizes,
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
            printf("Wall-clock Time Forward Repeat Elapsed: %f ms\n", t.gettime_ms() / iter);

        float kernel_average_time = iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU kernel Time Forward Repeat Elapsed: %f ms\n", kernel_average_time);
    }

    out_dev->FromGPU(GetStream(), out.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RepeatDriver<Tgpu, Tref>::RunForwardCPU()
{
    //todo : runforwardcpu
    return 0;
}

template <typename Tgpu, typename Tref>
int RepeatDriver<Tgpu, Tref>::RunBackwardGPU()
{
    //todo : runbackwardgpu
    return 0;
}

template <typename Tgpu, typename Tref>
int RepeatDriver<Tgpu, Tref>::RunBackwardCPU()
{
    //todo : runbackwardcpu
    return 0;
}

template <typename Tgpu, typename Tref>
Tref RepeatDriver<Tgpu, Tref>::GetTolerance()
{
    if(data_type == miopenHalf)
    {
        return 1e-3;
    }
    else if(data_type == miopenFloat)
    {
        return 5e-5;
    }
    else if(data_type == miopenDouble)
    {
        return 1e-10;
    }
    else if(data_type == miopenBFloat16)
    {
        return 5e-3;
    }
    return 1e-3;
}

template <typename Tgpu, typename Tref>
int RepeatDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error          = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Repeat Failed: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        printf("Forward Repeat Verifies on CPU and GPU (err=%f)\n", error);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int RepeatDriver<Tgpu, Tref>::VerifyBackward()
{
    //todo : verifybackward
    return 0;
}

#endif // GUARD_MIOPEN_REPEAT_DRIVER_HPP
