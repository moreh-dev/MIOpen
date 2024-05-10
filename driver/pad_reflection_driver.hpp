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
#ifndef GUARD_MIOPEN_PAD_REFLECTION_DRIVER_HPP
#define GUARD_MIOPEN_PAD_REFLECTION_DRIVER_HPP

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

#ifndef MLO_PADREFLECTIONHOST_H_
#define MLO_PADREFLECTIONHOST_H_

template <typename Tgpu, typename Tcheck>
void mloPadReflectionRunForwardHost(miopenTensorDescriptor_t inputDesc,
                                    miopenTensorDescriptor_t outputDesc,
                                    int contiguous,
                                    Tgpu* input,
                                    Tcheck* outputhost,
                                    std::vector<size_t> padding)
{
    auto input_size  = miopen::deref(inputDesc).GetSize();
    auto input_dims  = miopen::deref(inputDesc).GetLengths();
    auto output_dims = miopen::deref(outputDesc).GetLengths();
    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());
    if(input_size == 3 && contiguous == 1)
    {
        long padding_l     = padding[0];
        auto input_strides = miopen::deref(inputDesc).GetStrides();
        size_t in_W        = input_dims[2];

        long in_start_x  = max(0L, -padding_l);
        long out_start_x = max(0L, padding_l);

        for(size_t gid = 0; gid < output_numel; gid++)
        {
            long n, c, w;
            ulong nc = gid / output_dims[2];
            w        = gid % output_dims[2];
            n        = nc / output_dims[1];
            c        = nc % output_dims[1];

            if(w < padding_l)
            {
                w = padding_l * 2 - w;
            }
            else if(!(padding_l <= w && w < in_W + padding_l))
            {
                w = (in_W + padding_l - 1) * 2 - w;
            }
            w = w - out_start_x + in_start_x;

            outputhost[gid] = input[(input_strides[2] * (w)) + (input_strides[1] * (c)) +
                                    (input_strides[0] * (n)) + 0];
        }
    }
    else if(input_size == 3 && contiguous == 0)
    {
        long padding_l      = padding[0];
        auto input_strides  = miopen::deref(inputDesc).GetStrides();
        auto output_strides = miopen::deref(outputDesc).GetStrides();
        std::cout << "Vector elements: ";
        for(size_t num : output_strides)
        {
            std::cout << num << " ";
        }
        std::cout << std::endl;
        size_t in_W = input_dims[2];

        long in_start_x  = max(0L, -padding_l);
        long out_start_x = max(0L, padding_l);

        for(size_t gid = 0; gid < output_numel; gid++)
        {
            long n, c, w;
            ulong nc = gid / output_dims[2];
            w        = gid % output_dims[2];
            n        = nc / output_dims[1];
            c        = nc % output_dims[1];

            if(w < padding_l)
            {
                w = padding_l * 2 - w;
            }
            else if(!(padding_l <= w && w < in_W + padding_l))
            {
                w = (in_W + padding_l - 1) * 2 - w;
            }
            w                 = w - out_start_x + in_start_x;
            size_t output_idx = output_strides[0] * (gid / output_dims[2] / output_dims[1]) +
                                output_strides[1] * ((gid / output_dims[2]) % output_dims[1]) +
                                output_strides[2] * (gid % output_dims[2]) + 0;
            Tgpu val               = input[(input_strides[2] * (w)) + (input_strides[1] * (c)) +
                             (input_strides[0] * (n)) + 0];
            outputhost[output_idx] = val;
        }
    }
    // else if(input_size == 4)
    // {
    //     long padding_l     = padding[0];
    //     long padding_t     = padding[2];
    //     auto input_strides = miopen::deref(inputDesc).GetStrides();
    //     size_t in_H        = input_dims[2];
    //     size_t in_W        = input_dims[3];

    //     for(size_t gid = 0; gid < output_numel; gid++)
    //     {
    //         long n, c, h, w;
    //         // GET_NCHW(n, c, h, w, gid, output);
    //         ulong nch = (gid) / output_dims[3];
    //         w         = (gid) % output_dims[3];
    //         ulong nc  = nch / output_dims[2];
    //         h         = nch % output_dims[2];
    //         n         = nc / output_dims[1];
    //         c         = nc % output_dims[1];

    //         long in_start_x  = max(0L, -padding_l);
    //         long in_start_y  = max(0L, -padding_t);
    //         long out_start_x = max(0L, padding_l);
    //         long out_start_y = max(0L, padding_t);

    //         if(w < padding_l)
    //         {
    //             w = padding_l * 2 - w;
    //         }
    //         else if(padding_l <= w && w < in_W + padding_l)
    //         {
    //         }
    //         else
    //         {
    //             w = (in_W + padding_l - 1) * 2 - w;
    //         }
    //         w = w - out_start_x + in_start_x;

    //         if(h < padding_t)
    //         {
    //             h = padding_t * 2 - h;
    //         }
    //         else if(padding_t <= h && h < in_H + padding_t)
    //         {
    //         }
    //         else
    //         {
    //             h = (in_H + padding_t - 1) * 2 - h;
    //         }
    //         h = h - out_start_y + in_start_y;

    //         outputhost[gid] = input[(input_strides[3] * (w)) + (input_strides[2] * (h)) +
    //                                 (input_strides[1] * (c)) + (input_strides[0] * (n)) + 0];
    //     }
    // }
}

template <typename Tgpu, typename Tcheck>
void mloPadReflectionRunBackwardHost(miopenTensorDescriptor_t inputDesc,
                                     miopenTensorDescriptor_t outputDesc,
                                     int contiguous,
                                     Tcheck* input,
                                     Tgpu* output,
                                     std::vector<size_t> padding)
{
    auto input_size  = miopen::deref(inputDesc).GetSize();
    auto input_dims  = miopen::deref(inputDesc).GetLengths();
    auto output_dims = miopen::deref(outputDesc).GetLengths();
    auto output_numel =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());
    if(input_size == 3 && contiguous == 1)
    {
        long padding_l     = padding[0];
        auto input_strides = miopen::deref(inputDesc).GetStrides();
        size_t in_W        = input_dims[2];

        long in_start_x  = max(0L, -padding_l);
        long out_start_x = max(0L, padding_l);

        for(size_t gid = 0; gid < output_numel; gid++)
        {
            long n, c, w;
            ulong nc = gid / output_dims[2];
            w        = gid % output_dims[2];
            n        = nc / output_dims[1];
            c        = nc % output_dims[1];

            if(w < padding_l)
            {
                w = padding_l * 2 - w;
            }
            else if(!(padding_l <= w && w < in_W + padding_l))
            {
                w = (in_W + padding_l - 1) * 2 - w;
            }
            w        = w - out_start_x + in_start_x;
            input[(input_strides[2] * (w)) + (input_strides[1] * (c)) + (input_strides[0] * (n)) +
                  0] = input[(input_strides[2] * (w)) + (input_strides[1] * (c)) +
                             (input_strides[0] * (n)) + 0] +
                       output[gid];
        }
    }
    else if(input_size == 3 && contiguous == 0)
    {
        long padding_l      = padding[0];
        auto input_strides  = miopen::deref(inputDesc).GetStrides();
        auto output_strides = miopen::deref(outputDesc).GetStrides();
        std::cout << std::endl;
        size_t in_W = input_dims[2];

        long in_start_x  = max(0L, -padding_l);
        long out_start_x = max(0L, padding_l);

        for(size_t gid = 0; gid < output_numel; gid++)
        {
            long n, c, w;
            ulong nc = gid / output_dims[2];
            w        = gid % output_dims[2];
            n        = nc / output_dims[1];
            c        = nc % output_dims[1];

            if(w < padding_l)
            {
                w = padding_l * 2 - w;
            }
            else if(!(padding_l <= w && w < in_W + padding_l))
            {
                w = (in_W + padding_l - 1) * 2 - w;
            }
            w                 = w - out_start_x + in_start_x;
            size_t output_idx = output_strides[0] * (gid / output_dims[2] / output_dims[1]) +
                                output_strides[1] * ((gid / output_dims[2]) % output_dims[1]) +
                                output_strides[2] * (gid % output_dims[2]) + 0;
            input[(input_strides[2] * (w)) + (input_strides[1] * (c)) + (input_strides[0] * (n)) +
                  0] = input[(input_strides[2] * (w)) + (input_strides[1] * (c)) +
                             (input_strides[0] * (n)) + 0] +
                       output[output_idx];
        }
    }
}
#endif

template <typename Tgpu, typename Tref>
class PadReflectionDriver : public Driver
{
public:
    PadReflectionDriver() : Driver()
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

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~PadReflectionDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;

    int forw;

    int contiguous;
    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;
    std::vector<Tref> inhost;

    std::vector<size_t> padding;
};

template <typename Tgpu, typename Tref>
int PadReflectionDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename T>
inline std::vector<T> GetStrides(std::vector<T> input, bool contiguous)
{
    if(!contiguous)
        std::swap(input.front(), input.back());
    std::vector<T> strides(input.size());
    strides.back() = 1;
    for(int i = input.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * input[i + 1];
    if(!contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
int PadReflectionDriver<Tgpu, Tref>::GetandSetData()
{
    std::string padding_str = inflags.GetValueStr("padding");
    std::stringstream padding_ss(padding_str);
    std::string padding_token;
    while(std::getline(padding_ss, padding_token, ','))
    {
        padding.push_back(std::stoul(padding_token));
    }

    // if(!(padding.size() == 1 || padding.size() == 4))
    if(!(padding.size() == 1))
    {
        std::cerr << "Error Padding Lengths\n" << std::endl;
    }
    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();

    auto input_strides = GetStrides(in_len, contiguous == 1);
    SetTensorNd(inputDesc, in_len, input_strides, data_type);

    std::vector<int> out_len;
    auto in_len_size = in_len.size();
    if(in_len_size == 3)
    {
        for(int i = 0; i < in_len_size; i++)
        {
            // If W
            if(i == 2)
            {
                out_len.push_back(in_len[i] + 2 * padding[0]);
            }
            else
            {
                out_len.push_back(in_len[i]);
            }
        }
    }
    else if(in_len_size == 4)
    {
        for(int i = 0; i < in_len.size(); i++)
        {
            // If H
            if(i == 2)
            {
                out_len.push_back(in_len[i] + 2 * padding[2]);
            }
            // If W
            else if(i == 3)
            {
                out_len.push_back(in_len[i] + 2 * padding[0]);
            }
            else
            {
                out_len.push_back(in_len[i]);
            }
        }
    }

    if(out_len.empty())
        out_len.push_back(1);
    auto output_strides = GetStrides(out_len, contiguous == 1);

    SetTensorNd(outputDesc, out_len, output_strides, data_type);

    return 0;
}

template <typename Tgpu, typename Tref>
int PadReflectionDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Pad Reflection (Default=1)", "int");
    inflags.AddInputFlag("batchsize", 'n', "256", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "4", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_d", 'D', "0", "Input Depth (Default=0)", "int");
    inflags.AddInputFlag("in_h", 'H', "0", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag("padding", 'P', "1", "Padding array (Default=1 or 1,1,1,1)", "str");
    inflags.AddInputFlag("contiguous", 'C', "1", "Contiguous (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> PadReflectionDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n   = inflags.GetValueInt("batchsize");
    int in_c   = inflags.GetValueInt("in_channels");
    int in_w   = inflags.GetValueInt("in_w");
    int in_h   = inflags.GetValueInt("in_h");
    int in_d   = inflags.GetValueInt("in_d");
    contiguous = inflags.GetValueInt("contiguous");

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
int PadReflectionDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz  = GetTensorSize(inputDesc);
    size_t out_sz = GetTensorSize(outputDesc);

    uint32_t ctx = 0;

    in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    in      = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    inhost  = std::vector<Tref>(in_sz, static_cast<Tref>(0));
    fill(out.begin(), out.end(), static_cast<Tgpu>(0));
    fill(outhost.begin(), outhost.end(), static_cast<Tgpu>(0));

    for(int i = 0; i < in_sz; i++)
    {
        in[i]     = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        inhost[i] = static_cast<Tref>(in[i]);
    }

    if(in_dev->ToGPU(GetStream(), in.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;

    if(out_dev->ToGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PadReflectionDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        if(contiguous == 1)
        {
            miopenPadReflection1dFwdContiguous(GetHandle(),
                                               inputDesc,
                                               in_dev->GetMem(),
                                               outputDesc,
                                               out_dev->GetMem(),
                                               padding.data(),
                                               padding.size());
        }
        else if(contiguous == 0)
        {
            miopenPadReflection1dFwd(GetHandle(),
                                     inputDesc,
                                     in_dev->GetMem(),
                                     outputDesc,
                                     out_dev->GetMem(),
                                     padding.data(),
                                     padding.size());
        }
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
            std::cout << "Wall-clock Time Forward Pad Reflection Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward Pad Reflection Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PadReflectionDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloPadReflectionRunForwardHost<Tgpu, Tref>(
        inputDesc, outputDesc, contiguous, in.data(), outhost.data(), padding);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PadReflectionDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        if(contiguous == 1)
        {
            miopenPadReflection1dBwdContiguous(GetHandle(),
                                               inputDesc,
                                               in_dev->GetMem(),
                                               outputDesc,
                                               out_dev->GetMem(),
                                               padding.data(),
                                               padding.size());
        }
        else if(contiguous == 0)
        {
            miopenPadReflection1dBwd(GetHandle(),
                                     inputDesc,
                                     in_dev->GetMem(),
                                     outputDesc,
                                     out_dev->GetMem(),
                                     padding.data(),
                                     padding.size());
        }
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
            std::cout << "Wall-clock Time Backward Pad Reflection Elapsed: "
                      << t.gettime_ms() / iter << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward Pad Reflection Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(in_dev->FromGPU(GetStream(), in.data()) != 0)
        std::cerr << "Error copying (in_dev) from GPU, size: " << in_dev->GetSize() << std::endl;
    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (in_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PadReflectionDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloPadReflectionRunBackwardHost<Tgpu, Tref>(
        inputDesc, outputDesc, contiguous, inhost.data(), out.data(), padding);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref PadReflectionDriver<Tgpu, Tref>::GetTolerance()
{
    // Computation error of fp16 is ~2^13 (=8192) bigger than
    // the one of fp32 because mantissa is shorter by 13 bits.
    auto tolerance = std::is_same<Tgpu, float>::value ? 1.5e-6 : 8.2e-3;

    // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
    if(std::is_same<Tgpu, bfloat16>::value)
        tolerance *= 80.0;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int PadReflectionDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = 0.0;
    auto error           = miopen::rms_range(outhost, out);

    if(std::abs(static_cast<float>(error)) != 0.0f)
    {
        std::cout << "Pad Reflection Fwd FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Pad Reflection Verifies OK on CPU reference (" << error << " == " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int PadReflectionDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(inhost, in);

    if(error > tolerance)
    {
        std::cout << "Pad Reflection Bwd FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Pad Reflection Verifies OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_PAD_REFLECTION_DRIVER_HPP
