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
#ifndef GUARD_MIOPEN_SOFTMAXCROSSENTROPYWITHLOGITS_DRIVER_HPP
#define GUARD_MIOPEN_SOFTMAXCROSSENTROPYWITHLOGITS_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "mloSoftmaxCrossEntropyWithLogitsHost.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "util_driver.hpp"

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <vector>

inline std::vector<int> GetStrides(std::vector<int> lengths, int contiguous)
{
    if(contiguous != 0 && contiguous != 1)
        std::cerr << "Error Tensor Contiguous should be 0 or 1" << std::endl;
    if(contiguous == 0)
        std::swap(lengths.front(), lengths.back());
    std::vector<int> strides(lengths.size());
    strides.back() = 1;
    for(int i = lengths.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * lengths[i + 1];
    if(contiguous == 0)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
class SoftmaxCrossEntropyWithLogitsDriver : public Driver
{
public:
    SoftmaxCrossEntropyWithLogitsDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&backpropDesc);
        miopenCreateTensorDescriptor(&outputGradDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&targetGradDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    std::vector<int> GetInputTensorDimsFromCmd();
    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    int VerifyBackward() override;
    int VerifyForward() override;
    ~SoftmaxCrossEntropyWithLogitsDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(backpropDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(targetGradDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t backpropDesc;
    miopenTensorDescriptor_t outputGradDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t targetGradDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> backprop_dev;

    std::unique_ptr<GPUMem> workspace_dev_fwd;
    std::unique_ptr<GPUMem> workspace_dev_bwd;
    std::unique_ptr<GPUMem> out_grad_dev;
    std::unique_ptr<GPUMem> in_grad_dev;
    std::unique_ptr<GPUMem> target_grad_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> target;
    std::vector<Tgpu> out;
    std::vector<Tgpu> backprop;
    std::vector<Tref> out_host;
    std::vector<Tref> backprop_host;

    std::vector<Tgpu> out_grad;
    std::vector<Tgpu> in_grad;
    std::vector<Tgpu> target_grad;
    std::vector<Tref> in_grad_host;
    std::vector<Tref> target_grad_host;

    size_t ws_sizeInBytes_fwd = 0;
    size_t ws_sizeInBytes_bwd = 0;

    std::vector<int> input_sizes;
};

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::GetInputTensorDimsFromCmd()
{
    std::string lengthsStr = inflags.GetValueStr("input_dims");

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
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::GetandSetData()
{
    input_sizes = GetInputTensorDimsFromCmd();

    std::vector<int> in_len       = input_sizes;
    std::vector<int> target_len   = input_sizes;
    std::vector<int> out_len      = std::vector<int>{in_len[0]};
    std::vector<int> backprop_len = input_sizes;

    auto in_strides       = GetStrides(in_len, inflags.GetValueInt("contiguous"));
    auto tar_strides      = GetStrides(target_len, 1);
    auto output_strides   = GetStrides(out_len, 1);
    auto backprop_strides = GetStrides(backprop_len, 1);

    SetTensorNd(inputDesc, in_len, in_strides, data_type);
    SetTensorNd(targetDesc, target_len, tar_strides, data_type);
    SetTensorNd(outputDesc, out_len, output_strides, data_type);
    SetTensorNd(backpropDesc, backprop_len, backprop_strides, data_type);

    SetTensorNd(outputGradDesc, out_len, output_strides, data_type);
    SetTensorNd(inputGradDesc, in_len, in_strides, data_type);
    SetTensorNd(targetGradDesc, target_len, tar_strides, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag(
        "forw", 'F', "1", "Run only Forward SoftmaxCrossEntropyWithLogits (Default=1)", "int");
    inflags.AddInputFlag("input_dims",
                         'D',
                         "16,21",
                         "The dimensional lengths of the input tensor: N,C. Example: 16,64.",
                         "string");
    inflags.AddInputFlag("contiguous",
                         'c',
                         "1",
                         "Is input tensor contiguous? (Default=1 for contiguous tensor)",
                         "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz       = GetTensorSize(inputDesc);
    size_t target_sz   = GetTensorSize(targetDesc);
    size_t out_sz      = GetTensorSize(outputDesc);
    size_t backprop_sz = GetTensorSize(backpropDesc);

    uint32_t ctx = 0;

    in_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    target_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(Tgpu)));
    out_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    backprop_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, backprop_sz, sizeof(Tgpu)));

    workspace_dev_fwd =
        std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes_fwd, sizeof(std::byte)));

    workspace_dev_bwd =
        std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes_bwd, sizeof(std::byte)));

    out_grad_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    in_grad_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    target_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(Tgpu)));

    in            = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    target        = std::vector<Tgpu>(target_sz, static_cast<Tgpu>(0));
    out           = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    backprop      = std::vector<Tgpu>(backprop_sz, static_cast<Tgpu>(0));
    out_host      = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    backprop_host = std::vector<Tref>(backprop_sz, static_cast<Tref>(0));

    out_grad         = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    in_grad          = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    target_grad      = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    in_grad_host     = std::vector<Tref>(in_sz, static_cast<Tref>(0));
    target_grad_host = std::vector<Tref>(in_sz, static_cast<Tref>(0));

    int status;

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-5.0f), static_cast<Tgpu>(1.0f));
    }
    status = in_dev->ToGPU(q, in.data());

    size_t num_classes = out_sz;
    size_t num_batches = in_sz / num_classes;
    for(int i = 0; i < num_batches; i++)
    {
        for(int j = 0; j < num_classes; j++)
        {
            if(j == i % num_classes)
                target[i * num_classes + j] = (static_cast<Tgpu>(1.0f));
            else
                target[i * num_classes + j] = (static_cast<Tgpu>(0.0f));
        }
    }

    status |= target_dev->ToGPU(q, target.data());

    status |= out_dev->ToGPU(q, out.data());

    status |= backprop_dev->ToGPU(q, backprop.data());

    status |= in_grad_dev->ToGPU(q, in_grad.data());
    status |= target_grad_dev->ToGPU(q, target_grad.data());

    for(int i = 0; i < out_sz; i++)
    {
        out_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-10.0), static_cast<Tgpu>(10.0));
    }
    status |= out_grad_dev->ToGPU(q, out_grad.data());

    if(status != 0)
        std::cout << "Error copying data to GPU\n" << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenSoftmaxCrossEntropyWithLogitsForward(GetHandle(),
                                                   workspace_dev_fwd->GetMem(),
                                                   ws_sizeInBytes_fwd,
                                                   inputDesc,
                                                   in_dev->GetMem(),
                                                   targetDesc,
                                                   target_dev->GetMem(),
                                                   outputDesc,
                                                   out_dev->GetMem(),
                                                   backpropDesc,
                                                   backprop_dev->GetMem());

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
            printf("Wall-clock Time Forward SoftmaxCrossEntropyWithLogits Elapsed: %f ms\n",
                   t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Forward SoftmaxCrossEntropyWithLogits Elapsed: %f ms\n",
               kernel_average_time);
    }

    out_dev->FromGPU(GetStream(), out.data());
    backprop_dev->FromGPU(GetStream(), backprop.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloSoftmaxCrossEntropyWithLogitsForward<Tgpu, Tref>(inputDesc,
                                                        targetDesc,
                                                        outputDesc,
                                                        backpropDesc,
                                                        in.data(),
                                                        target.data(),
                                                        out_host.data(),
                                                        backprop_host.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenSoftmaxCrossEntropyWithLogitsBackward(GetHandle(),
                                                    workspace_dev_bwd->GetMem(),
                                                    ws_sizeInBytes_bwd,
                                                    outputGradDesc,
                                                    out_grad_dev->GetMem(),
                                                    backpropDesc,
                                                    backprop_dev->GetMem(),
                                                    inputDesc,
                                                    in_dev->GetMem(),
                                                    inputGradDesc,
                                                    in_grad_dev->GetMem(),
                                                    targetGradDesc,
                                                    target_grad_dev->GetMem());

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
            printf("Wall-clock Time Backward SoftmaxCrossEntropyWithLogits Elapsed: %f ms\n",
                   t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Backward SoftmaxCrossEntropyWithLogits Elapsed: %f ms\n",
               kernel_average_time);
    }

    in_grad_dev->FromGPU(GetStream(), in_grad.data());
    target_grad_dev->FromGPU(GetStream(), target_grad.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloSoftmaxCrossEntropyWithLogitsBackward<Tgpu, Tref>(outputGradDesc,
                                                         backpropDesc,
                                                         inputDesc,
                                                         inputGradDesc,
                                                         targetGradDesc,
                                                         out_grad.data(),
                                                         backprop.data(),
                                                         in.data(),
                                                         in_grad_host.data(),
                                                         target_grad_host.data(),
                                                         true,
                                                         true);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    auto tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;

    auto error = miopen::rms_range(out_host, out);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Output Forward SoftmaxCrossEntropyWithLogits FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        printf("Output Forward SoftmaxCrossEntropyWithLogits Verifies on CPU and GPU (err=%f)\n",
               error);
    }

    auto backprop_error = miopen::rms_range(backprop_host, backprop);
    if(!std::isfinite(backprop_error) || backprop_error > tolerance)
    {
        std::cout << "Backprop Forward SoftmaxCrossEntropyWithLogits FAILED: " << backprop_error
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        printf("Backprop Forward SoftmaxCrossEntropyWithLogits Verifies on CPU and GPU (err=%f)\n",
               backprop_error);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxCrossEntropyWithLogitsDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    auto tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    auto error1    = miopen::rms_range(in_grad_host, in_grad);

    if(!std::isfinite(error1) || error1 > tolerance)
    {
        std::cout << "Backward SoftmaxCrossEntropyWithLogits in Input Grad FAILED: " << error1
                  << " while tolerance: " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        printf("Backward SoftmaxCrossEntropyWithLogits Verifies in Input Grad on CPU and GPU "
               "(err=%f)\n",
               error1);
    }

    auto error2 = miopen::rms_range(target_grad_host, target_grad);

    if(!std::isfinite(error2) || error2 > tolerance)
    {
        std::cout << "Backward SoftmaxCrossEntropyWithLogits in Target Grad FAILED: " << error2
                  << " while tolerance: " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        printf("Backward SoftmaxCrossEntropyWithLogits Verifies in Target Grad on CPU and GPU "
               "(err=%f)\n",
               error2);
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_SOFTMAXCROSSENTROPYWITHLOGITS_DRIVER_HPP
