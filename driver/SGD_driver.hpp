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

#ifndef GUARD_MIOPEN_SGD_DRIVER_HPP
#define GUARD_MIOPEN_SGD_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#ifndef MLO_SGDHOST_H_
#define MLO_SGDHOST_H_

template <typename Tgpu, typename Tcheck>
int32_t mloSGDForwardRunHost(miopenTensorDescriptor_t paramInputDesc,
                             Tgpu* paramInput,
                             miopenTensorDescriptor_t paramOutputDesc,
                             Tcheck* paramOutputHost,
                             miopenTensorDescriptor_t gradDesc,
                             Tgpu* grad,
                             miopenTensorDescriptor_t momentumBufferInputDesc,
                             Tgpu* momentumBufferInput,
                             miopenTensorDescriptor_t momentumBufferOutputDesc,
                             Tcheck* momentumBufferOutputHost,
                             double lr,
                             double momentum,
                             double dampening,
                             double weightDecay,
                             char nesterov,
                             char momentumInitialized)
{
    auto dims         = miopen::deref(paramInputDesc).GetLengths();
    size_t param_size = std::accumulate(dims.begin(), dims.end(), 1ULL, std::multiplies<size_t>());

    int32_t ret = 0;

    for(int id = 0; id < param_size; ++id)
    {
        Tcheck param = static_cast<Tcheck>(paramInput[id]);
        Tcheck d_p   = static_cast<Tcheck>(grad[id]);

        if(weightDecay != 0)
        {
            d_p += param * static_cast<Tcheck>(weightDecay);
        }

        if(momentum != 0)
        {
            Tcheck momentum_v;
            if(momentumInitialized)
            {
                momentum_v = static_cast<Tcheck>(momentumBufferInput[id]);
                momentum_v = momentum_v * static_cast<Tcheck>(momentum) +
                             d_p * static_cast<Tcheck>(1 - dampening);
            }
            else
            {
                momentum_v = d_p;
            }
            momentumBufferOutputHost[id] = momentum_v;

            if(nesterov)
            {
                d_p = d_p + momentum_v * static_cast<Tcheck>(momentum);
            }
            else
            {
                d_p = momentum_v;
            }
        }

        paramOutputHost[id] = param - static_cast<Tcheck>(lr) * d_p;
    }
    return ret;
}
#endif

template <typename Tgpu, typename Tref = Tgpu>
class SGDDriver : public Driver
{
public:
    SGDDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&paramInDesc);
        miopenCreateTensorDescriptor(&paramOutDesc);
        miopenCreateTensorDescriptor(&gradDesc);
        miopenCreateTensorDescriptor(&momentumBufferInDesc);
        miopenCreateTensorDescriptor(&momentumBufferOutDesc);

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

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~SGDDriver() override
    {
        miopenDestroyTensorDescriptor(paramInDesc);
        miopenDestroyTensorDescriptor(paramOutDesc);
        miopenDestroyTensorDescriptor(gradDesc);
        miopenDestroyTensorDescriptor(momentumBufferInDesc);
        miopenDestroyTensorDescriptor(momentumBufferOutDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t paramInDesc;
    miopenTensorDescriptor_t paramOutDesc;
    miopenTensorDescriptor_t gradDesc;
    miopenTensorDescriptor_t momentumBufferInDesc;
    miopenTensorDescriptor_t momentumBufferOutDesc;

    std::unique_ptr<GPUMem> param_in_dev;
    std::unique_ptr<GPUMem> param_out_dev;
    std::unique_ptr<GPUMem> grad_dev;
    std::unique_ptr<GPUMem> momentum_buffer_in_dev;
    std::unique_ptr<GPUMem> momentum_buffer_out_dev;

    std::vector<Tgpu> param_in;
    std::vector<Tgpu> param_out;
    std::vector<Tgpu> grad;
    std::vector<Tgpu> momentum_buffer_in;
    std::vector<Tgpu> momentum_buffer_out;

    std::vector<Tref> param_outhost;
    std::vector<Tref> momentum_buffer_outhost;

    double lr;
    double momentum;
    double dampening;
    double weight_decay;
    char nesterov;
    char momentum_initialized;
};

template <typename Tgpu, typename Tref>
int SGDDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SGDDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> len = GetInputTensorLengthsFromCmdLine();
    lr                   = inflags.GetValueDouble("lr");
    momentum             = inflags.GetValueDouble("momentum");
    dampening            = inflags.GetValueDouble("dampening");
    weight_decay         = inflags.GetValueDouble("weight_decay");
    nesterov             = inflags.GetValueInt("nesterov");
    momentum_initialized = inflags.GetValueInt("momentum_initialized");

    SetTensorNd(paramInDesc, len, data_type);
    SetTensorNd(paramOutDesc, len, data_type);
    SetTensorNd(gradDesc, len, data_type);
    SetTensorNd(momentumBufferInDesc, len, data_type);
    SetTensorNd(momentumBufferOutDesc, len, data_type);

    return 0;
}

template <typename Tgpu, typename Tref>
int SGDDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward SGD (Default=1)", "int");
    inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_d", 'D', "0", "Input Depth (Default=0)", "int");
    inflags.AddInputFlag("in_h", 'H', "0", "Input Height (Default=0)", "int");
    inflags.AddInputFlag("in_w", 'W', "0", "Input Width (Default=0)", "int");

    inflags.AddInputFlag("lr", 'l', "0.01", "Learning rate (Default=0.01)", "double");
    inflags.AddInputFlag("momentum", 'm', "0.9", "Momentum factor (Default=0.9)", "double");
    inflags.AddInputFlag("dampening", 'd', "0", "Dampening for momentum (Default=0)", "double");
    inflags.AddInputFlag("weight_decay", 'e', "0", "Weight decay (Default=0)", "double");
    inflags.AddInputFlag("nesterov", 'N', "0", "Enables Nesterow momentum (Default=0)", "int");
    inflags.AddInputFlag(
        "momentum_initialized", 'M', "0", "Is momentum initiated (Default=0)", "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> SGDDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_w = inflags.GetValueInt("in_w");
    int in_h = inflags.GetValueInt("in_h");
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
int SGDDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    auto dims         = miopen::deref(paramInDesc).GetLengths();
    size_t param_size = std::accumulate(dims.begin(), dims.end(), 1ULL, std::multiplies<size_t>());

    uint32_t ctx = 0;

    param_in_dev            = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_size, sizeof(Tgpu)));
    param_out_dev           = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_size, sizeof(Tgpu)));
    grad_dev                = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_size, sizeof(Tgpu)));
    momentum_buffer_in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_size, sizeof(Tgpu)));
    momentum_buffer_out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_size, sizeof(Tgpu)));

    param_in            = std::vector<Tgpu>(param_size, static_cast<Tgpu>(0));
    param_out           = std::vector<Tgpu>(param_size, static_cast<Tgpu>(0));
    grad                = std::vector<Tgpu>(param_size, static_cast<Tgpu>(0));
    momentum_buffer_in  = std::vector<Tgpu>(param_size, static_cast<Tgpu>(0));
    momentum_buffer_out = std::vector<Tgpu>(param_size, static_cast<Tgpu>(0));

    param_outhost           = std::vector<Tref>(param_size, static_cast<Tref>(0));
    momentum_buffer_outhost = std::vector<Tref>(param_size, static_cast<Tref>(0));

    for(int i = 0; i < param_size; i++)
    {
        param_in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        grad[i]     = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        momentum_buffer_in[i] =
            prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    if(param_in_dev->ToGPU(GetStream(), param_in.data()) != 0)
        std::cerr << "Error copying param (in) to GPU, size: " << param_in_dev->GetSize()
                  << std::endl;
    if(param_out_dev->ToGPU(GetStream(), param_out.data()) != 0)
        std::cerr << "Error copying param (out) to GPU, size: " << param_out_dev->GetSize()
                  << std::endl;
    if(grad_dev->ToGPU(GetStream(), grad.data()) != 0)
        std::cerr << "Error copying grad (in) to GPU, size: " << grad_dev->GetSize() << std::endl;
    if(momentum_buffer_in_dev->ToGPU(GetStream(), momentum_buffer_in.data()) != 0)
        std::cerr << "Error copying momentum buffer (in) to GPU, size: "
                  << momentum_buffer_in_dev->GetSize() << std::endl;
    if(momentum_buffer_out_dev->ToGPU(GetStream(), momentum_buffer_out.data()) != 0)
        std::cerr << "Error copying momentum buffer (out) to GPU, size: "
                  << momentum_buffer_out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SGDDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenSGDForward(GetHandle(),
                         paramInDesc,
                         param_in_dev->GetMem(),
                         paramOutDesc,
                         param_out_dev->GetMem(),
                         gradDesc,
                         grad_dev->GetMem(),
                         momentumBufferInDesc,
                         momentum_buffer_in_dev->GetMem(),
                         momentumBufferOutDesc,
                         momentum_buffer_out_dev->GetMem(),
                         lr,
                         momentum,
                         dampening,
                         weight_decay,
                         nesterov,
                         momentum_initialized);

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
            std::cout << "Wall-clock Time Forward SGD Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward SGD Elapsed: " << kernel_average_time << " ms\n";
    }

    if(param_out_dev->FromGPU(GetStream(), param_out.data()) != 0)
        std::cerr << "Error copying (param_out_dev) from GPU, size: " << param_out_dev->GetSize()
                  << std::endl;

    if(momentum_buffer_out_dev->FromGPU(GetStream(), momentum_buffer_out.data()) != 0)
        std::cerr << "Error copying (momentum_buffer_out_dev) from GPU, size: "
                  << momentum_buffer_out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SGDDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloSGDForwardRunHost<Tgpu, Tref>(paramInDesc,
                                     param_in.data(),
                                     paramOutDesc,
                                     param_outhost.data(),
                                     gradDesc,
                                     grad.data(),
                                     momentumBufferInDesc,
                                     momentum_buffer_in.data(),
                                     momentumBufferOutDesc,
                                     momentum_buffer_outhost.data(),
                                     lr,
                                     momentum,
                                     dampening,
                                     weight_decay,
                                     nesterov,
                                     momentum_initialized);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SGDDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref SGDDriver<Tgpu, Tref>::GetTolerance()
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
int SGDDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance       = GetTolerance();
    auto param_error           = miopen::rms_range(param_outhost, param_out);
    auto momentum_buffer_error = miopen::rms_range(momentum_buffer_outhost, momentum_buffer_out);

    if(!std::isfinite(param_error) || param_error > tolerance)
    {
        std::cout << "Forward SGD Param Verifies FAILED: " << param_error << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else if(!std::isfinite(momentum_buffer_error) || momentum_buffer_error > tolerance)
    {
        std::cout << "Forward SGD Momentum Buffer Verifies FAILED: " << momentum_buffer_error
                  << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward SGD Verifies OK on CPU reference "
                  << "(param_error:" << param_error << " < " << tolerance << ", "
                  << "momentum_buffer_error:" << momentum_buffer_error << " < " << tolerance << ')'
                  << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SGDDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_SGD_DRIVER_HPP
