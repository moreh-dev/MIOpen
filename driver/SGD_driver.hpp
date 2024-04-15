#ifndef GUARD_MIOPEN_SGD_DRIVER_HPP
#define GUARD_MIOPEN_SGD_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <algorithm>
#include <cfloat>
#include <cstddef>
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
                             Tcheck* momentumBufferOut,
                             double lr,
                             double momentum,
                             double dampening,
                             double weight_decay,
                             char nesterov,
                             char momentum_initialized)
{
    auto dims = miopen::deref(paramInputDesc).GetLengths();
    size_t param_size = 0;
    for(size_t dim : dims)
    {
        param_size += dim;
    }

    int32_t ret = 0;

    for(int id = 0; id < param_size; ++id)
    {
        Tcheck param = static_cast<Tcheck>(paramInput[id]);
        Tcheck d_p = static_cast<Tcheck>(grad[id]);

        if (weight_decay != 0)
        {
            d_p += param * weight_decay;
        }

        if (momentum != 0)
        {
            Tcheck momentum_v;
            if (momentum_initialized)
            {
                momentum_v = static_cast<Tcheck>(momentumBufferInput[id]);
                momentum_v = momentum_v * momentum + d_p * (1 - dampening);
            }
            else
            {
                momentum_v = d_p;
            }
            momentumBufferOut[id] = momentum_v;

            if (nesterov)
            {
                d_p = d_p + momentum_v * momentum;
            }
            else
            {
                d_p = momentum_v;
            }
        }

        paramOutputHost[id] = param - lr * d_p;
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

    std::unique_ptr<GPUMem>  param_in_dev;
    std::unique_ptr<GPUMem>  param_out_dev;
    std::unique_ptr<GPUMem>  grad_dev;
    std::unique_ptr<GPUMem>  momentum_buffer_in_dev;
    std::unique_ptr<GPUMem>  momentum_buffer_out_dev;

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

    SetTensorNd(paramInDesc, len, data_type);
    SetTensorNd(paramInDesc, len, data_type);
    SetTensorNd(gradDesc, len, data_type);
    SetTensorNd(momentumBufferInDesc, len, data_type);
    SetTensorNd(momentumBufferOutDesc, len, data_type);

    return 0;
}

template <typename Tgpu, typename Tref>
int SGDDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward SGD (Default=1)", "int");
    inflags.AddInputFlag("batchsize", 'n', "256", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "4", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_d", 'D', "0", "Input Depth (Default=0)", "int");
    inflags.AddInputFlag("in_h", 'H', "0", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "8732", "Input Width (Default=32)", "int");

    inflags.AddInputFlag("lr", 'l', "0.01", "Learning rate (Default=0.01)", "double");
    inflags.AddInputFlag("momentum", 'm', "0.1", "Momentum factor (Default=0.1)", "double");
    inflags.AddInputFlag("dampening", 'd', "0", "Dampening for momentum (Default=0)", "double");
    inflags.AddInputFlag("weight_decay", 'w', "0", "Weight decay (Default=0)", "double");
    inflags.AddInputFlag("nesterov", 'W', "N", "Enables Nesterow momentum (Default=0)", "char");
    inflags.AddInputFlag("momentum_initiated", 'M', "0", "Is momentum initiated (Default=0)", "char");

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
    auto dims = miopen::deref(paramInDesc).GetLengths();
    size_t param_sz = 0;
    for(size_t dim : dims)
    {
        param_sz += dim;
    }

    uint32_t ctx = 0;

    param_in_dev            = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_sz, sizeof(Tgpu)));
    param_out_dev           = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_sz, sizeof(Tgpu)));
    grad_dev                = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_sz, sizeof(Tgpu)));
    momentum_buffer_in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_sz, sizeof(Tgpu)));
    momentum_buffer_out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, param_sz, sizeof(Tgpu)));

    param_in            = std::vector<Tgpu>(param_sz, static_cast<Tgpu>(0));
    param_out           = std::vector<Tgpu>(param_sz, static_cast<Tgpu>(0));
    grad                = std::vector<Tgpu>(param_sz, static_cast<Tgpu>(0));
    momentum_buffer_in  = std::vector<Tgpu>(param_sz, static_cast<Tgpu>(0));
    momentum_buffer_out = std::vector<Tgpu>(param_sz, static_cast<Tgpu>(0));

    param_outhost           = std::vector<Tref>(param_sz, static_cast<Tref>(0));
    momentum_buffer_outhost = std::vector<Tref>(param_sz, static_cast<Tref>(0));

    for(int i = 0; i < param_sz; i++)
    {
        param_in[i]            = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        param_out[i]           = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        grad[i]                = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        momentum_buffer_in[i]  = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        momentum_buffer_out[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    if(param_in_dev->ToGPU(GetStream(), param_in.data()) != 0)
        std::cerr << "Error copying param (in) to GPU, size: " << param_in_dev->GetSize() << std::endl;
    if(param_out_dev->ToGPU(GetStream(), param_out.data()) != 0)
        std::cerr << "Error copying param (out) to GPU, size: " << param_out_dev->GetSize() << std::endl;
    if(grad_dev->ToGPU(GetStream(), grad.data()) != 0)
        std::cerr << "Error copying grad (in) to GPU, size: " << grad_dev->GetSize() << std::endl;
    if(momentum_buffer_in_dev->ToGPU(GetStream(), momentum_buffer_in.data()) != 0)
        std::cerr << "Error copying momentum buffer (in) to GPU, size: " << momentum_buffer_in_dev->GetSize() << std::endl;
    if(momentum_buffer_out_dev->ToGPU(GetStream(), momentum_buffer_out.data()) != 0)
        std::cerr << "Error copying momentum buffer (out) to GPU, size: " << momentum_buffer_out_dev->GetSize() << std::endl;
    
    return miopenStatusSuccess;
}


#endif // GUARD_MIOPEN_SGD_DRIVER_HPP
