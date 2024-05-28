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
#ifndef GUARD_MIOPEN_HINGE_EMBEDDING_LOSS_DRIVER_HPP
#define GUARD_MIOPEN_HINGE_EMBEDDING_LOSS_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "miopen/errors.hpp"
#include "miopen/loss/utils.hpp"
#include "miopen/miopen.h"
#include "tensor_driver.hpp"
#include "tensor_view_5d.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#ifndef MLO_HINGE_EMBEDDING_LOSS_MHOST_H_
#define MLO_HINGE_EMBEDDING_LOSS_MHOST_H_

template <class TIO, class TT>
void mloHingeEmbeddingLossFwdRunHost(TIO* input,
                                     miopenTensorDescriptor_t inputDesc,
                                     TT* target,
                                     miopenTensorDescriptor_t targetDesc,
                                     TIO* workspace,
                                     TIO* ref_output,
                                     float margin  = 1,
                                     float divisor = 1)
{
    tensor_view_5d_t input_tv  = get_inner_expanded_tv(miopen::deref(inputDesc));
    tensor_view_5d_t target_tv = get_inner_expanded_tv(miopen::deref(targetDesc));
    size_t size                = miopen::deref(inputDesc).GetElementSize();
    size_t n[5];

    // Compute loss in each elem
    for(size_t idx = 0; idx < size; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, input_tv);

        TIO i = TV_5D_AT(input, n[0], n[1], n[2], n[3], n[4]);
        TT t  = TV_5D_AT(target, n[0], n[1], n[2], n[3], n[4]);

        if(t == 1)
            workspace[idx] = i / divisor;
        else
            workspace[idx] = std::max(0.0f, margin - i) / divisor;
    }

    // Reduce loss
    const int local_size = 256;
    int offset_a         = 0;
    int offset_b         = size;
    size_t _size         = size;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            TIO shared[local_size];
            for(int j = 0; j < local_size; ++j)
                shared[j] = i + j < _size ? workspace[offset_a + i + j] : 0.0f;
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                for(int j = 0; j < offset; ++j)
                    shared[j] += shared[j + offset];
            if(_size <= local_size)
                ref_output[0] = shared[0];
            else
                workspace[offset_b + i / local_size] = shared[0];
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);
}

template <class TIO, class TT>
void mloHingeEmbeddingLossBwdRunHost(TIO* input,
                                     miopenTensorDescriptor_t inputDesc,
                                     TT* target,
                                     miopenTensorDescriptor_t targetDesc,
                                     TIO* doutput,
                                     miopenTensorDescriptor_t doutputDesc,
                                     TIO* dinput,
                                     float margin  = 1,
                                     float divisor = 1)
{
    tensor_view_5d_t input_tv   = get_inner_expanded_tv(miopen::deref(inputDesc));
    tensor_view_5d_t target_tv  = get_inner_expanded_tv(miopen::deref(targetDesc));
    tensor_view_5d_t doutput_tv = get_inner_expanded_tv(miopen::deref(doutputDesc));
    size_t inputSize            = miopen::deref(inputDesc).GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < inputSize; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, input_tv);

        TIO i = TV_5D_AT(input, n[0], n[1], n[2], n[3], n[4]);
        TT t  = TV_5D_AT(target, n[0], n[1], n[2], n[3], n[4]);
        TIO o = TV_5D_AT(doutput, 0, 0, 0, 0, 0);

        if(t == 1)
        {
            dinput[idx] = o / divisor;
        }
        else
        {
            if(margin - i > 0)
                dinput[idx] = -o / divisor;
            else
                dinput[idx] = 0.0f;
        }
    }
}

template <typename TIO, typename TT>
void mloHingeEmbeddingLossUnreducedFwdRunHost(TIO* input,
                                              miopenTensorDescriptor_t inputDesc,
                                              TT* target,
                                              miopenTensorDescriptor_t targetDesc,
                                              TIO* outputhost,
                                              float margin = 1)
{
    tensor_view_5d_t input_tv  = get_inner_expanded_tv(miopen::deref(inputDesc));
    tensor_view_5d_t target_tv = get_inner_expanded_tv(miopen::deref(targetDesc));
    size_t inputSize           = miopen::deref(inputDesc).GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < inputSize; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, input_tv);

        TIO i = TV_5D_AT(input, n[0], n[1], n[2], n[3], n[4]);
        TT t  = TV_5D_AT(target, n[0], n[1], n[2], n[3], n[4]);

        if(t == 1)
            outputhost[idx] = i;
        else
            outputhost[idx] = std::max(0.0f, margin - i);
    }
}

template <class TIO, class TT>
void mloHingeEmbeddingLossUnreducedBwdRunHost(TIO* input,
                                              miopenTensorDescriptor_t inputDesc,
                                              TT* target,
                                              miopenTensorDescriptor_t targetDesc,
                                              TIO* doutput,
                                              miopenTensorDescriptor_t doutputDesc,
                                              TIO* dinput,
                                              float margin = 1)
{
    tensor_view_5d_t input_tv   = get_inner_expanded_tv(miopen::deref(inputDesc));
    tensor_view_5d_t target_tv  = get_inner_expanded_tv(miopen::deref(targetDesc));
    tensor_view_5d_t doutput_tv = get_inner_expanded_tv(miopen::deref(doutputDesc));
    size_t inputSize            = miopen::deref(inputDesc).GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < inputSize; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, input_tv);

        TIO i = TV_5D_AT(input, n[0], n[1], n[2], n[3], n[4]);
        TT t  = TV_5D_AT(target, n[0], n[1], n[2], n[3], n[4]);

        if(t == 1)
        {
            dinput[idx] = TV_5D_AT(doutput, n[0], n[1], n[2], n[3], n[4]);
        }
        else
        {
            if(margin - i > 0)
                dinput[idx] = -TV_5D_AT(doutput, n[0], n[1], n[2], n[3], n[4]);
            else
                dinput[idx] = 0.0f;
        }
    }
}
#endif

template <typename TIO, typename TT>
class HingeEmbeddingLossDriver : public Driver
{
public:
    HingeEmbeddingLossDriver() : Driver()
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
    ~HingeEmbeddingLossDriver() override
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
    std::vector<TT> target;
    std::vector<TIO> output;
    std::vector<TIO> outputHost;
    std::vector<TIO> doutput;
    std::vector<TIO> dinput;
    std::vector<TIO> dinputHost;
    std::vector<TIO> workspace;

    float margin;
    float divisor;
    bool isContiguous;
    std::string reduction;

    size_t workSpaceSizeInBytes;
};

template <typename TIO, typename TT>
int HingeEmbeddingLossDriver<TIO, TT>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
int HingeEmbeddingLossDriver<TIO, TT>::GetandSetData()
{
    std::vector<int> inDim = GetInputTensorLengthsFromCmdLine();
    margin                 = inflags.GetValueDouble("margin");
    isContiguous           = inflags.GetValueInt("is-contiguous") == 1 ? true : false;
    reduction              = inflags.GetValueStr("reduction");

    std::vector<int> inStride = GetTensorStride(inDim);
    if(!isContiguous)
    {
        std::swap(inDim.front(), inDim.back());
    }

    SetTensorNd(inputDesc, inDim, inStride, data_type);
    SetTensorNd(targetDesc, inDim, inStride, miopen_type<TT>{});
    SetTensorNd(doutputDesc, inDim, data_type);
    SetTensorNd(dinputDesc, inDim, data_type);

    if(reduction == "none")
    {
        SetTensorNd(outputDesc, inDim, data_type);
    }
    else
    {
        std::vector<int> outDim(1);
        outDim[0] = 1;
        SetTensorNd(outputDesc, outDim, data_type);
        divisor = 1;
        if(reduction == "mean")
        {
            divisor = miopen::deref(inputDesc).GetElementSize();
        }
    }

    return 0;
}

template <typename TIO, typename TT>
std::vector<int> HingeEmbeddingLossDriver<TIO, TT>::GetTensorStride(std::vector<int> dim)
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

template <typename TIO, typename TT>
int HingeEmbeddingLossDriver<TIO, TT>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward (Default=1)", "int");
    inflags.AddInputFlag("DimLengths",
                         'D',
                         "256,4,1,1,8723",
                         "The dimensional lengths of the input tensor",
                         "string");
    inflags.AddInputFlag("is-contiguous", 'c', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag("reduction", 'R', "none", "reduction (Default='none')", "string");
    inflags.AddInputFlag("margin", 'M', "1", "Margin (Default=1)", "float");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
std::vector<int> HingeEmbeddingLossDriver<TIO, TT>::GetInputTensorLengthsFromCmdLine()
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

template <typename TIO, typename TT>
int HingeEmbeddingLossDriver<TIO, TT>::AllocateBuffersAndCopy()
{
    size_t in_sz     = miopen::deref(inputDesc).GetElementSize();
    size_t target_sz = miopen::deref(targetDesc).GetElementSize();
    size_t out_sz    = miopen::deref(outputDesc).GetElementSize();
    size_t dO_sz     = miopen::deref(doutputDesc).GetElementSize();
    size_t dI_sz     = miopen::deref(dinputDesc).GetElementSize();

    uint32_t ctx = 0;

    input_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(TIO)));
    target_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(TT)));
    output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(TIO)));
    doutput_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, dO_sz, sizeof(TIO)));
    dinput_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, dI_sz, sizeof(TIO)));

    miopenGetHingeEmbeddingLossForwardWorkspaceSize(
        handle, inputDesc, targetDesc, outputDesc, &workSpaceSizeInBytes);
    workspace_dev =
        std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceSizeInBytes / sizeof(TIO), sizeof(TIO)));

    input      = std::vector<TIO>(in_sz, static_cast<TIO>(0));
    target     = std::vector<TT>(target_sz, static_cast<TT>(0));
    output     = std::vector<TIO>(out_sz, static_cast<TIO>(0));
    outputHost = std::vector<TIO>(out_sz, static_cast<TIO>(0));
    doutput    = std::vector<TIO>(dO_sz, static_cast<TIO>(0));
    dinput     = std::vector<TIO>(dI_sz, static_cast<TIO>(0));
    dinputHost = std::vector<TIO>(dI_sz, static_cast<TIO>(0));
    workspace  = std::vector<TIO>(workSpaceSizeInBytes / sizeof(TIO), static_cast<TIO>(0));

    for(int i = 0; i < in_sz; i++)
    {
        input[i] = prng::gen_A_to_B<TIO>(static_cast<TIO>(-2), static_cast<TIO>(2));
        // tar is 1 or -1
        target[i] = prng::gen_A_to_B<TT>(static_cast<TT>(0), static_cast<TT>(2)) * 2 - 1;
    }

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

template <typename TIO, typename TT>
int HingeEmbeddingLossDriver<TIO, TT>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        if(reduction == "none")
        {
            miopenHingeEmbeddingLossUnreducedForward(GetHandle(),
                                                     inputDesc,
                                                     input_dev->GetMem(),
                                                     targetDesc,
                                                     target_dev->GetMem(),
                                                     outputDesc,
                                                     output_dev->GetMem(),
                                                     margin);
        }
        else
        {
            miopenHingeEmbeddingLossForward(GetHandle(),
                                            workspace_dev->GetMem(),
                                            workSpaceSizeInBytes,
                                            inputDesc,
                                            input_dev->GetMem(),
                                            targetDesc,
                                            target_dev->GetMem(),
                                            outputDesc,
                                            output_dev->GetMem(),
                                            margin,
                                            divisor);
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
            std::cout << "Wall-clock Time Hinge Embedding Loss Unreduced Fwd Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Hinge Embedding Loss Unreduced Fwd Elapsed: "
                  << kernel_average_time << " ms" << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
int HingeEmbeddingLossDriver<TIO, TT>::RunForwardCPU()
{
    if(reduction == "none")
    {
        mloHingeEmbeddingLossUnreducedFwdRunHost<TIO, TT>(
            input.data(), inputDesc, target.data(), targetDesc, outputHost.data(), margin);
    }
    else
    {
        mloHingeEmbeddingLossFwdRunHost<TIO, TT>(input.data(),
                                                 inputDesc,
                                                 target.data(),
                                                 targetDesc,
                                                 workspace.data(),
                                                 outputHost.data(),
                                                 margin,
                                                 divisor);
    }

    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
int HingeEmbeddingLossDriver<TIO, TT>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {

        if(reduction == "none")
        {
            miopenHingeEmbeddingLossUnreducedBackward(GetHandle(),
                                                      inputDesc,
                                                      input_dev->GetMem(),
                                                      targetDesc,
                                                      target_dev->GetMem(),
                                                      doutputDesc,
                                                      doutput_dev->GetMem(),
                                                      dinputDesc,
                                                      dinput_dev->GetMem(),
                                                      margin);
        }
        else
        {
            miopenHingeEmbeddingLossBackward(GetHandle(),
                                             inputDesc,
                                             input_dev->GetMem(),
                                             targetDesc,
                                             target_dev->GetMem(),
                                             doutputDesc,
                                             doutput_dev->GetMem(),
                                             dinputDesc,
                                             dinput_dev->GetMem(),
                                             margin,
                                             divisor);
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
            std::cout << "Wall-clock Time Hinge Embedding Loss Unreduced Bwd Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Hinge Embedding Loss Unreduced Bwd Elapsed: "
                  << kernel_average_time << " ms" << std::endl;
    }

    if(dinput_dev->FromGPU(GetStream(), dinput.data()) != 0)
        std::cerr << "Error copying (dI_dev) from GPU, size: " << dinput_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
int HingeEmbeddingLossDriver<TIO, TT>::RunBackwardCPU()
{
    if(reduction == "none")
    {

        mloHingeEmbeddingLossUnreducedBwdRunHost<TIO, TT>(input.data(),
                                                          inputDesc,
                                                          target.data(),
                                                          targetDesc,
                                                          doutput.data(),
                                                          doutputDesc,
                                                          dinputHost.data(),
                                                          margin);
    }
    else
    {
        mloHingeEmbeddingLossBwdRunHost<TIO, TT>(input.data(),
                                                 inputDesc,
                                                 target.data(),
                                                 targetDesc,
                                                 doutput.data(),
                                                 doutputDesc,
                                                 dinputHost.data(),
                                                 margin,
                                                 divisor);
    }

    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
int HingeEmbeddingLossDriver<TIO, TT>::VerifyForward()
{
    RunForwardCPU();
    double tolerance = std::numeric_limits<TIO>::epsilon() * 10;
    auto error       = miopen::rms_range(outputHost, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward " << reduction << " Hinge Embedding Loss FAILED: " << error << " > "
                  << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward " << reduction
                  << " Hinge Embedding Loss Verifies OK on CPU reference (" << error << "< "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
int HingeEmbeddingLossDriver<TIO, TT>::VerifyBackward()
{
    RunBackwardCPU();
    double tolerance = std::numeric_limits<TIO>::epsilon() * 10;
    auto error       = miopen::rms_range(dinputHost, dinput);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward " << reduction << " Hinge Embedding Loss FAILED: " << error << " > "
                  << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backward " << reduction
                  << " Hinge Embedding Loss Verifies OK on CPU reference (" << error << "< "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_HINGE_EMBEDDING_LOSS_DRIVER_HPP
