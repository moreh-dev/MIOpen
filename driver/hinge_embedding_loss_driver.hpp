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

inline tensor_view_5d_t get_inner_expanded_tv(const miopen::TensorDescriptor Desc)
{
    auto dims    = Desc.GetLengths();
    auto strides = Desc.GetStrides();

    tensor_view_5d_t tv_5d;
    for(size_t i = 0; i < strides.size(); ++i)
    {
        tv_5d.stride[i] = strides[i];
        tv_5d.size[i]   = dims[i];
    }
    auto rest = strides.size();
    for(size_t j = rest; j < 5; ++j)
    {
        tv_5d.stride[j] = (rest == 0 ? 1 : strides[rest - 1]);
        tv_5d.size[j]   = 1;
    }
    return tv_5d;
}

template <class TIO, class TT>
void mloHingeEmbeddingLossFwdRunHost(TIO* I,
                                     miopenTensorDescriptor_t inputDesc,
                                     TT* T,
                                     miopenTensorDescriptor_t targetDesc,
                                     TIO* workspace,
                                     TIO* ref_output,
                                     float margin  = 1,
                                     float divisor = 1)
{
    tensor_view_5d_t I_tv = get_inner_expanded_tv(miopen::deref(inputDesc));
    tensor_view_5d_t T_tv = get_inner_expanded_tv(miopen::deref(targetDesc));
    size_t size           = miopen::deref(inputDesc).GetElementSize();
    size_t n[5];

    // Compute loss in each elem
    for(size_t idx = 0; idx < size; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, I_tv);

        TIO i = TV_5D_AT(I, n[0], n[1], n[2], n[3], n[4]);
        TT t  = TV_5D_AT(T, n[0], n[1], n[2], n[3], n[4]);

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
void mloHingeEmbeddingLossBwdRunHost(TIO* I,
                                     miopenTensorDescriptor_t inputDesc,
                                     TT* T,
                                     miopenTensorDescriptor_t targetDesc,
                                     TIO* dO,
                                     miopenTensorDescriptor_t dODesc,
                                     TIO* dI,
                                     miopenTensorDescriptor_t dIDesc,
                                     float margin  = 1,
                                     float divisor = 1)
{
    tensor_view_5d_t I_tv  = get_inner_expanded_tv(miopen::deref(inputDesc));
    tensor_view_5d_t T_tv  = get_inner_expanded_tv(miopen::deref(targetDesc));
    tensor_view_5d_t dO_tv = get_inner_expanded_tv(miopen::deref(dODesc));
    size_t inputSize       = miopen::deref(inputDesc).GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < inputSize; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, I_tv);

        TIO i = TV_5D_AT(I, n[0], n[1], n[2], n[3], n[4]);
        TT t  = TV_5D_AT(T, n[0], n[1], n[2], n[3], n[4]);
        TIO o = TV_5D_AT(dO, 0, 0, 0, 0, 0);

        if(t == 1)
        {
            dI[idx] = o / divisor;
        }
        else
        {
            if(margin - i > 0)
                dI[idx] = -o / divisor;
            else
                dI[idx] = 0.0f;
        }
    }
}

template <typename TIO, typename TT>
void mloHingeEmbeddingLossUnreducedFwdRunHost(TIO* I,
                                              miopenTensorDescriptor_t inputDesc,
                                              TT* T,
                                              miopenTensorDescriptor_t targetDesc,
                                              TIO* outputhost,
                                              float margin = 1)
{
    tensor_view_5d_t I_tv = get_inner_expanded_tv(miopen::deref(inputDesc));
    tensor_view_5d_t T_tv = get_inner_expanded_tv(miopen::deref(targetDesc));
    size_t inputSize      = miopen::deref(inputDesc).GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < inputSize; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, I_tv);

        TIO i = TV_5D_AT(I, n[0], n[1], n[2], n[3], n[4]);
        TT t  = TV_5D_AT(T, n[0], n[1], n[2], n[3], n[4]);

        if(t == 1)
            outputhost[idx] = i;
        else
            outputhost[idx] = std::max(0.0f, margin - i);
    }
}

template <class TIO, class TT>
void mloHingeEmbeddingLossUnreducedBwdRunHost(TIO* I,
                                              miopenTensorDescriptor_t inputDesc,
                                              TT* T,
                                              miopenTensorDescriptor_t targetDesc,
                                              TIO* dO,
                                              miopenTensorDescriptor_t dODesc,
                                              TIO* dI,
                                              miopenTensorDescriptor_t dIDesc,
                                              float margin = 1)
{
    tensor_view_5d_t I_tv  = get_inner_expanded_tv(miopen::deref(inputDesc));
    tensor_view_5d_t T_tv  = get_inner_expanded_tv(miopen::deref(targetDesc));
    tensor_view_5d_t dO_tv = get_inner_expanded_tv(miopen::deref(dODesc));
    size_t inputSize       = miopen::deref(inputDesc).GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < inputSize; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, I_tv);

        TIO i = TV_5D_AT(I, n[0], n[1], n[2], n[3], n[4]);
        TT t  = TV_5D_AT(T, n[0], n[1], n[2], n[3], n[4]);

        if(t == 1)
        {
            dI[idx] = TV_5D_AT(dO, n[0], n[1], n[2], n[3], n[4]);
        }
        else
        {
            if(margin - i > 0)
                dI[idx] = -TV_5D_AT(dO, n[0], n[1], n[2], n[3], n[4]);
            else
                dI[idx] = 0.0f;
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
        miopenCreateTensorDescriptor(&dODesc);
        miopenCreateTensorDescriptor(&dIDesc);

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
        miopenDestroyTensorDescriptor(dODesc);
        miopenDestroyTensorDescriptor(dIDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t dODesc;
    miopenTensorDescriptor_t dIDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> tar_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> dO_dev;
    std::unique_ptr<GPUMem> dI_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<TIO> in;
    std::vector<TT> tar;
    std::vector<TIO> out;
    std::vector<TIO> outHost;
    std::vector<TIO> dO;
    std::vector<TIO> dI;
    std::vector<TIO> dIHost;
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
    SetTensorNd(dODesc, inDim, data_type);
    SetTensorNd(dIDesc, inDim, data_type);

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
    size_t dO_sz     = miopen::deref(dODesc).GetElementSize();
    size_t dI_sz     = miopen::deref(dIDesc).GetElementSize();

    uint32_t ctx = 0;

    in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(TIO)));
    tar_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(TT)));
    out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(TIO)));
    dO_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, dO_sz, sizeof(TIO)));
    dI_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, dI_sz, sizeof(TIO)));

    miopenGetHingeEmbeddingLossForwardWorkspaceSize(
        handle, inputDesc, targetDesc, outputDesc, &workSpaceSizeInBytes);
    workspace_dev =
        std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceSizeInBytes / sizeof(TIO), sizeof(TIO)));

    in        = std::vector<TIO>(in_sz, static_cast<TIO>(0));
    tar       = std::vector<TT>(target_sz, static_cast<TT>(0));
    out       = std::vector<TIO>(out_sz, static_cast<TIO>(0));
    outHost   = std::vector<TIO>(out_sz, static_cast<TIO>(0));
    dO        = std::vector<TIO>(dO_sz, static_cast<TIO>(0));
    dI        = std::vector<TIO>(dI_sz, static_cast<TIO>(0));
    dIHost    = std::vector<TIO>(dI_sz, static_cast<TIO>(0));
    workspace = std::vector<TIO>(workSpaceSizeInBytes / sizeof(TIO), static_cast<TIO>(0));

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<TIO>(static_cast<TIO>(-2), static_cast<TIO>(2));
        // tar is 1 or -1
        tar[i] = prng::gen_A_to_B<TT>(static_cast<TT>(0), static_cast<TT>(2)) * 2 - 1;
    }

    for(int i = 0; i < dO_sz; ++i)
    {
        dO[i] = prng::gen_A_to_B<TIO>(static_cast<TIO>(-2), static_cast<TIO>(2));
    }

    fill(out.begin(), out.end(), static_cast<TIO>(0));
    fill(dI.begin(), dI.end(), static_cast<TIO>(0));

    if(in_dev->ToGPU(GetStream(), in.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;

    if(tar_dev->ToGPU(GetStream(), tar.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << tar_dev->GetSize() << std::endl;

    if(out_dev->ToGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;

    if(dO_dev->ToGPU(GetStream(), dO.data()) != 0)
        std::cerr << "Error copying (dO) to GPU, size: " << dO_dev->GetSize() << std::endl;

    if(dI_dev->ToGPU(GetStream(), dI.data()) != 0)
        std::cerr << "Error copying (dI) to GPU, size: " << dI_dev->GetSize() << std::endl;

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
                                                     in_dev->GetMem(),
                                                     targetDesc,
                                                     tar_dev->GetMem(),
                                                     outputDesc,
                                                     out_dev->GetMem(),
                                                     margin);
        }
        else
        {
            miopenHingeEmbeddingLossForward(GetHandle(),
                                            workspace_dev->GetMem(),
                                            workSpaceSizeInBytes,
                                            inputDesc,
                                            in_dev->GetMem(),
                                            targetDesc,
                                            tar_dev->GetMem(),
                                            outputDesc,
                                            out_dev->GetMem(),
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
                      << t.gettime_ms() / iter << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Hinge Embedding Loss Unreduced Fwd Elapsed: "
                  << kernel_average_time << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
int HingeEmbeddingLossDriver<TIO, TT>::RunForwardCPU()
{
    if(reduction == "none")
    {
        mloHingeEmbeddingLossUnreducedFwdRunHost<TIO, TT>(
            in.data(), inputDesc, tar.data(), targetDesc, outHost.data(), margin);
    }
    else
    {
        mloHingeEmbeddingLossFwdRunHost<TIO, TT>(in.data(),
                                                 inputDesc,
                                                 tar.data(),
                                                 targetDesc,
                                                 workspace.data(),
                                                 outHost.data(),
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
                                                      in_dev->GetMem(),
                                                      targetDesc,
                                                      tar_dev->GetMem(),
                                                      dODesc,
                                                      dO_dev->GetMem(),
                                                      dIDesc,
                                                      dI_dev->GetMem(),
                                                      margin);
        }
        else
        {
            miopenHingeEmbeddingLossBackward(GetHandle(),
                                             inputDesc,
                                             in_dev->GetMem(),
                                             targetDesc,
                                             tar_dev->GetMem(),
                                             dODesc,
                                             dO_dev->GetMem(),
                                             dIDesc,
                                             dI_dev->GetMem(),
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
                      << t.gettime_ms() / iter << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Hinge Embedding Loss Unreduced Bwd Elapsed: "
                  << kernel_average_time << " ms\n";
    }

    if(dI_dev->FromGPU(GetStream(), dI.data()) != 0)
        std::cerr << "Error copying (dI_dev) from GPU, size: " << dI_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
int HingeEmbeddingLossDriver<TIO, TT>::RunBackwardCPU()
{
    if(reduction == "none")
    {

        mloHingeEmbeddingLossUnreducedBwdRunHost<TIO, TT>(in.data(),
                                                          inputDesc,
                                                          tar.data(),
                                                          targetDesc,
                                                          dO.data(),
                                                          dODesc,
                                                          dIHost.data(),
                                                          dIDesc,
                                                          margin);
    }
    else
    {
        mloHingeEmbeddingLossBwdRunHost<TIO, TT>(in.data(),
                                                 inputDesc,
                                                 tar.data(),
                                                 targetDesc,
                                                 dO.data(),
                                                 dODesc,
                                                 dIHost.data(),
                                                 dIDesc,
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
    auto error       = miopen::rms_range(outHost, out);

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
    auto error       = miopen::rms_range(dIHost, dI);

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
