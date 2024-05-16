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
#include "tensor_driver.hpp"
#include "tensor_view_5d.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <memory>
// #include <miopen/miopen.h>
// #include <miopen/tensor.hpp>
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

template <typename TIO, typename TT>
void mloHingeEmbeddingLossRunHost(TIO* I,
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

    TIO GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~HingeEmbeddingLossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> tar_dev;
    std::unique_ptr<GPUMem> out_dev;

    std::vector<TIO> in;
    std::vector<TT> tar;
    std::vector<TIO> out;
    std::vector<TIO> outhost;

    float margin;
    bool isContiguous;
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
    std::vector<int> inDim    = GetInputTensorLengthsFromCmdLine();
    margin                    = inflags.GetValueDouble("margin");
    isContiguous              = inflags.GetValueInt("is-contiguous") == 1 ? true : false;
    std::vector<int> inStride = GetTensorStride(inDim);
    if(!isContiguous)
    {
        std::swap(inDim.front(), inDim.back());
    }

    SetTensorNd(inputDesc, inDim, inStride, data_type);
    SetTensorNd(targetDesc, inDim, inStride, miopen_type<TT>{});
    SetTensorNd(outputDesc, inDim, data_type);

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
    inflags.AddInputFlag("margin", 'R', "1", "Margin (Default=1)", "float");
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

    uint32_t ctx = 0;

    in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(TIO)));
    tar_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(TT)));
    out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(TIO)));

    in      = std::vector<TIO>(in_sz, static_cast<TIO>(0));
    tar     = std::vector<TT>(target_sz, static_cast<TT>(0));
    out     = std::vector<TIO>(out_sz, static_cast<TIO>(0));
    outhost = std::vector<TIO>(out_sz, static_cast<TIO>(0));

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<TIO>(static_cast<TIO>(-2), static_cast<TIO>(2));
        // tar is 1 or -1
        tar[i] = prng::gen_A_to_B<TT>(static_cast<TT>(0), static_cast<TT>(2)) * 2 - 1;
    }

    fill(out.begin(), out.end(), static_cast<TIO>(0));

    if(in_dev->ToGPU(GetStream(), in.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;

    if(tar_dev->ToGPU(GetStream(), tar.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << tar_dev->GetSize() << std::endl;

    if(out_dev->ToGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;

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
        miopenHingeEmbeddingLossUnreducedForward(GetHandle(),
                                                 inputDesc,
                                                 in_dev->GetMem(),
                                                 targetDesc,
                                                 tar_dev->GetMem(),
                                                 outputDesc,
                                                 out_dev->GetMem(),
                                                 margin);

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
            std::cout << "Wall-clock Time Hinge Embedding Loss Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Hinge Embedding Loss Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
int HingeEmbeddingLossDriver<TIO, TT>::RunForwardCPU()
{
    mloHingeEmbeddingLossRunHost<TIO, TT>(
        in.data(), inputDesc, tar.data(), targetDesc, outhost.data(), margin);

    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
int HingeEmbeddingLossDriver<TIO, TT>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
int HingeEmbeddingLossDriver<TIO, TT>::VerifyForward()
{
    RunForwardCPU();
    double tolerance = std::is_same<TIO, float>::value ? 1.5e-6 : 8.2e-3;

    // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
    if(std::is_same<TIO, bfloat16>::value)
        tolerance *= 8.0;
    auto error = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Hinge Embedding Loss FAILED: " << error << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Hinge Embedding Loss Verifies OK on CPU reference (" << error << "< "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename TIO, typename TT>
int HingeEmbeddingLossDriver<TIO, TT>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_HINGE_EMBEDDING_LOSS_DRIVER_HPP
