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
#ifndef GUARD_MIOPEN_BCELOGITSLOSS_DRIVER_HPP
#define GUARD_MIOPEN_BCELOGITSLOSS_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "miopen/errors.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/ford.hpp>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_5d.hpp>

#include <vector>

#ifndef MLO_BCELOGITSLOSSMHOST_H_
#define MLO_BCELOGITSLOSSMHOST_H_

template <typename Tgpu, typename Tcheck>
int32_t mloBCELogitsLossReducedForwardRunHost(const miopenTensorDescriptor_t iDesc,
                                              const miopenTensorDescriptor_t tDesc,
                                              const miopenTensorDescriptor_t wDesc,
                                              const miopenTensorDescriptor_t pwDesc,
                                              const Tgpu* input,
                                              const Tgpu* target,
                                              const Tgpu* weight,
                                              const Tgpu* pos_weight,
                                              Tcheck* workspacehost,
                                              Tcheck* outputhost,
                                              const float divisor)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto input_tv      = get_inner_expanded_tv(miopen::deref(iDesc));
    auto target_tv     = get_inner_expanded_tv(miopen::deref(tDesc));
    auto weight_tv     = get_inner_expanded_tv(miopen::deref(wDesc));
    auto pos_weight_tv = get_inner_expanded_tv(miopen::deref(pwDesc));

    auto size = miopen::deref(iDesc).GetElementSize();

    /* Phase 1: Calc loss for each element. */
    par_ford(size)([&](size_t i) {
        size_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, input_tv);

        if(n[0] >= input_tv.size[0])
            return;

        size_t c = i % pos_weight_tv.size[0];

        double w  = static_cast<double>(TV_5D_AT(weight, n[0], n[1], n[2], n[3], n[4]));
        double pw = static_cast<double>(TV_1D_AT(pos_weight, c));

        double x = static_cast<double>(TV_5D_AT(input, n[0], n[1], n[2], n[3], n[4]));
        double y = static_cast<double>(TV_5D_AT(target, n[0], n[1], n[2], n[3], n[4]));

        double max_val;
        max_val = (x < 0) ? -x : 0.0f;

        double loss      = w * (((1.0f - y) * x) +
                           (1 + (pw - 1) * y) * (log(exp(-max_val) + exp(-x - max_val)) + max_val));
        workspacehost[i] = static_cast<Tcheck>(loss / divisor);
    });

    /* Phase 2: Reduce */
    const int local_size = 256;
    int offset_a         = 0;
    int offset_b         = size;
    size_t _size         = size;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            double shared[local_size];
            for(int j = 0; j < local_size; ++j)
                shared[j] =
                    static_cast<double>(i + j < _size ? workspacehost[offset_a + i + j] : 0.0f);
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                for(int j = 0; j < offset; ++j)
                    shared[j] += shared[j + offset];
            if(_size <= local_size)
                outputhost[0] = static_cast<Tcheck>(shared[0]);
            else
                workspacehost[offset_b + i / local_size] = static_cast<Tcheck>(shared[0]);
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);

    return miopenStatusSuccess;
}

inline double sigmoid(double x) { return 1.0f / (1.0f + exp(-x)); }

template <typename Tgpu, typename Tcheck>
int32_t mloBCELogitsLossReducedBackwardRunHost(const miopenTensorDescriptor_t iDesc,
                                               const miopenTensorDescriptor_t tDesc,
                                               const miopenTensorDescriptor_t wDesc,
                                               const miopenTensorDescriptor_t pwDesc,
                                               const miopenTensorDescriptor_t diDesc,
                                               const miopenTensorDescriptor_t dtDesc,
                                               const Tgpu* input,
                                               const Tgpu* target,
                                               const Tgpu* weight,
                                               const Tgpu* pos_weight,
                                               const Tgpu* dO,
                                               Tcheck* dI,
                                               Tcheck* dT,
                                               const float divisor)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto input_tv      = get_inner_expanded_tv(miopen::deref(iDesc));
    auto target_tv     = get_inner_expanded_tv(miopen::deref(tDesc));
    auto weight_tv     = get_inner_expanded_tv(miopen::deref(wDesc));
    auto pos_weight_tv = get_inner_expanded_tv(miopen::deref(pwDesc));
    auto ref_dI_tv     = get_inner_expanded_tv(miopen::deref(diDesc));
    auto ref_dT_tv     = get_inner_expanded_tv(miopen::deref(dtDesc));

    auto size = miopen::deref(iDesc).GetElementSize();

    par_ford(size)([&](size_t i) {
        size_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, input_tv);

        if(n[0] >= input_tv.size[0])
            return;

        size_t c = i % pos_weight_tv.size[0];

        double w  = static_cast<double>(TV_5D_AT(weight, n[0], n[1], n[2], n[3], n[4]));
        double pw = static_cast<double>(TV_1D_AT(pos_weight, c));

        double x = static_cast<double>(TV_5D_AT(input, n[0], n[1], n[2], n[3], n[4]));
        double y = static_cast<double>(TV_5D_AT(target, n[0], n[1], n[2], n[3], n[4]));

        {
            size_t dIidx  = TV5D_IDX(ref_dI_tv, n[0], n[1], n[2], n[3], n[4]);
            double result = -w * (pw * y * (1.0f - sigmoid(x)) + (y - 1.0f) * sigmoid(x));
            result *= static_cast<double>(dO[0]) / divisor;
            dI[dIidx] = static_cast<Tcheck>(result);
        }

        {
            size_t dTidx  = TV5D_IDX(ref_dT_tv, n[0], n[1], n[2], n[3], n[4]);
            double result = w * (log(1.0f - sigmoid(x)) - pw * log(sigmoid(x)));
            result *= static_cast<double>(dO[0]) / divisor;
            dT[dTidx] = static_cast<Tcheck>(result);
        }
    });

    return miopenStatusSuccess;
}
#endif // MLO_BCELOGITSLOSSMHOST_H_

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
class BCELogitsLossDriver : public Driver
{
public:
    BCELogitsLossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&wDesc);
        miopenCreateTensorDescriptor(&pwDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&diDesc);
        miopenCreateTensorDescriptor(&dtDesc);
        miopenCreateTensorDescriptor(&doDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~BCELogitsLossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(wDesc);
        miopenDestroyTensorDescriptor(pwDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(diDesc);
        miopenDestroyTensorDescriptor(dtDesc);
        miopenDestroyTensorDescriptor(doDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t wDesc;
    miopenTensorDescriptor_t pwDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t diDesc;
    miopenTensorDescriptor_t dtDesc;
    miopenTensorDescriptor_t doDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> tar_dev;
    std::unique_ptr<GPUMem> weight_dev;
    std::unique_ptr<GPUMem> pos_weight_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> workspace_dev;
    std::unique_ptr<GPUMem> dI_dev;
    std::unique_ptr<GPUMem> dT_dev;
    std::unique_ptr<GPUMem> dO_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> tar;
    std::vector<Tgpu> out;
    std::vector<Tgpu> weight;
    std::vector<Tgpu> pos_weight;
    std::vector<Tgpu> workspace;
    std::vector<Tgpu> dI;
    std::vector<Tgpu> dT;
    std::vector<Tgpu> dO;

    std::vector<Tref> outhost;
    std::vector<Tref> workspacehost;
    std::vector<Tref> dIhost;
    std::vector<Tref> dThost;

    size_t ws_sizeInBytes;

    float divisor;
    bool hasWeight;
    bool hasPosWeight;
};

template <typename Tgpu, typename Tref>
int BCELogitsLossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int BCELogitsLossDriver<Tgpu, Tref>::GetandSetData()
{
    hasWeight      = (inflags.GetValueInt("HasWeight") != 0);
    hasPosWeight   = (inflags.GetValueInt("HasPosWeight") != 0);
    auto reduction = inflags.GetValueStr("Reduction");
    if(reduction != "none" && reduction != "mean" && reduction != "sum")
        return miopenStatusInvalidValue;

    auto length               = GetTensorLengthsFromCmdLine();
    std::vector<int> pwlength = {length.back()};
    auto in_strides           = GetStrides(length, 1);
    auto tar_strides          = GetStrides(length, inflags.GetValueInt("Contiguous"));

    SetTensorNd(inputDesc, length, in_strides, data_type);
    SetTensorNd(targetDesc, length, tar_strides, data_type);

    if(reduction == "none")
    {
        divisor = std::numeric_limits<float>::quiet_NaN();
        SetTensorNd(outputDesc, length, in_strides, data_type);
    }
    else
    {
        std::vector<int> out_lens = {1};
        SetTensorNd(outputDesc, out_lens, data_type);
        if(reduction == "sum")
            divisor = 1;
        if(reduction == "mean")
            divisor = miopen::deref(inputDesc).GetElementSize();
    }

    SetTensorNd(diDesc, length, in_strides, data_type);
    SetTensorNd(dtDesc, length, tar_strides, data_type);
    SetTensorNd(wDesc, length, data_type);
    SetTensorNd(pwDesc, pwlength, data_type);

    if(reduction == "none")
    {
        divisor = std::numeric_limits<float>::quiet_NaN();
        SetTensorNd(doDesc, length, in_strides, data_type);
    }
    else
    {
        std::vector<int> out_lens = {1};
        SetTensorNd(doDesc, out_lens, data_type);
        if(reduction == "sum")
            divisor = 1;
        if(reduction == "mean")
            divisor = miopen::deref(inputDesc).GetElementSize();
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int BCELogitsLossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward BCELogitsLoss (Default=1)", "int");
    inflags.AddInputFlag("DimLengths",
                         'D',
                         "256,4,1,1,8723",
                         "The dimensional lengths of the input tensor",
                         "string");
    inflags.AddInputFlag("Contiguous",
                         'C',
                         "1",
                         "Is input tensor contiguous? (Default=1 for contiguous tensor)",
                         "int");
    inflags.AddInputFlag("Reduction",
                         'R',
                         "none",
                         "Specifies the reduction to apply to the output ('none'|'mean'|'sum') "
                         "(Default=none to indicate no reduction)",
                         "string");
    inflags.AddInputFlag("HasWeight",
                         'W',
                         "0",
                         "Applies rescaling weight given to the loss of each batch element. "
                         "(Default=0 to indicate no weight used)",
                         "int");
    inflags.AddInputFlag("HasPosWeight",
                         'P',
                         "0",
                         "Applies a weight of positive examples to be broadcasted with target. "
                         "(Default=0 to indicate no pos weight used)",
                         "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "0", "Verify Each Layer (Default=0)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> BCELogitsLossDriver<Tgpu, Tref>::GetTensorLengthsFromCmdLine()
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
int BCELogitsLossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz  = GetTensorSize(inputDesc);
    size_t tar_sz = GetTensorSize(targetDesc);
    size_t w_sz   = GetTensorSize(wDesc);
    size_t pw_sz  = GetTensorSize(pwDesc);
    size_t out_sz = GetTensorSize(outputDesc);

    if(!std::isnan(divisor))
    {
        miopenGetBCELogitsLossReducedForwardWorkspaceSize(
            GetHandle(), inputDesc, targetDesc, wDesc, pwDesc, outputDesc, &ws_sizeInBytes);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            return miopenStatusAllocFailed;
    }
    else
        ws_sizeInBytes = 0;
    size_t ws_sz = ws_sizeInBytes / sizeof(Tgpu);

    uint32_t ctx = 0;

    in_dev         = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    tar_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, tar_sz, sizeof(Tgpu)));
    weight_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, w_sz, sizeof(Tgpu)));
    pos_weight_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, pw_sz, sizeof(Tgpu)));
    out_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    workspace_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));
    dI_dev         = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    dT_dev         = std::unique_ptr<GPUMem>(new GPUMem(ctx, tar_sz, sizeof(Tgpu)));
    dO_dev         = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    in         = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    tar        = std::vector<Tgpu>(tar_sz, static_cast<Tgpu>(0));
    weight     = std::vector<Tgpu>(w_sz, static_cast<Tgpu>(0));
    pos_weight = std::vector<Tgpu>(pw_sz, static_cast<Tgpu>(0));
    out        = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    workspace  = std::vector<Tgpu>(ws_sz, static_cast<Tgpu>(0));
    dI         = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    dT         = std::vector<Tgpu>(tar_sz, static_cast<Tgpu>(0));
    dO         = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

    outhost       = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    workspacehost = std::vector<Tref>(ws_sz, static_cast<Tref>(0));
    dIhost        = std::vector<Tref>(in_sz, static_cast<Tref>(0));
    dThost        = std::vector<Tref>(tar_sz, static_cast<Tref>(0));

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.001f), static_cast<Tgpu>(0.99f));
    }

    for(int i = 0; i < tar_sz; i++)
    {
        tar[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.001f), static_cast<Tgpu>(0.99f));
    }

    fill(weight.begin(), weight.end(), static_cast<Tgpu>(1.0f));
    fill(pos_weight.begin(), pos_weight.end(), static_cast<Tgpu>(1.0f));

    fill(out.begin(), out.end(), static_cast<Tgpu>(0.0f));

    fill(dO.begin(), dO.end(), static_cast<Tgpu>(0.5f));

    if(in_dev->ToGPU(GetStream(), in.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;

    if(tar_dev->ToGPU(GetStream(), tar.data()) != 0)
        std::cerr << "Error copying (tar) to GPU, size: " << tar_dev->GetSize() << std::endl;

    if(weight_dev->ToGPU(GetStream(), weight.data()) != 0)
        std::cerr << "Error copying (weight) to GPU, size: " << weight_dev->GetSize() << std::endl;

    if(pos_weight_dev->ToGPU(GetStream(), pos_weight.data()) != 0)
        std::cerr << "Error copying (pos_weight) to GPU, size: " << pos_weight_dev->GetSize()
                  << std::endl;

    if(dO_dev->ToGPU(GetStream(), dO.data()) != 0)
        std::cerr << "Error copying (out grad) to GPU, size: " << dO_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int BCELogitsLossDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        if(!std::isnan(divisor))
        {
            miopenBCELogitsLossReducedForward(GetHandle(),
                                              workspace_dev->GetMem(),
                                              ws_sizeInBytes,
                                              inputDesc,
                                              in_dev->GetMem(),
                                              targetDesc,
                                              tar_dev->GetMem(),
                                              wDesc,
                                              (hasWeight ? weight_dev->GetMem() : nullptr),
                                              pwDesc,
                                              (hasPosWeight ? pos_weight_dev->GetMem() : nullptr),
                                              outputDesc,
                                              out_dev->GetMem(),
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
            std::cout << "Wall-clock Time Forward BCELogitsLoss Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward BCELogitsLoss Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int BCELogitsLossDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(!std::isnan(divisor))
    {
        mloBCELogitsLossReducedForwardRunHost<Tgpu, Tref>(inputDesc,
                                                          targetDesc,
                                                          wDesc,
                                                          pwDesc,
                                                          in.data(),
                                                          tar.data(),
                                                          weight.data(),
                                                          pos_weight.data(),
                                                          workspacehost.data(),
                                                          outhost.data(),
                                                          divisor);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int BCELogitsLossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopen::deref(GetHandle()).ResetKernelTime();
        if(!std::isnan(divisor))
        {
            miopenBCELogitsLossReducedBackward(GetHandle(),
                                               inputDesc,
                                               in_dev->GetMem(),
                                               targetDesc,
                                               tar_dev->GetMem(),
                                               wDesc,
                                               (hasWeight ? weight_dev->GetMem() : nullptr),
                                               pwDesc,
                                               (hasPosWeight ? pos_weight_dev->GetMem() : nullptr),
                                               doDesc,
                                               dO_dev->GetMem(),
                                               diDesc,
                                               dI_dev->GetMem(),
                                               dtDesc,
                                               dT_dev->GetMem(),
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
            std::cout << "Wall-clock Time Backward BCELogitsLoss Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward BCELogitsLoss Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(dI_dev->FromGPU(GetStream(), dI.data()) != 0)
        std::cerr << "Error copying (dI_dev) from GPU, size: " << dI_dev->GetSize() << std::endl;
    if(dT_dev->FromGPU(GetStream(), dT.data()) != 0)
        std::cerr << "Error copying (dT_dev) from GPU, size: " << dT_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int BCELogitsLossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    if(!std::isnan(divisor))
    {
        mloBCELogitsLossReducedBackwardRunHost<Tgpu, Tref>(inputDesc,
                                                           targetDesc,
                                                           wDesc,
                                                           pwDesc,
                                                           diDesc,
                                                           dtDesc,
                                                           in.data(),
                                                           tar.data(),
                                                           weight.data(),
                                                           pos_weight.data(),
                                                           dO.data(),
                                                           dIhost.data(),
                                                           dThost.data(),
                                                           divisor);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref BCELogitsLossDriver<Tgpu, Tref>::GetTolerance()
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
int BCELogitsLossDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward BCELogitsLoss FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward BCELogitsLoss Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int BCELogitsLossDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error_dI        = miopen::rms_range(dIhost, dI);
    auto error_dT        = miopen::rms_range(dThost, dT);

    if(!std::isfinite(error_dI) || error_dI > tolerance || !std::isfinite(error_dT) ||
       error_dT > tolerance)
    {
        std::cout << "Backward BCELogitsLoss FAILED: {" << error_dI << "," << error_dT << "} > "
                  << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backward BCELogitsLoss Verifies OK on CPU reference ({" << error_dI << ","
                  << error_dT << "} < " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_BCELOGITSLOSS_DRIVER_HPP
