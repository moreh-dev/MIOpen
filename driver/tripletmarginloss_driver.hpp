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

#pragma once

#include "driver.hpp"
#include "mloTripletMarginLossHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/ford.hpp>
#include <../test/verify.hpp>

#include <miopen/miopen.h>

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
class TripletMarginLossDriver : public Driver
{
public:
    TripletMarginLossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&anchorDesc);
        miopenCreateTensorDescriptor(&positiveDesc);
        miopenCreateTensorDescriptor(&negativeDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&dODesc);
        miopenCreateTensorDescriptor(&dADesc);
        miopenCreateTensorDescriptor(&dPDesc);
        miopenCreateTensorDescriptor(&dNDesc);

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
    ~TripletMarginLossDriver() override
    {
        miopenDestroyTensorDescriptor(anchorDesc);
        miopenDestroyTensorDescriptor(positiveDesc);
        miopenDestroyTensorDescriptor(negativeDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(dODesc);
        miopenDestroyTensorDescriptor(dADesc);
        miopenDestroyTensorDescriptor(dPDesc);
        miopenDestroyTensorDescriptor(dNDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t anchorDesc;
    miopenTensorDescriptor_t positiveDesc;
    miopenTensorDescriptor_t negativeDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t dODesc;
    miopenTensorDescriptor_t dADesc;
    miopenTensorDescriptor_t dPDesc;
    miopenTensorDescriptor_t dNDesc;

    std::unique_ptr<GPUMem> anchor_dev;
    std::unique_ptr<GPUMem> positive_dev;
    std::unique_ptr<GPUMem> negative_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> dO_dev;
    std::unique_ptr<GPUMem> dA_dev;
    std::unique_ptr<GPUMem> dP_dev;
    std::unique_ptr<GPUMem> dN_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> anchor;
    std::vector<Tgpu> positive;
    std::vector<Tgpu> negative;
    std::vector<Tgpu> out;
    std::vector<Tgpu> dO;
    std::vector<Tgpu> dA;
    std::vector<Tgpu> dP;
    std::vector<Tgpu> dN;

    std::vector<Tref> outhost;
    std::vector<Tref> dAhost;
    std::vector<Tref> dPhost;
    std::vector<Tref> dNhost;

    size_t ws_sizeInBytes;

    float margin;
    int p;
    float eps;
    bool swap;
    float divisor;
};

template <typename Tgpu, typename Tref>
int TripletMarginLossDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TripletMarginLossDriver<Tgpu, Tref>::GetandSetData()
{
    auto reduction = inflags.GetValueStr("Reduction");
    if(reduction != "none" && reduction != "mean" && reduction != "sum")
    {
        std::cout << R"(Reduction must be "none"|"mean"|"sum")";
        return miopenStatusInvalidValue;
    }

    margin = inflags.GetValueDouble("Margin");
    p      = inflags.GetValueInt("P");
    eps    = inflags.GetValueDouble("Eps");
    swap   = (inflags.GetValueInt("Swap") != 0);

    auto length           = GetTensorLengthsFromCmdLine();
    auto anchor_strides   = GetStrides(length, inflags.GetValueInt("Contiguous"));
    auto positive_strides = GetStrides(length, 1);
    auto negative_strides = GetStrides(length, 1);

    SetTensorNd(anchorDesc, length, anchor_strides, data_type);
    SetTensorNd(positiveDesc, length, positive_strides, data_type);
    SetTensorNd(negativeDesc, length, negative_strides, data_type);
    SetTensorNd(dADesc, length, anchor_strides, data_type);
    SetTensorNd(dPDesc, length, positive_strides, data_type);
    SetTensorNd(dNDesc, length, negative_strides, data_type);

    if(reduction == "none")
    {
        std::vector<int> out_lens = {length[0]};
        SetTensorNd(outputDesc, out_lens, data_type);
        SetTensorNd(dODesc, out_lens, data_type);
        divisor = std::numeric_limits<float>::quiet_NaN();
    }
    else
    {
        std::vector<int> out_lens = {1};
        SetTensorNd(outputDesc, out_lens, data_type);
        SetTensorNd(dODesc, out_lens, data_type);
        if(reduction == "sum")
            divisor = 1;
        if(reduction == "mean")
            divisor = length[0];
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TripletMarginLossDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward TripletMarginLoss (Default=1)", "int");
    inflags.AddInputFlag("NBatch", 'N', "256", "The number of batches", "int");
    inflags.AddInputFlag("Dim", 'D', "256", "The vector dimension", "int");
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
    inflags.AddInputFlag("Margin", 'm', "1", "(Default=1)", "double");
    inflags.AddInputFlag("P", 'p', "2", "The norm degree for pairwise distance (Default=1)", "int");
    inflags.AddInputFlag("Eps",
                         'e',
                         "0.0000001",
                         "Small constant for numerical stability (Default=0.0000001)",
                         "double");
    inflags.AddInputFlag(
        "Swap",
        's',
        "0",
        "The distance swap is described in detail in the paper Learning shallow convolutional "
        "feature descriptors with triplet losses by V. Balntas, E. Riba et al (Default=0 for "
        "False)",
        "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "0", "Verify Each Layer (Default=0)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> TripletMarginLossDriver<Tgpu, Tref>::GetTensorLengthsFromCmdLine()
{
    auto N = inflags.GetValueInt("NBatch");
    auto D = inflags.GetValueInt("Dim");
    if(N <= 0 || D <= 0)
        std::cerr << "NBatch and Dim must be positive integer\n" << std::endl;
    return {N, D};
}

template <typename Tgpu, typename Tref>
int TripletMarginLossDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t anchor_sz   = GetTensorSize(anchorDesc);
    size_t positive_sz = GetTensorSize(positiveDesc);
    size_t negative_sz = GetTensorSize(negativeDesc);
    size_t out_sz      = GetTensorSize(outputDesc);

    miopenGetTripletMarginLossForwardWorkspaceSize(
        GetHandle(), anchorDesc, outputDesc, &ws_sizeInBytes);

    if(ws_sizeInBytes == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;

    uint32_t ctx = 0;

    anchor_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, anchor_sz, sizeof(Tgpu)));
    positive_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, positive_sz, sizeof(Tgpu)));
    negative_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, negative_sz, sizeof(Tgpu)));
    out_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    dO_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    dA_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, anchor_sz, sizeof(Tgpu)));
    dP_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, positive_sz, sizeof(Tgpu)));
    dN_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, negative_sz, sizeof(Tgpu)));
    workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));

    anchor   = std::vector<Tgpu>(anchor_sz);
    positive = std::vector<Tgpu>(positive_sz);
    negative = std::vector<Tgpu>(negative_sz);
    out      = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    dO       = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0.5));
    dA       = std::vector<Tgpu>(anchor_sz, static_cast<Tgpu>(0));
    dP       = std::vector<Tgpu>(positive_sz, static_cast<Tgpu>(0));
    dN       = std::vector<Tgpu>(negative_sz, static_cast<Tgpu>(0));

    outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    dAhost  = std::vector<Tref>(anchor_sz, static_cast<Tref>(0));
    dPhost  = std::vector<Tref>(positive_sz, static_cast<Tref>(0));
    dNhost  = std::vector<Tref>(negative_sz, static_cast<Tref>(0));

    for(int i = 0; i < anchor_sz; i++)
        anchor[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(0.2));

    for(int i = 0; i < positive_sz; i++)
        positive[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.1), static_cast<Tgpu>(0.3));

    for(int i = 0; i < negative_sz; i++)
        negative[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.2), static_cast<Tgpu>(0.4));

    if(anchor_dev->ToGPU(GetStream(), anchor.data()) != 0)
        std::cerr << "Error copying (anchor) to GPU, size: " << anchor_dev->GetSize() << std::endl;

    if(positive_dev->ToGPU(GetStream(), positive.data()) != 0)
        std::cerr << "Error copying (positive) to GPU, size: " << positive_dev->GetSize()
                  << std::endl;

    if(negative_dev->ToGPU(GetStream(), negative.data()) != 0)
        std::cerr << "Error copying (negative) to GPU, size: " << negative_dev->GetSize()
                  << std::endl;

    if(out_dev->ToGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;

    if(dO_dev->ToGPU(GetStream(), dO.data()) != 0)
        std::cerr << "Error copying (out grad) to GPU, size: " << dO_dev->GetSize() << std::endl;

    if(dA_dev->ToGPU(GetStream(), dA.data()) != 0)
        std::cerr << "Error copying (anchor grad) to GPU, size: " << dA_dev->GetSize() << std::endl;

    if(dP_dev->ToGPU(GetStream(), dP.data()) != 0)
        std::cerr << "Error copying (positive grad) to GPU, size: " << dP_dev->GetSize()
                  << std::endl;

    if(dN_dev->ToGPU(GetStream(), dN.data()) != 0)
        std::cerr << "Error copying (negative grad) to GPU, size: " << dN_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TripletMarginLossDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenTripletMarginLossForward(GetHandle(),
                                       workspace_dev->GetMem(),
                                       ws_sizeInBytes,
                                       anchorDesc,
                                       anchor_dev->GetMem(),
                                       positiveDesc,
                                       positive_dev->GetMem(),
                                       negativeDesc,
                                       negative_dev->GetMem(),
                                       outputDesc,
                                       out_dev->GetMem(),
                                       margin,
                                       p,
                                       eps,
                                       swap,
                                       divisor);

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
            std::cout << "Wall-clock Time Forward TripletMarginLoss Elapsed: "
                      << t.gettime_ms() / iter << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward TripletMarginLoss Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TripletMarginLossDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(std::isnan(divisor))
    {
        mloTripletMarginLossUnreducedForwardRunHost<Tgpu, Tref>(anchorDesc,
                                                                positiveDesc,
                                                                negativeDesc,
                                                                outputDesc,
                                                                anchor.data(),
                                                                positive.data(),
                                                                negative.data(),
                                                                outhost.data(),
                                                                margin,
                                                                p,
                                                                eps,
                                                                swap);
    }
    else
    {
        mloTripletMarginLossForwardRunHost<Tgpu, Tref>(anchorDesc,
                                                       positiveDesc,
                                                       negativeDesc,
                                                       anchor.data(),
                                                       positive.data(),
                                                       negative.data(),
                                                       outhost.data(),
                                                       margin,
                                                       p,
                                                       eps,
                                                       swap,
                                                       divisor);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TripletMarginLossDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenTripletMarginLossBackward(GetHandle(),
                                        workspace_dev->GetMem(),
                                        ws_sizeInBytes,
                                        anchorDesc,
                                        anchor_dev->GetMem(),
                                        positiveDesc,
                                        positive_dev->GetMem(),
                                        negativeDesc,
                                        negative_dev->GetMem(),
                                        dODesc,
                                        dO_dev->GetMem(),
                                        dADesc,
                                        dA_dev->GetMem(),
                                        dPDesc,
                                        dP_dev->GetMem(),
                                        dNDesc,
                                        dN_dev->GetMem(),
                                        margin,
                                        p,
                                        eps,
                                        swap,
                                        divisor);

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
            std::cout << "Wall-clock Time Forward TripletMarginLoss Elapsed: "
                      << t.gettime_ms() / iter << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward TripletMarginLoss Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(dA_dev->FromGPU(GetStream(), dA.data()) != 0)
        std::cerr << "Error copying (dA_dev) from GPU, size: " << dA_dev->GetSize() << std::endl;
    if(dP_dev->FromGPU(GetStream(), dP.data()) != 0)
        std::cerr << "Error copying (dP_dev) from GPU, size: " << dP_dev->GetSize() << std::endl;
    if(dN_dev->FromGPU(GetStream(), dN.data()) != 0)
        std::cerr << "Error copying (dN_dev) from GPU, size: " << dN_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TripletMarginLossDriver<Tgpu, Tref>::RunBackwardCPU()
{
    if(std::isnan(divisor))
    {
        mloTripletMarginLossUnreducedBackwardRunHost<Tgpu, Tref>(anchorDesc,
                                                                 positiveDesc,
                                                                 negativeDesc,
                                                                 dODesc,
                                                                 dADesc,
                                                                 dPDesc,
                                                                 dNDesc,
                                                                 anchor.data(),
                                                                 positive.data(),
                                                                 negative.data(),
                                                                 dO.data(),
                                                                 dAhost.data(),
                                                                 dPhost.data(),
                                                                 dNhost.data(),
                                                                 margin,
                                                                 p,
                                                                 eps,
                                                                 swap);
    }
    else
    {
        mloTripletMarginLossBackwardRunHost<Tgpu, Tref>(anchorDesc,
                                                        positiveDesc,
                                                        negativeDesc,
                                                        dADesc,
                                                        dPDesc,
                                                        dNDesc,
                                                        anchor.data(),
                                                        positive.data(),
                                                        negative.data(),
                                                        dO.data(),
                                                        dAhost.data(),
                                                        dPhost.data(),
                                                        dNhost.data(),
                                                        margin,
                                                        p,
                                                        eps,
                                                        swap,
                                                        divisor);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref TripletMarginLossDriver<Tgpu, Tref>::GetTolerance()
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
int TripletMarginLossDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward TripletMarginLoss FAILED: " << error << " > " << tolerance
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward TripletMarginLoss Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TripletMarginLossDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error_dA        = miopen::rms_range(dAhost, dA);
    auto error_dP        = miopen::rms_range(dPhost, dP);
    auto error_dN        = miopen::rms_range(dNhost, dN);

    if(!std::isfinite(error_dA) || error_dA > tolerance || !std::isfinite(error_dP) ||
       error_dP > tolerance || !std::isfinite(error_dN) || error_dN > tolerance)
    {
        std::cout << "Backward TripletMarginLoss FAILED: {" << error_dA << "," << error_dP << ","
                  << error_dN << "} > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backward TripletMarginLoss Verifies OK on CPU reference ({" << error_dA << ","
                  << error_dP << "," << error_dN << "} < " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
