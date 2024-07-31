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
#include "InputFlags.hpp"
#include "driver.hpp"
#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include "tensor_driver.hpp"
#include <rocfft/rocfft.h>
#include "timer.hpp"
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>
#include <cmath>

template <typename Tgpu, typename Tcheck>
class SigmoidFocalLossDriver : public Driver
{
public:
    SigmoidFocalLossDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&targetDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&doutputDesc);
        miopenCreateTensorDescriptor(&dinputDesc);
        miopenCreateTensorDescriptor(&dtargetDesc);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> input);
    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tcheck GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~SigmoidFocalLossDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(targetDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(doutputDesc);
        miopenDestroyTensorDescriptor(dinputDesc);
        miopenDestroyTensorDescriptor(dtargetDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t targetDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t doutputDesc;
    miopenTensorDescriptor_t dinputDesc;
    miopenTensorDescriptor_t dtargetDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> target_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> doutput_dev;
    std::unique_ptr<GPUMem> dinput_dev;
    std::unique_ptr<GPUMem> dtarget_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> target;
    std::vector<Tgpu> output;
    std::vector<Tcheck> outputHost;
    std::vector<Tgpu> doutput;
    std::vector<Tgpu> dinput;
    std::vector<Tcheck> dinputHost;
    std::vector<Tgpu> dtarget;
    std::vector<Tcheck> dtargetHost;
    std::vector<Tgpu> workspace;
    std::vector<Tcheck> workspaceHost;

    bool isContiguous = true;
};

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::GetandSetData()
{
    auto inDims               = inflags.GetValueTensor("dim-lengths").lengths;
    std::vector<int> inStride = ComputeStrides(inDims);

    SetTensorNd(inputDesc, inDims, inStride, data_type);
    SetTensorNd(targetDesc, inDims, inStride, data_type);
    SetTensorNd(doutputDesc, inDims, data_type);
    SetTensorNd(dinputDesc, inDims, data_type);

    SetTensorNd(outputDesc, inDims, data_type);

    return 0;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tcheck>
std::vector<int> SigmoidFocalLossDriver<Tgpu, Tcheck>::ComputeStrides(std::vector<int> inputDim)
{
    if(!isContiguous)
        std::swap(inputDim.front(), inputDim.back());
    std::vector<int> strides(inputDim.size());
    strides.back() = 1;
    for(int i = inputDim.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * inputDim[i + 1];
    if(!isContiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward (Default=1)", "int");
    inflags.AddTensorFlag(
        "dim-lengths", 'D', "256x4x2", "The dimensional lengths of the input tensor");
    inflags.AddInputFlag("is-contiguous", 'c', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag(
        "reduction", 'R', "0", "reduction mode: 0(default) - unreduced, 1 - sum, 2 -mean", "int");
    inflags.AddInputFlag("alpha", 'A', "0.25", "Alpha (Default=0.25)", "float");
    inflags.AddInputFlag("gamma", 'G', "2", "Gamma (Default=2)", "float");
    inflags.AddInputFlag(
        "target-gradient", 'T', "0", "Is target gradient computed (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::AllocateBuffersAndCopy()
{
    size_t in_sz     = miopen::deref(inputDesc).GetElementSize();
    size_t target_sz = miopen::deref(targetDesc).GetElementSize();
    size_t out_sz    = miopen::deref(outputDesc).GetElementSize();
    size_t dO_sz     = miopen::deref(doutputDesc).GetElementSize();
    size_t dI_sz     = miopen::deref(dinputDesc).GetElementSize();
    size_t dT_sz     = miopen::deref(dtargetDesc).GetElementSize();

    uint32_t ctx = 0;

    input_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    target_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, target_sz, sizeof(Tgpu)));
    output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    doutput_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, dO_sz, sizeof(Tgpu)));
    dinput_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, dI_sz, sizeof(Tgpu)));
    dtarget_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, dT_sz, sizeof(Tgpu)));

    input       = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    target      = std::vector<Tgpu>(target_sz, static_cast<Tgpu>(0));
    output      = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    outputHost  = std::vector<Tcheck>(out_sz, static_cast<Tcheck>(0));
    doutput     = std::vector<Tgpu>(dO_sz, static_cast<Tgpu>(0));
    dinput      = std::vector<Tgpu>(dI_sz, static_cast<Tgpu>(0));
    dinputHost  = std::vector<Tcheck>(dI_sz, static_cast<Tcheck>(0));
    dtarget     = std::vector<Tgpu>(dT_sz, static_cast<Tgpu>(0));
    dtargetHost = std::vector<Tcheck>(dT_sz, static_cast<Tcheck>(0));
    // workspace             = std::vector<Tgpu>(workSpaceElems, static_cast<Tgpu>(0));
    // workspaceHost         = std::vector<Tcheck>(workSpaceElems, static_cast<Tcheck>(0));

    float randomBound = 2;
    // For half, the random bound is smaller to avoid half overflow
    for(int i = 0; i < in_sz; i++)
    {
        input[i] =
            prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-randomBound), static_cast<Tgpu>(randomBound));
    }
    for(int i = 0; i < dO_sz; ++i)
    {
        doutput[i] =
            prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(-randomBound), static_cast<Tgpu>(randomBound));
    }

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

    if(dtarget_dev->ToGPU(GetStream(), dtarget.data()) != 0)
        std::cerr << "Error copying (dT) to GPU, size: " << dtarget_dev->GetSize() << std::endl;

    if(workspace_dev->ToGPU(GetStream(), workspace.data()) != 0)
        std::cerr << "Error copying (dI) to GPU, size: " << workspace_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

void CHECK_ROCFFT_STATUS(rocfft_status err)
{
    if(err != rocfft_status_success)
    {
        std::cerr << "rocFFT error: " << err << std::endl;
    }
}

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::RunForwardGPU()
{
    bool output_non_contiguous = false;

    rocfft_plan_description desc;
    CHECK_ROCFFT_STATUS(rocfft_plan_description_create(&desc));

    // NOTE(kyuhyeon): input tensor need to clone due to rocfft modifies input
    // tensor values.
    TensorWrapper input_clone_tensor = workspace::RegisterWorkspace(
        ctx, input_real_tensor.dims(), input_real_tensor.dtype(), true);
    common::Assign(ctx, input_real_tensor, input_clone_tensor);

    TensorWrapper output_target_tensor;
    if(output_non_contiguous)
    {
        output_target_tensor = workspace::RegisterWorkspace(
            ctx, output_complex_tensor.dims(), output_complex_tensor.dtype(), true);
    }
    else
    {
        output_target_tensor = output_complex_tensor;
    }

    MODNN_CHECK(input_clone_tensor.offset() == 0 && input_clone_tensor.stride(-1) == 1,
                "Input tensor should have no offset. And last dimension's stride should "
                "be 1.")
    MODNN_CHECK(output_target_tensor.offset() == 0 && output_target_tensor.stride(-1) == 1,
                "Output tensor should have no offset. And last dimension's "
                "stride should be 1")

    size_t in_offset[1]    = {static_cast<size_t>(input_clone_tensor.offset())};
    size_t out_offset[1]   = {static_cast<size_t>(output_target_tensor.offset())};
    size_t in_stride_size  = 1;
    size_t out_stride_size = 1;
    size_t in_stride[1]    = {1};
    size_t out_stride[1]   = {1};
    size_t in_distance     = fft_length[0];
    size_t out_distance    = 1 + fft_length[0] / 2;
    size_t batch_size      = input_real_tensor.numel() / fft_length[0];

    MODNN_CHECK_ROCFFT_STATUS(
        rocfft_plan_description_set_data_layout(desc,
                                                rocfft_array_type_real,
                                                rocfft_array_type_hermitian_interleaved,
                                                in_offset,
                                                out_offset,
                                                in_stride_size,
                                                in_stride,
                                                in_distance,
                                                out_stride_size,
                                                out_stride,
                                                out_distance));

    rocfft_plan rocfft_rttf_plan;
    MODNN_CHECK_ROCFFT_STATUS(rocfft_plan_create(
        &rocfft_rttf_plan,
        rocfft_placement_notinplace,
        rocfft_transform_type_real_forward,
        (input_real_tensor.dtype() == moreh::DTYPE::FLOAT32) ? rocfft_precision_single
                                                             : rocfft_precision_double,
        dim,
        fft_length.data(),
        batch_size,
        desc));

    size_t work_buf_size = 0;
    MODNN_CHECK_ROCFFT_STATUS(rocfft_plan_get_work_buffer_size(rocfft_rttf_plan, &work_buf_size));

    rocfft_execution_info info;
    MODNN_CHECK_ROCFFT_STATUS(rocfft_execution_info_create(&info));

    MODNN_CHECK_ROCFFT_STATUS(
        rocfft_execution_info_set_stream(info, reinterpret_cast<hipStream_t>(ctx->GetStream())));

    if(work_buf_size)
    {
        TensorWrapper ws_tensor = workspace::RegisterWorkspace<1>(
            ctx, {static_cast<int64_t>(work_buf_size)}, moreh::DTYPE::UINT8);
        void* work_buf = ws_tensor.dev_ptr();
        MODNN_CHECK_ROCFFT_STATUS(
            rocfft_execution_info_set_work_buffer(info, work_buf, work_buf_size));
    }

    void* input_mem  = input_clone_tensor.hip_mem();
    void* output_mem = output_target_tensor.hip_mem();

    bool ws_prof            = moreh::env::IsWSProfilingMode();
    bool kernel_prof        = env::KernelProfilingMode() && hip::DoKernelProfile();
    std::string kernel_name = std::string("rocfftRFFT");
    if(kernel_prof)
        spdlog::debug("ROCFFT_SUBMITTED {}", kernel_name);
    if(ws_prof || kernel_prof)
    {
        auto profile_info = hip::kernel_profile::KernelProfileInfo{};
        profile_info.SetKernelName(kernel_name);
        profile_info.SetProfileNbytes(input_clone_tensor.nbytes());
        hip::SetCallbackData(profile_info);
    }

    MODNN_CHECK_ROCFFT_STATUS(rocfft_execute(rocfft_rttf_plan, &input_mem, &output_mem, info));

    MODNN_CHECK_ROCFFT_STATUS(rocfft_execution_info_destroy(info));

    MODNN_CHECK_ROCFFT_STATUS(rocfft_plan_description_destroy(desc));

    MODNN_CHECK_ROCFFT_STATUS(rocfft_plan_destroy(rocfft_rttf_plan));

    if(output_non_contiguous)
    {
        common::Assign(ctx, output_target_tensor, output_complex_tensor);
    }

    // float kernel_total_time = 0;
    // float kernel_first_time = 0;

    // Timer t;
    // START_TIME

    // for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    // {
    //     float time = 0.0;
    //     miopenGetKernelTime(GetHandle(), &time);
    //     kernel_total_time += time;
    //     if(i == 0)
    //         kernel_first_time = time;
    // }

    // if(inflags.GetValueInt("time") == 1)
    // {
    //     STOP_TIME
    //     int iter = inflags.GetValueInt("iter");
    //     if(WALL_CLOCK)
    //         std::cout << "Wall-clock Time Rfft Fwd Elapsed: " << t.gettime_ms() / iter << " ms"
    //                   << std::endl;

    //     float kernel_average_time =
    //         iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
    //     std::cout << "GPU Kernel Time Rfft Fwd Elapsed: " << kernel_average_time << " ms"
    //               << std::endl;
    // }

    // if(output_dev->FromGPU(GetStream(), output.data()) != 0)
    //     std::cerr << "Error copying (out_dev) from GPU, size: " << output_dev->GetSize()
    //               << std::endl;

    // return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::RunForwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::RunBackwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
Tcheck SigmoidFocalLossDriver<Tgpu, Tcheck>::GetTolerance()
{
    // Computation error of fp16 is ~2^13 (=8192) bigger than
    // the one of fp32 because mantissa is shorter by 13 bits.
    auto tolerance = std::is_same<Tgpu, float>::value ? 1.5e-6 : 8.2e-3;

    // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
    if(std::is_same<Tgpu, bfloat16>::value)
        tolerance *= 8.0;
    return tolerance;
}

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::VerifyForward()
{
    RunForwardCPU();

    const Tcheck tolerance = GetTolerance();
    auto error             = miopen::rms_range(outputHost, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward rfft FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward rfft Verifies OK on CPU reference (" << error << "< " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int SigmoidFocalLossDriver<Tgpu, Tcheck>::VerifyBackward()
{
    RunBackwardCPU();

    const Tcheck tolerance = GetTolerance();
    auto dinputError       = miopen::rms_range(dinputHost, dinput);
    auto dtargetError      = miopen::rms_range(dtargetHost, dtarget);

    if(!std::isfinite(dinputError) || dinputError > tolerance)
    {
        std::cout << "Backward rfft FAILED: " << dinputError << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backward rfft Verifies OK on CPU reference (dinput: " << dinputError
                  << ", dtarget: " << dtargetError << "< " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
