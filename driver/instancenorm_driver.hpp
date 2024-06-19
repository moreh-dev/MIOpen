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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACTORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_MIOPEN_INSTANCENORM_DRIVER_HPP
#define GUARD_MIOPEN_INSTANCENORM_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "mloInstanceNormHost.hpp"
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

template <typename Tgpu, typename Tref>
class InstanceNormDriver : public Driver
{
public:
    InstanceNormDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&weightDesc);
        miopenCreateTensorDescriptor(&biasDesc);
        miopenCreateTensorDescriptor(&meanInDesc);
        miopenCreateTensorDescriptor(&varInDesc);
        miopenCreateTensorDescriptor(&meanVarDesc);
        miopenCreateTensorDescriptor(&doutputDesc);
        miopenCreateTensorDescriptor(&dinputDesc);
        miopenCreateTensorDescriptor(&dweightDesc);
        miopenCreateTensorDescriptor(&dbiasDesc);

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
    ~InstanceNormDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(weightDesc);
        miopenDestroyTensorDescriptor(biasDesc);
        miopenDestroyTensorDescriptor(meanInDesc);
        miopenDestroyTensorDescriptor(varInDesc);
        miopenDestroyTensorDescriptor(meanVarDesc);
        miopenDestroyTensorDescriptor(doutputDesc);
        miopenDestroyTensorDescriptor(dinputDesc);
        miopenDestroyTensorDescriptor(dweightDesc);
        miopenDestroyTensorDescriptor(dbiasDesc);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t weightDesc;
    miopenTensorDescriptor_t biasDesc;
    miopenTensorDescriptor_t meanInDesc;
    miopenTensorDescriptor_t varInDesc;
    miopenTensorDescriptor_t meanVarDesc;

    miopenTensorDescriptor_t doutputDesc;
    miopenTensorDescriptor_t dinputDesc;
    miopenTensorDescriptor_t dweightDesc;
    miopenTensorDescriptor_t dbiasDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> weight_dev;
    std::unique_ptr<GPUMem> bias_dev;
    std::unique_ptr<GPUMem> meanIn_dev;
    std::unique_ptr<GPUMem> varIn_dev;
    std::unique_ptr<GPUMem> meanVar_dev;

    std::unique_ptr<GPUMem> doutput_dev;
    std::unique_ptr<GPUMem> dinput_dev;
    std::unique_ptr<GPUMem> dweight_dev;
    std::unique_ptr<GPUMem> dbias_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> output;
    std::vector<Tgpu> weight;
    std::vector<Tgpu> bias;
    std::vector<Tgpu> meanIn;
    std::vector<Tgpu> varIn;
    std::vector<Tgpu> meanVar;

    std::vector<Tgpu> doutput;
    std::vector<Tgpu> dinput;
    std::vector<Tgpu> dweight;
    std::vector<Tgpu> dbias;

    std::vector<Tref> output_host;
    std::vector<Tref> meanIn_host;
    std::vector<Tref> varIn_host;
    std::vector<Tref> meanVar_host;

    std::vector<Tref> doutput_host;
    std::vector<Tref> dinput_host;
    std::vector<Tref> dweight_host;
    std::vector<Tref> dbias_host;

    float epsilon;
    float momentum;
    bool useInputStats;
};

template <typename Tgpu, typename Tref>
int InstanceNormDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int InstanceNormDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> input_length = GetTensorLengthsFromCmdLine();
    epsilon                       = inflags.GetValueDouble("epsilon");
    momentum                      = inflags.GetValueDouble("momentum");
    useInputStats                 = inflags.GetValueInt("useInputStats") == 1;
    if(!(input_length.size() == 5))
    {
        std::cout << "Tensor must not be 5D";
        return miopenStatusInvalidValue;
    }
    std::vector<int> weight_length  = {input_length[1]};
    std::vector<int> bias_length    = {input_length[1]};
    std::vector<int> meanIn_length  = {input_length[1]};
    std::vector<int> varIn_length   = {input_length[1]};
    std::vector<int> meanVar_length = {input_length[0], input_length[1] * 2};

    SetTensorNd(inputDesc, input_length, data_type);
    SetTensorNd(outputDesc, input_length, data_type);
    SetTensorNd(weightDesc, weight_length, data_type);
    SetTensorNd(biasDesc, bias_length, data_type);
    SetTensorNd(meanInDesc, meanIn_length, data_type);
    SetTensorNd(varInDesc, varIn_length, data_type);
    SetTensorNd(meanVarDesc, meanVar_length, data_type);

    SetTensorNd(doutputDesc, input_length, data_type);
    SetTensorNd(dinputDesc, input_length, data_type);
    SetTensorNd(dweightDesc, weight_length, data_type);
    SetTensorNd(dbiasDesc, weight_length, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int InstanceNormDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Instance Norm Forward (Default=1)", "int");
    inflags.AddInputFlag("DimLengths",
                         'D',
                         "16,32,128,128,128",
                         "The dimensional lengths of the input tensor",
                         "string");
    inflags.AddInputFlag("epsilon", 'E', "1e-05f", "Epsilon (Default=1e-05f)", "float");
    inflags.AddInputFlag("momentum", 'M', "0.1", "Momentum (Default=0.1)", "float");
    inflags.AddInputFlag("useInputStats", 'U', "1", "Use input stats (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "0", "Verify Each Layer (Default=0)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> InstanceNormDriver<Tgpu, Tref>::GetTensorLengthsFromCmdLine()
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
int InstanceNormDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_sz   = GetTensorSize(inputDesc);
    size_t output_sz  = GetTensorSize(outputDesc);
    size_t weight_sz  = GetTensorSize(weightDesc);
    size_t bias_sz    = GetTensorSize(biasDesc);
    size_t meanIn_sz  = GetTensorSize(meanInDesc);
    size_t varIn_sz   = GetTensorSize(varInDesc);
    size_t meanVar_sz = GetTensorSize(meanVarDesc);

    size_t doutput_sz = GetTensorSize(doutputDesc);
    size_t dinput_sz  = GetTensorSize(dinputDesc);
    size_t dweight_sz = GetTensorSize(dweightDesc);
    size_t dbias_sz   = GetTensorSize(dbiasDesc);

    uint32_t ctx = 0;

    input_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));
    weight_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, weight_sz, sizeof(Tgpu)));
    bias_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, bias_sz, sizeof(Tgpu)));
    meanIn_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, meanIn_sz, sizeof(Tgpu)));
    varIn_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, varIn_sz, sizeof(Tgpu)));
    meanVar_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, meanVar_sz, sizeof(Tgpu)));

    doutput_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, doutput_sz, sizeof(Tgpu)));
    dinput_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, dinput_sz, sizeof(Tgpu)));
    dweight_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, dweight_sz, sizeof(Tgpu)));
    dbias_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, dbias_sz, sizeof(Tgpu)));

    input   = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0.0f));
    output  = std::vector<Tgpu>(output_sz, static_cast<Tgpu>(0.0f));
    weight  = std::vector<Tgpu>(weight_sz, static_cast<Tgpu>(0.0f));
    bias    = std::vector<Tgpu>(bias_sz, static_cast<Tgpu>(0.0f));
    meanIn  = std::vector<Tgpu>(meanIn_sz, static_cast<Tgpu>(0.0f));
    varIn   = std::vector<Tgpu>(varIn_sz, static_cast<Tgpu>(1.0f));
    meanVar = std::vector<Tgpu>(meanVar_sz, static_cast<Tgpu>(0.0f));

    doutput = std::vector<Tgpu>(doutput_sz, static_cast<Tgpu>(1.0f));
    dinput  = std::vector<Tgpu>(dinput_sz, static_cast<Tgpu>(0.0f));
    dweight = std::vector<Tgpu>(dweight_sz, static_cast<Tgpu>(0.0f));
    dbias   = std::vector<Tgpu>(dbias_sz, static_cast<Tgpu>(0.0f));

    output_host  = std::vector<Tref>(output_sz, static_cast<Tref>(0.0f));
    meanIn_host  = std::vector<Tref>(meanIn_sz, static_cast<Tref>(0.0f));
    varIn_host   = std::vector<Tref>(varIn_sz, static_cast<Tref>(1.0f));
    meanVar_host = std::vector<Tref>(meanVar_sz, static_cast<Tref>(0.0f));

    doutput_host = std::vector<Tref>(doutput_sz, static_cast<Tref>(0.0f));
    dinput_host  = std::vector<Tref>(dinput_sz, static_cast<Tref>(0.0f));
    dweight_host = std::vector<Tref>(dweight_sz, static_cast<Tref>(0.0f));
    dbias_host   = std::vector<Tref>(dbias_sz, static_cast<Tref>(0.0f));

    int status;

    for(int i = 0; i < input_sz; i++)
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    status = input_dev->ToGPU(GetStream(), input.data());

    for(int i = 0; i < weight_sz; i++)
        weight[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    status |= weight_dev->ToGPU(GetStream(), weight.data());
    for(int i = 0; i < bias_sz; i++)
    {
        bias[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status |= bias_dev->ToGPU(GetStream(), bias.data());
    status |= meanIn_dev->ToGPU(GetStream(), meanIn.data());
    status |= varIn_dev->ToGPU(GetStream(), varIn.data());
    status |= meanVar_dev->ToGPU(GetStream(), meanVar.data());
    for(int i = 0; i < doutput_sz; i++)
    {
        doutput[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status |= doutput_dev->ToGPU(GetStream(), doutput.data());
    status |= dinput_dev->ToGPU(GetStream(), dinput.data());
    status |= dweight_dev->ToGPU(GetStream(), dweight.data());
    status |= dbias_dev->ToGPU(GetStream(), dbias.data());

    if(status != 0)
        std::cout << "Instance Norm Driver Error copying data to GPU\n" << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int InstanceNormDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenInstanceNormForward(GetHandle(),
                                  inputDesc,
                                  input_dev->GetMem(),
                                  outputDesc,
                                  output_dev->GetMem(),
                                  weightDesc,
                                  weight_dev->GetMem(),
                                  biasDesc,
                                  bias_dev->GetMem(),
                                  meanInDesc,
                                  meanIn_dev->GetMem(),
                                  varInDesc,
                                  varIn_dev->GetMem(),
                                  meanInDesc,
                                  meanIn_dev->GetMem(),
                                  varInDesc,
                                  varIn_dev->GetMem(),
                                  meanVarDesc,
                                  meanVar_dev->GetMem(),
                                  epsilon,
                                  momentum,
                                  useInputStats);

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
            std::cout << "Wall-clock Time Instance Norm Forward Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Instance Norm Forward Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;
    if(useInputStats)
        if(meanVar_dev->FromGPU(GetStream(), meanVar.data()) != 0)
            std::cerr << "Error copying (meanVar_dev) from GPU, size: " << meanVar_dev->GetSize()
                      << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int InstanceNormDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(useInputStats)
    {
        mloInstanceNormTrainRunHost<Tgpu, Tref>(input.data(),
                                       inputDesc,
                                       output_host.data(),
                                       outputDesc,
                                       weight.data(),
                                       weightDesc,
                                       bias.data(),
                                       biasDesc,
                                       meanIn_host.data(),
                                       meanInDesc,
                                       varIn_host.data(),
                                       varInDesc,
                                       meanIn_host.data(),
                                       meanInDesc,
                                       varIn_host.data(),
                                       varInDesc,
                                       meanVar_host.data(),
                                       meanVarDesc,
                                       epsilon,
                                       momentum,
                                       useInputStats);
    }
    else
    {
        mloInstanceNormTestRunHost<Tgpu, Tref>(input.data(),
                                      inputDesc,
                                      output_host.data(),
                                      outputDesc,
                                      weight.data(),
                                      weightDesc,
                                      bias.data(),
                                      biasDesc,
                                      meanIn_host.data(),
                                      meanInDesc,
                                      varIn_host.data(),
                                      varInDesc,
                                      epsilon,
                                      useInputStats);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int InstanceNormDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenInstanceNormBackward(GetHandle(),
                                   inputDesc,
                                   input_dev->GetMem(),
                                   weightDesc,
                                   weight_dev->GetMem(),
                                   dinputDesc,
                                   dinput_dev->GetMem(),
                                   doutputDesc,
                                   doutput_dev->GetMem(),
                                   dweightDesc,
                                   dweight_dev->GetMem(),
                                   dbiasDesc,
                                   dbias_dev->GetMem(),
                                   meanVarDesc,
                                   meanVar_dev->GetMem());

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
            std::cout << "Wall-clock Time Instance Norm Backward Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Instance Norm Backward Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(dinput_dev->FromGPU(GetStream(), dinput.data()) != 0)
        std::cerr << "Error copying (dinput_dev) from GPU, size: " << dinput_dev->GetSize()
                  << std::endl;

    if(dweight_dev->FromGPU(GetStream(), dweight.data()) != 0)
        std::cerr << "Error copying (dinput_dev) from GPU, size: " << dweight_dev->GetSize()
                  << std::endl;
    if(dbias_dev->FromGPU(GetStream(), dbias.data()) != 0)
        std::cerr << "Error copying (dinput_dev) from GPU, size: " << dbias_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int InstanceNormDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloInstanceNormBackwardRunHost<Tgpu, Tref>(input.data(),
                                      inputDesc,
                                      weight.data(),
                                      weightDesc,
                                      dinput_host.data(),
                                      dinputDesc,
                                      doutput.data(),
                                      doutputDesc,
                                      dweight_host.data(),
                                      dweightDesc,
                                      dbias_host.data(),
                                      dbiasDesc,
                                      meanVar.data(),
                                      meanVarDesc);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref InstanceNormDriver<Tgpu, Tref>::GetTolerance()
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
int InstanceNormDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error_output     = miopen::rms_range(output_host, output);
    auto error_mean_var   = miopen::rms_range(meanVar_host, meanVar);

    if(!std::isfinite(error_output) || error_output > tolerance ||
       ((!std::isfinite(error_mean_var) || error_mean_var > tolerance) && useInputStats))
    {
        if(useInputStats)
        {
            std::cout << "Forward Instance Norm FAILED: {" << error_output << "," << error_mean_var
                      << "} > " << tolerance << std::endl;
        }
        else
        {
            std::cout << "Forward Instance Norm FAILED: {" << error_output << "} > " << tolerance
                      << std::endl;
        }
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Instance Norm Verifies OK on CPU reference ({" << error_output << ","
                  << error_mean_var << "} < " << tolerance << ')' << std::endl;
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int InstanceNormDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error_dinput     = miopen::rms_range(dinput_host, dinput);
    auto error_dweight    = miopen::rms_range(dweight_host, dweight);
    auto error_dbias      = miopen::rms_range(dbias_host, dbias);

    if(!std::isfinite(error_dinput) || error_dinput > tolerance || !std::isfinite(error_dweight) ||
       error_dweight > tolerance || !std::isfinite(error_dbias) || error_dbias > tolerance)
    {
        std::cout << "Backward Instance Norm FAILED: {" << error_dinput << "," << error_dweight
                  << "," << error_dbias << "} > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backward Instance Norm Verifies OK on CPU reference ({" << error_dinput << ","
                  << error_dweight << "," << error_dbias << "} < " << tolerance << ')' << std::endl;
    }
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_INSTANCENORM_DRIVER_HPP
