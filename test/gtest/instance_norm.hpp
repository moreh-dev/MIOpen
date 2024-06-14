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
 * LIABILITY, WHETHER IN AN ACTN OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTN WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "../driver/tensor_driver.hpp"
#include "cpu_instance_norm.hpp"
#include "get_handle.hpp"
#include "miopen/allocator.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <cstddef>
#include <cstdlib>
#include <random>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/instance_norm.hpp>

struct InstanceNormTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    float epsilon  = 1e-05;
    float momentum = 0.1;
    bool useInputStats;
    std::string model_name;
    bool isContiguous = true;
    friend std::ostream& operator<<(std::ostream& os, const InstanceNormTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W;
    }

    std::vector<size_t> GetInput()
    {
        if((N != 0) && (C != 0) && (D != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, D, H, W});
        }
        else if((N != 0) && (C != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, H, W});
        }
        else if((N != 0) && (C != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, W});
        }
        else if((N != 0) && (W != 0))
        {
            return std::vector<size_t>({N, W});
        }
        else if((N != 0))
        {
            return std::vector<size_t>({N});
        }
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }

    std::vector<size_t> ComputeStrides(std::vector<size_t> inputDim) const
    {
        if(!isContiguous)
            std::swap(inputDim.front(), inputDim.back());
        std::vector<size_t> strides(inputDim.size());
        strides.back() = 1;
        for(int i = inputDim.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * inputDim[i + 1];
        if(!isContiguous)
            std::swap(strides.front(), strides.back());
        return strides;
    }
};

std::vector<InstanceNormTestCase> InstanceNormTestConfigs()
{ // n c d h w padding
    return {
        // {2, 128, 32, 32, 32, 1e-05, 0.1, false, "3dunet", true},
        // {2, 256, 16, 16, 16, 1e-05, 0.1, false, "3dunet", true},
        // {2, 320, 8, 8, 8, 1e-05, 0.1, false, "3dunet", true},
        // {2, 320, 4, 4, 4, 1e-05, 0.1, false, "3dunet", true},
        // {2, 32, 32, 32, 32, 1e-05, 0.1, true, "3dunet", true},
        // {2, 128, 32, 32, 32, 1e-05, 0.1, true, "3dunet", true},
        // {2, 256, 16, 16, 16, 1e-05, 0.1, true, "3dunet", true},
        // {2, 320, 8, 8, 8, 1e-05, 0.1, true, "3dunet", true},
        // {2, 80, 4, 4, 4, 1e-05, 0.1, true, "3dunet", true},
    };
}

template <typename T>
struct InstanceNormFwdTest : public ::testing::TestWithParam<InstanceNormTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        std::vector<size_t> in_dims          = config.GetInput();
        std::vector<size_t> in_strides       = config.ComputeStrides(in_dims);
        std::vector<size_t> weight_dims      = {in_dims[1]};
        std::vector<size_t> weight_strides   = config.ComputeStrides(weight_dims);
        std::vector<size_t> bias_dims        = {in_dims[1]};
        std::vector<size_t> bias_strides     = config.ComputeStrides(bias_dims);
        std::vector<size_t> mean_in_dims     = {in_dims[1]};
        std::vector<size_t> mean_in_strides  = config.ComputeStrides(mean_in_dims);
        std::vector<size_t> var_in_dims      = {in_dims[1]};
        std::vector<size_t> var_in_strides   = config.ComputeStrides(var_in_dims);
        std::vector<size_t> mean_var_dims    = {in_dims[0], in_dims[1] * 2};
        std::vector<size_t> mean_var_strides = config.ComputeStrides(mean_var_dims);

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_one   = [&](auto...) { return 1; };
        auto gen_zero  = [&](auto...) { return 0; };
        input          = tensor<T>{in_dims, in_strides}.generate(gen_value);
        weight         = tensor<T>{weight_dims, weight_strides}.generate(gen_value);
        bias           = tensor<T>{bias_dims, bias_strides}.generate(gen_value);
        meanIn         = tensor<T>{mean_in_dims, mean_in_strides}.generate(gen_zero);
        meanInHost     = tensor<T>{mean_in_dims, mean_in_strides}.generate(gen_zero);
        varIn          = tensor<T>{var_in_dims, var_in_strides}.generate(gen_one);
        varInHost      = tensor<T>{var_in_dims, var_in_strides}.generate(gen_one);
        meanVar        = tensor<T>{mean_var_dims, mean_var_strides}.generate(gen_zero);
        meanVarHost    = tensor<T>{mean_var_dims, mean_var_strides}.generate(gen_zero);

        output     = tensor<T>{in_dims}.generate(gen_zero);
        outputHost = tensor<T>{in_dims}.generate(gen_zero);

        input_dev   = handle.Write(input.data);
        output_dev  = handle.Write(output.data);
        weight_dev  = handle.Write(weight.data);
        bias_dev    = handle.Write(bias.data);
        meanIn_dev  = handle.Write(meanIn.data);
        varIn_dev   = handle.Write(varIn.data);
        meanVar_dev = handle.Write(meanVar.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        status = miopen::InstanceNormForward(handle,
                                             input.desc,
                                             input_dev.get(),
                                             output.desc,
                                             output_dev.get(),
                                             weight.desc,
                                             weight_dev.get(),
                                             bias.desc,
                                             bias_dev.get(),
                                             meanIn.desc,
                                             meanIn_dev.get(),
                                             varIn.desc,
                                             varIn_dev.get(),
                                             meanIn.desc,
                                             meanIn_dev.get(),
                                             varIn.desc,
                                             varIn_dev.get(),
                                             meanVar.desc,
                                             meanVar_dev.get(),
                                             config.epsilon,
                                             config.momentum,
                                             config.useInputStats);
        if (config.useInputStats)
        {
            cpu_instance_norm_forward_train<T>(input,
                                                    outputHost,
                                                    weight,
                                                    bias,
                                                    meanInHost,
                                                    varInHost,
                                                    meanInHost,
                                                    varInHost,
                                                    meanVarHost,
                                                    config.epsilon,
                                                    config.momentum,
                                                    config.useInputStats);
        }
        else
        {
            cpu_instance_norm_forward_test<T>(input,
                                        outputHost,
                                        weight,
                                        bias,
                                        meanInHost,
                                        varInHost,
                                        config.epsilon,
                                        config.useInputStats);
        }
            
        EXPECT_EQ(status, miopenStatusSuccess);
        output.data  = handle.Read<T>(output_dev, output.data.size());
        if (config.useInputStats)
            meanVar.data = handle.Read<T>(meanVar_dev, meanVar.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;
        auto error_output   = miopen::rms_range(outputHost, output);
        auto error_mean_var = miopen::rms_range(meanVarHost, meanVar);
        EXPECT_TRUE(miopen::range_distance(outputHost) == miopen::range_distance(output));
        if (config.useInputStats)
            EXPECT_TRUE(miopen::range_distance(meanVarHost) == miopen::range_distance(meanVar));
        if (config.useInputStats)
            EXPECT_TRUE(error_output < tolerance && error_mean_var < tolerance)
                << "Error forward output beyond tolerance Error: {" << error_output << ","
                << error_mean_var << "},  Tolerance: " << tolerance;
        else
            EXPECT_TRUE(error_output < tolerance) << "Error forward output beyond tolerance Error: {" << error_output <<  "},  Tolerance: " << tolerance;
    }
    InstanceNormTestCase config;

    tensor<T> input;
    tensor<T> output;
    tensor<T> weight;
    tensor<T> bias;
    tensor<T> meanIn;
    tensor<T> varIn;
    tensor<T> meanVar;

    tensor<T> outputHost;
    tensor<T> meanInHost;
    tensor<T> varInHost;
    tensor<T> meanVarHost;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr weight_dev;
    miopen::Allocator::ManageDataPtr bias_dev;
    miopen::Allocator::ManageDataPtr meanIn_dev;
    miopen::Allocator::ManageDataPtr varIn_dev;
    miopen::Allocator::ManageDataPtr meanVar_dev;
};

template <typename T>
struct InstanceNormBwdTest : public ::testing::TestWithParam<InstanceNormTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        std::vector<size_t> in_dims          = config.GetInput();
        std::vector<size_t> in_strides       = config.ComputeStrides(in_dims);
        std::vector<size_t> doutput_dims      = in_dims;
        std::vector<size_t> doutput_strides   = config.ComputeStrides(doutput_dims);
        std::vector<size_t> weight_dims        = {in_dims[1]};
        std::vector<size_t> weight_strides     = config.ComputeStrides(weight_dims);
        std::vector<size_t> mean_var_dims     = {in_dims[0], in_dims[1] * 2};
        std::vector<size_t> mean_var_strides  = config.ComputeStrides(mean_var_dims);
        std::vector<size_t> dinput_dims      = in_dims;
        std::vector<size_t> dinput_strides   = config.ComputeStrides(dinput_dims);
        std::vector<size_t> dweight_dims    = weight_dims;
        std::vector<size_t> dweight_strides = config.ComputeStrides(dweight_dims);
        std::vector<size_t> dbias_dims    = {in_dims[1]};
        std::vector<size_t> dbias_strides = config.ComputeStrides(dbias_dims);

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_zero  = [&](auto...) { return 0; };
        input          = tensor<T>{in_dims, in_strides}.generate(gen_value);
        doutput          = tensor<T>{doutput_dims, doutput_strides}.generate(gen_value);
        weight          = tensor<T>{weight_dims, weight_strides}.generate(gen_value);
        meanVar          = tensor<T>{mean_var_dims, mean_var_strides}.generate(gen_value);
        dinput          = tensor<T>{dinput_dims, dinput_strides}.generate(gen_zero);
        dweight          = tensor<T>{dweight_dims, dweight_strides}.generate(gen_zero);
        dbias          = tensor<T>{dbias_dims, dbias_strides}.generate(gen_zero);

        dinputHost     = tensor<T>{dinput_dims, dinput_strides}.generate(gen_zero);
        dweightHost      = tensor<T>{dweight_dims, dweight_strides}.generate(gen_zero);
        dbiasHost    = tensor<T>{dbias_dims, dbias_strides}.generate(gen_zero);

        input_dev   = handle.Write(input.data);
        doutput_dev  = handle.Write(doutput.data);
        weight_dev  = handle.Write(weight.data);
        meanVar_dev    = handle.Write(meanVar.data);
        dinput_dev  = handle.Write(dinput.data);
        dweight_dev   = handle.Write(dweight.data);
        dbias_dev = handle.Write(dbias.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        status = miopen::InstanceNormBackward(handle,
                                             input.desc,
                                             input_dev.get(),
                                             weight.desc,
                                             weight_dev.get(),
                                             dinput.desc,
                                             dinput_dev.get(),
                                             doutput.desc,
                                             doutput_dev.get(),
                                             dweight.desc,
                                             dweight_dev.get(),
                                             dbias.desc,
                                             dbias_dev.get(),
                                             meanVar.desc,
                                             meanVar_dev.get());

        cpu_instance_norm_backward<T>(input,
                             weight,
                             dinputHost,
                             doutput,
                             dweightHost,
                             dbiasHost,
                             meanVar);
            
        EXPECT_EQ(status, miopenStatusSuccess);
        dinput.data  = handle.Read<T>(dinput_dev, dinput.data.size());
        dweight.data  = handle.Read<T>(dweight_dev, dweight.data.size());
        dbias.data  = handle.Read<T>(dbias_dev, dbias.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;
        for (int i = 0; i < 10; ++i)
        {
            std::cout << "dinput[" << i << "]: " << dinput[i] << " ~ " << dinputHost[i] << std::endl;
        }
        for (int i = 0; i < 10; ++i)
        {
            std::cout << "dweight[" << i << "]: " << dweight[i] << " ~ " << dweightHost[i] << std::endl;
        }
        for (int i = 0; i < 10; ++i)
        {
            std::cout << "dbias[" << i << "]: " << dbias[i] << " ~ " << dbiasHost[i] << std::endl;
        }
        auto error_dinput   = miopen::rms_range(dinputHost, dinput);
        auto error_dweight = miopen::rms_range(dweightHost, dweight);
        auto error_dbias = miopen::rms_range(dbiasHost, dbias);
        EXPECT_TRUE(miopen::range_distance(dinputHost) == miopen::range_distance(dinput));
        EXPECT_TRUE(miopen::range_distance(dweightHost) == miopen::range_distance(dweight));
        EXPECT_TRUE(miopen::range_distance(dbiasHost) == miopen::range_distance(dbias));

        EXPECT_TRUE(error_dinput < tolerance && error_dweight < tolerance && error_dbias < tolerance)
                << "Error backward output beyond tolerance Error: {" << error_dinput << ","
                << error_dweight << "," << error_dbias << "},  Tolerance: " << tolerance;
    }
    InstanceNormTestCase config;

    tensor<T> input;
    tensor<T> doutput;
    tensor<T> weight;
    tensor<T> meanVar;
    tensor<T> dinput;
    tensor<T> dweight;
    tensor<T> dbias;

    tensor<T> dinputHost;
    tensor<T> dweightHost;
    tensor<T> dbiasHost;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr doutput_dev;
    miopen::Allocator::ManageDataPtr weight_dev;
    miopen::Allocator::ManageDataPtr meanVar_dev;
    miopen::Allocator::ManageDataPtr dinput_dev;
    miopen::Allocator::ManageDataPtr dweight_dev;
    miopen::Allocator::ManageDataPtr dbias_dev;
};
