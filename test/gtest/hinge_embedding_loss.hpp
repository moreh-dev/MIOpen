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

#include "../driver/tensor_driver.hpp"
#include "cpu_hinge_embedding_loss.hpp"
#include "get_handle.hpp"
#include "miopen/allocator.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <cstdlib>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/hinge_embedding_loss.hpp>

struct HingeEmbeddingLossTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    float margin;
    float divisor;
    std::string reduction;
    friend std::ostream& operator<<(std::ostream& os, const HingeEmbeddingLossTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " margin: " << tc.margin;
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
};

std::vector<HingeEmbeddingLossTestCase> HingeEmbeddingLossTestConfigs()
{ // n c d h w margin
    return {
        {1, 1, 1, 1, 10, 1, 1, "mean"},
        {2, 1, 1, 10, 10, 1, 1, "mean"},
        {4, 1, 1, 100, 100, 1, 1, "sum"},
        {8, 3, 1, 20, 100, 1, 1, "mean"},
        {8, 3, 1, 50, 50, 1, 1, "sum"},
        {4, 3, 1, 60, 50, 1, 1, "mean"},
        {1, 1, 1, 1, 5000, 2, 1, "sum"},
        {3, 2, 4, 3, 100, 2, 1, "mean"},
    };
}

template <typename TIO, typename TT>
struct HingeEmbeddingLossUnreducedFwdTest
    : public ::testing::TestWithParam<HingeEmbeddingLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        auto in_dims      = config.GetInput();
        input             = tensor<TIO>{in_dims}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_descreet_unsigned<TT>(1, 2) * 2 - 1; };
        target             = tensor<TT>{in_dims}.generate(tar_gen_value);

        output = tensor<TIO>{in_dims};
        std::fill(output.begin(), output.end(), 0);

        ref_output = tensor<TIO>{in_dims};
        std::fill(ref_output.begin(), ref_output.end(), 0);

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        cpu_hinge_embedding_loss_unreduced_forward<TIO, TT>(
            input, target, ref_output, config.margin);
        status = miopen::HingeEmbeddingLossUnreducedForward(handle,
                                                            input.desc,
                                                            input_dev.get(),
                                                            target.desc,
                                                            target_dev.get(),
                                                            output.desc,
                                                            output_dev.get(),
                                                            config.margin);
        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<TIO>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<TIO>::epsilon();

        auto error = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error: " << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    HingeEmbeddingLossTestCase config;

    tensor<TIO> input;
    tensor<TT> target;
    tensor<TIO> output;

    tensor<TIO> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    float margin;
    float divisor;
};

template <typename TIO, typename TT>
struct HingeEmbeddingLossUnreducedBwdTest
    : public ::testing::TestWithParam<HingeEmbeddingLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto in_dims      = config.GetInput();
        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        input             = tensor<TIO>{in_dims}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_descreet_unsigned<TT>(1, 2) * 2 - 1; };
        target             = tensor<TT>{in_dims}.generate(tar_gen_value);

        auto dOut_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        dOutput             = tensor<TIO>{in_dims}.generate(dOut_gen_value);

        dInput = tensor<TIO>{in_dims};
        std::fill(dInput.begin(), dInput.end(), 0);

        ref_dInput = tensor<TIO>{in_dims};
        std::fill(ref_dInput.begin(), ref_dInput.end(), 0);

        input_dev   = handle.Write(input.data);
        target_dev  = handle.Write(target.data);
        dOutput_dev = handle.Write(dOutput.data);
        dInput_dev  = handle.Write(dInput.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        cpu_hinge_embedding_loss_unreduced_backward<TIO, TT>(
            input, target, dOutput, ref_dInput, config.margin);
        status = miopen::HingeEmbeddingLossUnreducedBackward(handle,
                                                             input.desc,
                                                             input_dev.get(),
                                                             target.desc,
                                                             target_dev.get(),
                                                             dOutput.desc,
                                                             dOutput_dev.get(),
                                                             dInput.desc,
                                                             dInput_dev.get(),
                                                             config.margin);
        EXPECT_EQ(status, miopenStatusSuccess);

        dInput.data = handle.Read<TIO>(dInput_dev, dInput.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<TIO>::epsilon();

        auto error = miopen::rms_range(ref_dInput, dInput);

        EXPECT_TRUE(miopen::range_distance(ref_dInput) == miopen::range_distance(dInput));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error: " << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    HingeEmbeddingLossTestCase config;

    tensor<TIO> input;
    tensor<TT> target;
    tensor<TIO> dOutput;
    tensor<TIO> dInput;

    tensor<TIO> ref_dInput;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr dOutput_dev;
    miopen::Allocator::ManageDataPtr dInput_dev;

    float margin;
};

template <typename TIO, typename TT>
struct HingeEmbeddingLossFwdTest : public ::testing::TestWithParam<HingeEmbeddingLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto in_dims      = config.GetInput();
        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        input             = tensor<TIO>{in_dims}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_descreet_unsigned<TT>(1, 2) * 2 - 1; };
        target             = tensor<TT>{in_dims}.generate(tar_gen_value);

        size_t workspaceSizeBytes = miopen::GetHingeEmbeddingLossForwardWorkspaceSize(
            handle, input.desc, target.desc, output.desc);
        size_t workspaceElements = workspaceSizeBytes / sizeof(TIO);

        workspace = tensor<TIO>(workspaceElements);
        std::fill(workspace.begin(), workspace.end(), 0);

        output = tensor<TIO>(1);
        std::fill(output.begin(), output.end(), 0);

        ref_output = tensor<TIO>(1);
        std::fill(ref_output.begin(), ref_output.end(), 0);

        if(config.reduction == "mean")
        {
            config.divisor *= input.desc.GetElementSize();
        }
        input_dev     = handle.Write(input.data);
        target_dev    = handle.Write(target.data);
        workspace_dev = handle.Write(workspace.data);
        output_dev    = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        cpu_hinge_embedding_loss_forward<TIO, TT>(
            input, target, workspace, ref_output, config.margin, config.divisor);
        status = miopen::HingeEmbeddingLossForward(handle,
                                                   workspace_dev.get(),
                                                   workspace.GetDataByteSize(),
                                                   input.desc,
                                                   input_dev.get(),
                                                   target.desc,
                                                   target_dev.get(),
                                                   output.desc,
                                                   output_dev.get(),
                                                   config.margin,
                                                   config.divisor);
        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<TIO>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<TIO>::epsilon();

        auto error = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error: " << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    HingeEmbeddingLossTestCase config;

    tensor<TIO> input;
    tensor<TT> target;
    tensor<TIO> workspace;
    tensor<TIO> output;

    tensor<TIO> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    float margin;
};

template <typename TIO, typename TT>
struct HingeEmbeddingLossBwdTest : public ::testing::TestWithParam<HingeEmbeddingLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();
        auto in_dims  = config.GetInput();

        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        input             = tensor<TIO>{in_dims}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_descreet_unsigned<TT>(1, 2) * 2 - 1; };
        target             = tensor<TT>{in_dims}.generate(tar_gen_value);

        dOutput    = tensor<TIO>(1);
        dOutput[0] = prng::gen_descreet_uniform_sign<TIO>(0.1, 50);

        dInput = tensor<TIO>{in_dims};
        std::fill(dInput.begin(), dInput.end(), 0);

        ref_dInput = tensor<TIO>{in_dims};
        std::fill(ref_dInput.begin(), ref_dInput.end(), 0);

        if(config.reduction == "mean")
        {
            config.divisor *= input.desc.GetElementSize();
        }
        input_dev   = handle.Write(input.data);
        target_dev  = handle.Write(target.data);
        dOutput_dev = handle.Write(dOutput.data);
        dInput_dev  = handle.Write(dInput.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        cpu_hinge_embedding_loss_backward<TIO, TT>(
            input, target, dOutput, ref_dInput, config.margin, config.divisor);
        status = miopen::HingeEmbeddingLossBackward(handle,
                                                    input.desc,
                                                    input_dev.get(),
                                                    target.desc,
                                                    target_dev.get(),
                                                    dOutput.desc,
                                                    dOutput_dev.get(),
                                                    dInput.desc,
                                                    dInput_dev.get(),
                                                    config.margin,
                                                    config.divisor);
        EXPECT_EQ(status, miopenStatusSuccess);

        dInput.data = handle.Read<TIO>(dInput_dev, dInput.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<TIO>::epsilon();

        auto error = miopen::rms_range(ref_dInput, dInput);

        EXPECT_TRUE(miopen::range_distance(ref_dInput) == miopen::range_distance(dInput));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error: " << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    HingeEmbeddingLossTestCase config;

    tensor<TIO> input;
    tensor<TT> target;
    tensor<TIO> dOutput;
    tensor<TIO> dInput;

    tensor<TIO> ref_dInput;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr dOutput_dev;
    miopen::Allocator::ManageDataPtr dInput_dev;

    float margin;
    float divisor;
};
