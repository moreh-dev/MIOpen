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
#include "cpu_sigmoid_focal_loss.hpp"
#include "get_handle.hpp"
#include "miopen/allocator.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <cstddef>
#include <cstdlib>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/sigmoid_focal_loss.hpp>

struct SigmoidFocalLossTestCase
{
    std::vector<size_t> dims;
    float alpha = 0.25;
    float gamma = 2;
    miopenLossReductionMode_t reduction;
    friend std::ostream& operator<<(std::ostream& os, const SigmoidFocalLossTestCase& tc)
    {
        os << "dims: ";
        for(auto dim : tc.dims)
        {
            os << dim << " ";
        }
        return os << "alpha:" << tc.alpha << " gamma:" << tc.gamma << " reduction:" << tc.reduction;
    }

    std::vector<size_t> GetDims() const { return dims; }

    SigmoidFocalLossTestCase() {}

    SigmoidFocalLossTestCase(std::vector<size_t> dim_,
                             miopenLossReductionMode_t reduction_ = MIOPEN_LOSS_REDUCTION_NONE,
                             float alpha_                         = 0.25,
                             float gamma_                         = 2)
        : dims(dim_), alpha(alpha_), gamma(gamma_), reduction(reduction_)
    {
    }
};

inline std::vector<SigmoidFocalLossTestCase> SigmoidFocalLossTestConfigs()
{
    return {
        SigmoidFocalLossTestCase({4000}),            // 1D cont
        SigmoidFocalLossTestCase({100, 500}),        // 2D cont
        SigmoidFocalLossTestCase({10, 20, 200}),     // 3D cont
        SigmoidFocalLossTestCase({8, 3, 20, 100}),   // 4D cont
        SigmoidFocalLossTestCase({2, 2, 3, 4, 100}), // 5D cont
    };
}

template <typename TIO>
struct SigmoidFocalLossUnreducedFwdTest : public ::testing::TestWithParam<SigmoidFocalLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto in_dims = config.GetDims();

        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        input             = tensor<TIO>{in_dims}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        target             = tensor<TIO>{in_dims}.generate(tar_gen_value);

        output = tensor<TIO>{in_dims};
        std::fill(output.begin(), output.end(), 0);

        outputHost = tensor<TIO>{in_dims};
        std::fill(outputHost.begin(), outputHost.end(), 0);

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        status = miopen::SigmoidFocalLossForward(handle,
                                                 nullptr,
                                                 0,
                                                 input.desc,
                                                 input_dev.get(),
                                                 target.desc,
                                                 target_dev.get(),
                                                 output.desc,
                                                 output_dev.get(),
                                                 config.alpha,
                                                 config.gamma,
                                                 config.reduction);
        cpu_sigmoid_focal_loss_unreduced_forward<TIO>(input, target, outputHost, config.alpha);

        EXPECT_EQ(status, miopenStatusSuccess);
        output.data = handle.Read<TIO>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<TIO>::epsilon();

        auto error = miopen::rms_range(outputHost, output);

        EXPECT_TRUE(miopen::range_distance(outputHost) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error: " << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    SigmoidFocalLossTestCase config;

    tensor<TIO> input;
    tensor<TIO> target;
    tensor<TIO> output;

    tensor<TIO> outputHost;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

template <typename TIO>
struct SigmoidFocalLossUnreducedBwdTest : public ::testing::TestWithParam<SigmoidFocalLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto in_dims      = config.GetDims();
        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        input             = tensor<TIO>{in_dims}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        target             = tensor<TIO>{in_dims}.generate(tar_gen_value);

        auto dOut_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        dOutput             = tensor<TIO>{in_dims}.generate(dOut_gen_value);

        dInput = tensor<TIO>{in_dims};
        std::fill(dInput.begin(), dInput.end(), 0);

        dInputHost = tensor<TIO>{in_dims};
        std::fill(dInputHost.begin(), dInputHost.end(), 0);

        input_dev   = handle.Write(input.data);
        target_dev  = handle.Write(target.data);
        dOutput_dev = handle.Write(dOutput.data);
        dInput_dev  = handle.Write(dInput.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        status = miopen::SigmoidFocalLossBackward(handle,
                                                  input.desc,
                                                  input_dev.get(),
                                                  target.desc,
                                                  target_dev.get(),
                                                  dOutput.desc,
                                                  dOutput_dev.get(),
                                                  dInput.desc,
                                                  dInput_dev.get(),
                                                  config.alpha,
                                                  config.gamma,
                                                  config.reduction);
        cpu_sigmoid_focal_loss_unreduced_backward<TIO>(
            input, target, dOutput, dInputHost, config.alpha, config.gamma);

        EXPECT_EQ(status, miopenStatusSuccess);

        dInput.data = handle.Read<TIO>(dInput_dev, dInput.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<TIO>::epsilon();

        auto error = miopen::rms_range(dInputHost, dInput);

        EXPECT_TRUE(miopen::range_distance(dInputHost) == miopen::range_distance(dInput));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error: " << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    SigmoidFocalLossTestCase config;

    tensor<TIO> input;
    tensor<TIO> target;
    tensor<TIO> dOutput;
    tensor<TIO> dInput;

    tensor<TIO> dInputHost;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr dOutput_dev;
    miopen::Allocator::ManageDataPtr dInput_dev;
};

template <typename TIO>
struct SigmoidFocalLossFwdTest : public ::testing::TestWithParam<SigmoidFocalLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        config.reduction = miopenLossReductionMode_t(int(prng::gen_0_to_B(2) + 1));

        auto in_dims = config.GetDims();

        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 20); };
        input             = tensor<TIO>{in_dims}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 20); };
        target             = tensor<TIO>{in_dims}.generate(tar_gen_value);

        size_t workspaceSizeBytes = miopen::GetSigmoidFocalLossForwardWorkspaceSize(
            handle, input.desc, target.desc, output.desc, config.reduction);
        size_t workspaceElements = workspaceSizeBytes / sizeof(TIO);

        workspace = tensor<TIO>(workspaceElements);
        std::fill(workspace.begin(), workspace.end(), 0);

        output = tensor<TIO>(1);
        std::fill(output.begin(), output.end(), 0);

        outputHost = tensor<TIO>(1);
        std::fill(outputHost.begin(), outputHost.end(), 0);

        divisor = 1;
        if(config.reduction == MIOPEN_LOSS_REDUCTION_MEAN)
        {
            divisor *= input.desc.GetElementSize();
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

        status = miopen::SigmoidFocalLossForward(handle,
                                                 workspace_dev.get(),
                                                 workspace.GetDataByteSize(),
                                                 input.desc,
                                                 input_dev.get(),
                                                 target.desc,
                                                 target_dev.get(),
                                                 output.desc,
                                                 output_dev.get(),
                                                 config.alpha,
                                                 config.gamma,
                                                 config.reduction);
        cpu_sigmoid_focal_loss_forward<TIO>(
            input, target, workspace, outputHost, config.alpha, config.gamma, divisor);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<TIO>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<TIO>::epsilon();

        auto error = miopen::rms_range(outputHost, output);

        EXPECT_TRUE(miopen::range_distance(outputHost) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10)
            << "Error output beyond tolerance Error: " << error
            << ",  Thresholdx10: " << threshold * 10 << " Reduction: " << config.reduction;
    }
    SigmoidFocalLossTestCase config;

    tensor<TIO> input;
    tensor<TIO> target;
    tensor<TIO> workspace;
    tensor<TIO> output;

    tensor<TIO> outputHost;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    float divisor;
};

template <typename TIO>
struct SigmoidFocalLossBwdTest : public ::testing::TestWithParam<SigmoidFocalLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();
        auto in_dims  = config.GetDims();

        config.reduction = miopenLossReductionMode_t(int(prng::gen_0_to_B(2) + 1));

        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        input             = tensor<TIO>{in_dims}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<TIO>(0.1, 50); };
        target             = tensor<TIO>{in_dims}.generate(tar_gen_value);

        dOutput    = tensor<TIO>(1);
        dOutput[0] = prng::gen_descreet_uniform_sign<TIO>(0.1, 50);

        dInput = tensor<TIO>{in_dims};
        std::fill(dInput.begin(), dInput.end(), 0);

        dinputHost = tensor<TIO>{in_dims};
        std::fill(dinputHost.begin(), dinputHost.end(), 0);

        divisor = 1;
        if(config.reduction == MIOPEN_LOSS_REDUCTION_MEAN)
        {
            divisor *= input.desc.GetElementSize();
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

        status = miopen::SigmoidFocalLossBackward(handle,
                                                  input.desc,
                                                  input_dev.get(),
                                                  target.desc,
                                                  target_dev.get(),
                                                  dOutput.desc,
                                                  dOutput_dev.get(),
                                                  dInput.desc,
                                                  dInput_dev.get(),
                                                  config.alpha,
                                                  config.gamma,
                                                  config.reduction);
        cpu_sigmoid_focal_loss_backward<TIO>(
            input, target, dOutput, dinputHost, config.alpha, config.gamma, divisor);

        EXPECT_EQ(status, miopenStatusSuccess);

        dInput.data = handle.Read<TIO>(dInput_dev, dInput.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<TIO>::epsilon();

        auto error = miopen::rms_range(dinputHost, dInput);

        EXPECT_TRUE(miopen::range_distance(dinputHost) == miopen::range_distance(dInput));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error: " << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    SigmoidFocalLossTestCase config;

    tensor<TIO> input;
    tensor<TIO> target;
    tensor<TIO> dOutput;
    tensor<TIO> dInput;

    tensor<TIO> dinputHost;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr dOutput_dev;
    miopen::Allocator::ManageDataPtr dInput_dev;

    float divisor;
};
