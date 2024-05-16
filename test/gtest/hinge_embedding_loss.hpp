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
        {1, 1, 1, 1, 10, 1},
        {2, 1, 1, 10, 10, 1},
        {4, 1, 1, 100, 100, 1},
        {8, 3, 1, 20, 100, 1},
        {16, 3, 1, 100, 100, 1},
        {1, 1, 1, 1, 5000, 2},
    };
}

template <typename TIO, typename TT>
struct HingeEmbeddingLossUnreducedForwardTest
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
        status = miopen::HingeEmbeddingLossReducedForward(handle,
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
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<TIO, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<TIO, bfloat16>::value)
            tolerance *= 8.0;

        auto error = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < tolerance)
            << "Error output beyond tolerance Error: " << error << ",  Tolerance: " << tolerance;
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
};
