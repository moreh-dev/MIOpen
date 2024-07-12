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
#include "cpu_repeat.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/repeat.hpp>

struct RepeatTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    int32_t* sizes;
    int32_t num_sizes;
    friend std::ostream& operator<<(std::ostream& os, const RepeatTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " sizes:" << tc.sizes << " num_sizes:" << tc.num_sizes;
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
        else if((N != 0) && (C != 0))
        {
            return std::vector<size_t>({N, C});
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

std::vector<RepeatTestCase> RepeatTestConfigs()
{ // n c d h w sizes num_sizes
    return {
        {1, 1, 0, 0, 512, new int32_t[3]{48, 512, 1}, 3},        // stdc
        {1, 1, 0, 0, 512, new int32_t[3]{32, 512, 1}, 3},        // llama2
        {10, 1, 0, 100, 1, new int32_t[4]{10, 32, 1, 128}, 4},   // t5_3b
        {10, 1, 0, 149, 1, new int32_t[4]{10, 32, 1, 128}, 4},   // t5_3b
        {10, 1, 1, 100, 1, new int32_t[5]{10, 32, 2, 1, 64}, 5}, // llama2_7b
        {1, 1, 0, 0, 1, new int32_t[3]{10, 1, 1}, 3},            // llama2_7b
        {3, 0, 0, 0, 0, new int32_t[2]{4, 2}, 2},                // Custom Test Case ~
        {16, 0, 0, 0, 0, new int32_t[2]{16, 32}, 2},
        {32, 24, 0, 0, 0, new int32_t[3]{26, 32, 24}, 3},
        {16, 16, 0, 0, 0, new int32_t[3]{32, 24, 24}, 3},
        {24, 28, 0, 4, 0, new int32_t[4]{24, 24, 2, 3}, 4},
        {16, 16, 0, 24, 0, new int32_t[4]{3, 2, 2, 2}, 4},
        {1, 2, 3, 4, 5, new int32_t[5]{2, 2, 2, 2, 2}, 5},
        {2, 3, 4, 5, 6, new int32_t[5]{1, 2, 1, 2, 1}, 5},
        {10, 10, 0, 0, 0, new int32_t[3]{10, 10, 10}, 3},
        {20, 0, 0, 0, 0, new int32_t[2]{5, 4}, 2},
        {8, 16, 0, 0, 0, new int32_t[4]{2, 2, 2, 2}, 4},
        {7, 7, 7, 7, 7, new int32_t[5]{1, 1, 1, 1, 1}, 5},
        {100, 0, 0, 0, 0, new int32_t[2]{10, 10}, 2},
        {5, 10, 0, 20, 15, new int32_t[4]{5, 5, 5, 5}, 4},
        {6, 6, 6, 6, 6, new int32_t[5]{6, 6, 6, 6, 6}, 5},
        {4, 4, 0, 0, 0, new int32_t[3]{2, 2, 2}, 3},
        {12, 12, 0, 0, 0, new int32_t[4]{1, 2, 3, 4}, 4},
        {3, 5, 7, 9, 11, new int32_t[5]{2, 3, 4, 5, 6}, 5},
        {2, 2, 2, 2, 2, new int32_t[5]{2, 2, 2, 2, 2}, 5},
    };
}

template <typename T = float>
struct RepeatForwardTest : public ::testing::TestWithParam<RepeatTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        repeat_config  = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        sizes     = repeat_config.sizes;
        num_sizes = repeat_config.num_sizes;

        auto in_dims = repeat_config.GetInput();

        input = tensor<T>{in_dims}.generate(gen_value);

        std::vector<size_t> out_dims;
        int64_t offset = static_cast<int64_t>(num_sizes) - static_cast<int64_t>(in_dims.size());

        out_dims.resize(num_sizes);

        for(int i = 0; i < offset; ++i)
        {
            out_dims[i] = sizes[i];
        }

        for(size_t i = 0; i < in_dims.size(); ++i)
        {
            out_dims[offset + i] = in_dims[i] * sizes[offset + i];
        }

        output = tensor<T>{out_dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_repeat_forward<T>(input, ref_output, sizes, num_sizes);
        miopenStatus_t status;

        status = miopen::RepeatForward(
            handle, input.desc, input_dev.get(), sizes, num_sizes, output.desc, output_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        auto threshold = std::is_same<T, float>::value ? 1.5e-5 : 8.2e-2;

        if(std::is_same<T, bfloat16>::value)
            threshold *= 8.0;
        auto error = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold)
            << "Error output beyond tolerance Error:" << error << ",   Threshold " << threshold;
    }
    RepeatTestCase repeat_config;

    tensor<T> input;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    int32_t* sizes;
    int32_t num_sizes;
};

template <typename T = float>
struct RepeatBackwardTest : public ::testing::TestWithParam<RepeatTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        repeat_config  = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        sizes     = repeat_config.sizes;
        num_sizes = repeat_config.num_sizes;

        auto in_dims = repeat_config.GetInput();

        input_grad = tensor<T>{in_dims};

        std::vector<size_t> out_dims;
        int64_t offset = static_cast<int64_t>(num_sizes) - static_cast<int64_t>(in_dims.size());

        out_dims.resize(num_sizes);

        for(int i = 0; i < offset; ++i)
        {
            out_dims[i] = sizes[i];
        }

        for(size_t i = 0; i < in_dims.size(); ++i)
        {
            out_dims[offset + i] = in_dims[i] * sizes[offset + i];
        }

        output_grad = tensor<T>{out_dims}.generate(gen_value);

        std::fill(input_grad.begin(), input_grad.end(), T(0));
        ref_input_grad = tensor<T>{in_dims};
        std::fill(ref_input_grad.begin(), ref_input_grad.end(), T(0));

        input_grad_dev  = handle.Write(input_grad.data);
        output_grad_dev = handle.Write(output_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_repeat_backward<T>(output_grad, ref_input_grad, sizes, num_sizes);
        miopenStatus_t status;

        status = miopen::RepeatBackward(handle,
                                        output_grad.desc,
                                        output_grad_dev.get(),
                                        sizes,
                                        num_sizes,
                                        input_grad.desc,
                                        input_grad_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    void Verify()
    {
        auto threshold = std::is_same<T, float>::value ? 1.5e-5 : 8.2e-2;

        if(std::is_same<T, bfloat16>::value)
            threshold *= 8.0;
        auto error = miopen::rms_range(ref_input_grad, input_grad);

        EXPECT_TRUE(miopen::range_distance(ref_input_grad) == miopen::range_distance(input_grad));
        EXPECT_TRUE(error < threshold * 10)
            << "Error output beyond tolerance Error:" << error << ",   Threshold: " << threshold;
    }
    RepeatTestCase repeat_config;

    tensor<T> input_grad;
    tensor<T> output_grad;

    tensor<T> ref_input_grad;

    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;

    int32_t* sizes;
    int32_t num_sizes;
};
