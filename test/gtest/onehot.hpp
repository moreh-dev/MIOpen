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
#include "cpu_onehot.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <cstdlib>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/onehot.hpp>

struct OneHotTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    int num_classes;
    long input_size;
    friend std::ostream& operator<<(std::ostream& os, const OneHotTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " num_classes:" << tc.num_classes;
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

std::vector<OneHotTestCase> OneHotTestConfigs()
{ // n c d h w
    return {
        {1, 0, 0, 0, 10, 10, 1},
        {2, 0, 0, 10, 10, 10, 1},
        {4, 1, 0, 100, 100, 100, 1},
        {8, 3, 1, 20, 100, 200, 1},
        {16, 3, 1, 100, 100, 200, 1},
        {1, 1, 1, 1, 5000, 5000, 1},
    };
}

template <typename T = int>
struct OneHotTest : public ::testing::TestWithParam<OneHotTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        onehot_config  = GetParam();
        int gen_range  = onehot_config.num_classes;
        auto gen_value = [&gen_range](auto...) { return prng::gen_A_to_B(0, gen_range - 1); };

        auto in_dims = onehot_config.GetInput();

        input = tensor<T>{in_dims}.generate(gen_value);

        std::vector<size_t> out_dims;

        for(unsigned long in_dim : in_dims)
        {
            out_dims.push_back(in_dim);
            onehot_config.input_size *= in_dim;
        }

        out_dims.push_back(onehot_config.num_classes);

        output = tensor<T>{out_dims};
        std::fill(output.begin(), output.end(), 0);

        ref_output = tensor<T>{out_dims};
        std::fill(ref_output.begin(), ref_output.end(), 0);

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        status = miopen::OneHot(handle,
                                input.desc,
                                input_dev.get(),
                                onehot_config.input_size,
                                output.desc,
                                output_dev.get(),
                                onehot_config.num_classes);

        EXPECT_EQ(status, miopenStatusSuccess);
        cpu_onehot<T>(input, ref_output, onehot_config.input_size, onehot_config.num_classes);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = 1e-6;
        auto error       = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold)
            << "Error output beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }
    OneHotTestCase onehot_config;

    tensor<T> input;
    tensor<T> output;
    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    long input_size;
    long num_classes;
};
