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
#include "cpu_take.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/take.hpp>

struct TakeTestCase
{
    size_t in_N;
    size_t in_C;
    size_t in_D;
    size_t in_H;
    size_t in_W;
    size_t out_N;
    size_t out_C;
    size_t out_D;
    size_t out_H;
    size_t out_W;
    friend std::ostream& operator<<(std::ostream& os, const TakeTestCase& tc)
    {
        return os << " in_N:" << tc.in_N << " in_C:" << tc.in_C << " in_D:" << tc.in_D
                  << " in_H:" << tc.in_H << " in_W:" << tc.in_W << " out_N:" << tc.out_N
                  << " out_C:" << tc.out_C << " out_D:" << tc.out_D << " out_H:" << tc.out_H
                  << " out_W:" << tc.out_W;
    }

    std::vector<size_t> GetInput()
    {
        if((in_N != 0) && (in_C != 0) && (in_D != 0) && (in_H != 0) && (in_W != 0))
        {
            return std::vector<size_t>({in_N, in_C, in_D, in_H, in_W});
        }
        else if((in_N != 0) && (in_C != 0) && (in_H != 0) && (in_W != 0))
        {
            return std::vector<size_t>({in_N, in_C, in_H, in_W});
        }
        else if((in_N != 0) && (in_C != 0) && (in_W != 0))
        {
            return std::vector<size_t>({in_N, in_C, in_W});
        }
        else if((in_N != 0) && (in_W != 0))
        {
            return std::vector<size_t>({in_N, in_W});
        }
        else if((in_N != 0))
        {
            return std::vector<size_t>({in_N});
        }
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }

    std::vector<size_t> GetOutput()
    {
        if((out_N != 0) && (out_C != 0) && (out_D != 0) && (out_H != 0) && (out_W != 0))
        {
            return std::vector<size_t>({out_N, out_C, out_D, out_H, out_W});
        }
        else if((out_N != 0) && (out_C != 0) && (out_H != 0) && (out_W != 0))
        {
            return std::vector<size_t>({out_N, out_C, out_H, out_W});
        }
        else if((out_N != 0) && (out_C != 0) && (out_W != 0))
        {
            return std::vector<size_t>({out_N, out_C, out_W});
        }
        else if((out_N != 0) && (out_W != 0))
        {
            return std::vector<size_t>({out_N, out_W});
        }
        else if((out_N != 0))
        {
            return std::vector<size_t>({out_N});
        }
        else
        {
            std::cout << "Error Output Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }
};

std::vector<TakeTestCase> TakeTestConfigs()
{
    // clang-format off
    return {
        { 6, 0, 0, 0, 0,
          4, 0, 0, 0, 0}, // small test
        { 8, 120, 0, 0, 1,    
             96, 0, 0, 0, 0},  // 3d input, 1d output
        { 8, 120, 0, 0, 1,
             2, 30, 0, 0, 1}, // 3d input, 3d output
        { 8,    1023, 0,  0,   1,
             409, 0, 0, 0, 5}, // 3d input, 2d output  
        { 8,    1024, 0,  0,   768,
             1000, 20, 0, 4, 7}, // 3d input, 4d output
        { 8,    1023, 0,  0,   1,
             123, 22, 0, 0, 3}, // 3d input, 3d output
        { 8,    1024, 0,  0,   768,
             2516582, 0, 0, 0, 0}, // 3d input, 1d output
        { 16,   1024, 0,  0,   768,
             2200000,  2, 0, 0, 2},  // output = 70% element of input
        { 16,   1024, 0,  0,   768,
             1, 0, 0, 0, 0}, // 1 output element
        { 48,   8,    0,  512, 512,
             65536, 16, 0, 4, 4}, // 4d input, 4d output 
        { 48,   8,    0,  512, 512,
             16356, 64, 0, 0, 8}, // 4d input, 3d output
        { 16, 311,    0,  98,  512,
              48662, 88, 0, 2, 5},  // 4d input, 4d output
        { 16, 311,    0,  98,  512, 
              1234, 321, 0, 23, 21} // 4d input, 4d output
      };
    // clang-format on
}

template <typename T = float>
struct TakeTest : public ::testing::TestWithParam<TakeTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        take_config   = GetParam();

        auto in_dims   = take_config.GetInput();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        input          = tensor<T>{in_dims}.generate(gen_value);

        int32_t input_numel =
            std::accumulate(in_dims.begin(), in_dims.end(), 1ULL, std::multiplies<size_t>());
        auto out_dims = take_config.GetOutput();
        int32_t output_numel =
            std::accumulate(out_dims.begin(), out_dims.end(), 1ULL, std::multiplies<size_t>());

        index = tensor<int32_t>{out_dims};
        for(auto i = 0; i < output_numel; i++)
        {
            // Generate random index elements from [-input_numel, input_numel)
            index[i] = prng::gen_descreet_uniform_sign<int32_t>(1, input_numel);
        }

        output = tensor<T>{out_dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        index_dev  = handle.Write(index.data);
        output_dev = handle.Write(output.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_take_forward<T>(input, ref_output, index);

        miopenStatus_t status;

        status = miopen::TakeForward(handle,
                                     input.desc,
                                     input_dev.get(),
                                     index.desc,
                                     index_dev.get(),
                                     output.desc,
                                     output_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error:" << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    TakeTestCase take_config;

    tensor<T> input;
    tensor<int32_t> index;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr index_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};
