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

#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "cpu_pad_constant.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/pad_constant.hpp>

struct PadConstantTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;

    friend std::ostream& operator<<(std::ostream& os, const PadConstantTestCase& tc)
    {
        return os << "(N: " << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << ")";
    }

    std::vector<size_t> GetInput()
    {
        if((N != 0) && (C != 0) && (D != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, D, H, W});
        }
        else if((N != 0) && (C != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, 1, H, W});
        }
        else if((N != 0) && (C != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, 1, 1, W});
        }
        else if((N != 0) && (W != 0))
        {
            return std::vector<size_t>({N, 1, 1, 1, W});
        }
        else if(N != 0)
        {
            return std::vector<size_t>({N, 1, 1, 1, 1});
        }
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }
};

std::vector<PadConstantTestCase> PadConstantTestConfigs()
{
    // clang-format off
    return {
        { 8,    120,  0,  0,   1},  
        { 8,    120,  0,  0,   1},
        { 8,    1023, 0,  0,   1},  
        { 8,    1024, 0,  0,   768},
        { 8,    1023, 0,  0,   1},
        { 8,    1024, 0,  0,   768},
        { 16,   1024, 0,  0,   768},  
        { 16,   1024, 0,  0,   768},
        { 48,   8,    0,  512, 512}, 
        { 48,   8,    0,  512, 512},
        { 16, 311,    0,  98,  512},
        { 16, 311,    0,  98,  512}
    };
    // clang-format on
}

template <typename T>
struct PadConstantTest : public ::testing::TestWithParam<PadConstantTestCase>
{
protected:
    PadConstantTestCase pad_constant_config;
    tensor<T> input;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    size_t padding[10];

    void SetUp() override
    {
        auto&& handle       = get_handle();
        pad_constant_config = GetParam();
        auto gen_value      = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto in_dims = pad_constant_config.GetInput();
        input        = tensor<T>{in_dims}.generate(gen_value);
        input_dev    = handle.Write(input.data);

        // Generate random padding
        for(size_t& i : padding)
        {
            i = prng::gen_descreet_unsigned<size_t>(1, 5);
        }

        std::vector<size_t> out_dims;
        for(size_t i = 0; i < 5; i++)
        {
            out_dims.push_back(in_dims[i] + 2 * padding[2 * i]);
        }

        output = tensor<T>{out_dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());
        output_dev = handle.Write(output.data);

        ref_output = tensor<T>{out_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());
    };

    void RunTest()
    {
        auto&& handle = get_handle();
        auto out_dims = output.desc.GetLengths();

        float padding_value = 3.5f;

        cpu_pad_constant_fwd<T>(input.data.data(),
                                ref_output.data.data(),
                                &input.desc,
                                &output.desc,
                                padding,
                                padding_value);
        miopenStatus_t status;

        status = miopen::PadConstantForward(handle,
                                            input.desc,
                                            output.desc,
                                            input_dev.get(),
                                            output_dev.get(),
                                            padding,
                                            padding_value);
        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        auto error = miopen::rms_range(ref_output, output);
        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error == 0) << "Outputs do not match each other. Error:" << error;
    }
};
