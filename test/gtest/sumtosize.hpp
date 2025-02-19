/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include "cpu_sumtosize.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/sumtosize.hpp>

struct SumToSizeTestCase
{
    std::vector<size_t> input_dims;
    std::vector<size_t> output_dims;

    friend std::ostream& operator<<(std::ostream& os, const SumToSizeTestCase& tc)
    {
        os << "input_dims:";
        os << tc.input_dims[0];
        for(int i = 1; i < tc.input_dims.size(); i++)
            os << "x" << tc.input_dims[i];
        os << ", output_dims:" << tc.output_dims[0];
        for(int i = 1; i < tc.output_dims.size(); i++)
            os << "x" << tc.output_dims[i];
        return os;
    }
};

inline std::vector<SumToSizeTestCase> SumToSizeTestConfigs()
{
    // clang-format off
    return {
        {{8, 120, 1}, {1, 120, 1}}, 
        {{8, 120, 50265}, {1, 120, 50265}},
        {{8, 1023, 1}, {1, 1023, 1}},
        {{8, 1023, 50257}, {1, 1023, 50257}},
        {{16384, 768}, {1, 768}},
        {{8192, 3072}, {1, 3072}},
        {{8192, 2304}, {1, 2304}},
        {{16, 512, 1024}, {1, 512, 1024}},
        {{48, 8, 512, 512}, {1, 8, 512, 512}},
        {{40, 512, 768}, {1, 512, 768}},
        {{256, 4, 8732}, {256, 1, 8732}},
        {{841, 64, 2, 1024}, {841, 64, 1, 1024}},
        {{16, 311, 99, 512}, {16, 311, 1, 512}},
        {{1, 128, 512}, {1, 1, 512}},
        {{64, 8, 128, 128}, {1, 8, 128, 128}},
        {{64, 128, 768}, {1, 128, 768}},
        {{48, 512, 512}, {1, 1, 512}},
        {{16, 16, 16, 16, 16}, {1, 1, 1, 1, 16}}
    };
    // clang-format on
}

template <typename T = float>
struct SumToSizeForwardTest : public ::testing::TestWithParam<SumToSizeTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        input             = tensor<T>{config.input_dims};
        auto gen_in_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(0), static_cast<T>(1));
        };
        std::generate(input.begin(), input.end(), gen_in_value);
        input_dev = handle.Write(input.data);

        output     = tensor<T>{config.output_dims};
        output_dev = handle.Create<T>(output.GetSize());
        ref_output = tensor<T>{config.output_dims};

        ws_sizeInBytes = miopen::GetSumToSizeForwardWorkspaceSize(handle, input.desc, output.desc);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_FAIL() << "Call GetSumToSizeForwardWorkspaceSize failed!";
        if(ws_sizeInBytes > 0)
        {
            workspace_dev = handle.Create<std::byte>(ws_sizeInBytes);
        }
        else
        {
            workspace_dev = nullptr;
        }
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_sumtosize_forward<T>(input, ref_output);

        miopenStatus_t status;
        status = miopen::SumToSizeForward(handle,
                                          input.desc,
                                          input_dev.get(),
                                          output.desc,
                                          output_dev.get(),
                                          workspace_dev.get(),
                                          ws_sizeInBytes);

        ASSERT_EQ(status, miopenStatusSuccess);

        // Write from GPU to CPU
        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        auto tolerance = std::numeric_limits<T>::epsilon() * 10;

        auto error = miopen::rms_range(ref_output, output);
        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error, tolerance);
    }
    SumToSizeTestCase config;

    tensor<T> input;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;
};
