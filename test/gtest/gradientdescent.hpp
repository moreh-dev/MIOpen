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
#include "cpu_gradientdescent.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/gradientdescent.hpp>
#include <miopen/miopen.h>

template <class T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << '{';
    for(int i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

struct GradientDescentTestCase
{
    std::vector<size_t> dims;

    friend std::ostream& operator<<(std::ostream& os, const GradientDescentTestCase& tc)
    {
        return os << " dims:" << tc.dims;
    }
};

inline std::vector<GradientDescentTestCase> GradientDescentTestConfigs()
{ // n c d h w lr momentum dampening weightDecay nesterov momentumInitialized
    return {
        {{50, 10}},
        {{50, 10, 20}},
        {{50, 10, 20, 30}},
        {{50, 10, 20, 30, 4}},
    };
}

template <typename T = float>
struct GradientDescentTest : public ::testing::TestWithParam<GradientDescentTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle          = get_handle();
        GradientDescent_config = GetParam();

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto dims        = GradientDescent_config.dims;
        auto output_dims = dims;

        var_in   = tensor<T>{dims}.generate(gen_value);
        var_out  = tensor<T>{dims};
        alpha_in = tensor<T>{1}.generate(gen_value);
        delta_in = tensor<T>{dims}.generate(gen_value);

        ref_output = tensor<T>(dims);

        std::fill(output.begin(), output.end(), 0);
        std::fill(ref_output.begin(), ref_output.end(), 0);

        input_dev       = handle.Write(input.data);
        output_dev      = handle.Write(output.data);
        segment_ids_dev = handle.Write(segment_ids.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        cpu_GradientDescent_forward<T, int>(input, ref_output, segment_ids, num_segments);
        miopenStatus_t status = miopenStatusSuccess;

        status = miopen::GradientDescent::GradientDescent(handle,
                                                          input.desc,
                                                          input_dev.get(),
                                                          output.desc,
                                                          output_dev.get(),
                                                          segment_ids.desc,
                                                          segment_ids_dev.get());
        ASSERT_EQ(status, miopenStatusSuccess);
        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_output, output);

        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error, threshold * 10) << "Error output beyond tolerance Error:" << error
                                         << ",  Thresholdx10: " << threshold * 10;
    }
    GradientDescentTestCase GradientDescent_config;

    tensor<T> var_in;
    tensor<T> var_out;
    tensor<T> alpha_in;
    tensor<T> delta_in;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr var_in_dev;
    miopen::Allocator::ManageDataPtr var_out_dev;
    miopen::Allocator::ManageDataPtr alpha_in_dev;
    miopen::Allocator::ManageDataPtr delta_in_dev;
};
