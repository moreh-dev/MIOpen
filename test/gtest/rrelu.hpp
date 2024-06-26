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

#include "cpu_rrelu.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/rrelu.hpp>

template <typename T>
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

struct RReLUTestCase
{
    std::vector<size_t> lengths;
    float lower;
    float upper;
    bool contiguous;

    friend std::ostream& operator<<(std::ostream& os, const RReLUTestCase& tc)
    {
        return os << " Lengths:" << tc.lengths << " lower:" << tc.lower << " upper:" << tc.upper
                  << " Contiguous:" << (tc.contiguous ? "True" : "False");
    }
};

inline std::vector<RReLUTestCase> RReLUTestConfigs()
{
    std::vector<RReLUTestCase> tcs;
    tcs.push_back({{10}, 1.0f / 8, 1.0f / 3, true});
    tcs.push_back({{10}, 1.0f / 8, 1.0f / 3, false});

    tcs.push_back({{512, 64, 112}, 1.0f / 8, 1.0f / 3, true});
    tcs.push_back({{512, 64, 112}, 1.0f / 8, 1.0f / 3, false});

    tcs.push_back({{512, 64, 112, 112}, 1.0f / 8, 1.0f / 3, true});
    tcs.push_back({{512, 64, 56, 56}, 1.0f / 8, 1.0f / 3, true});
    tcs.push_back({{512, 128, 56, 56}, 1.0f / 8, 1.0f / 3, true});
    tcs.push_back({{512, 128, 28, 28}, 1.0f / 8, 1.0f / 3, true});
    tcs.push_back({{512, 256, 28, 28}, 1.0f / 8, 1.0f / 3, true});

    tcs.push_back({{512, 64, 112, 112}, 1.0f / 8, 1.0f / 3, false});
    tcs.push_back({{512, 64, 56, 56}, 1.0f / 8, 1.0f / 3, false});
    tcs.push_back({{512, 128, 56, 56}, 1.0f / 8, 1.0f / 3, false});
    tcs.push_back({{512, 128, 28, 28}, 1.0f / 8, 1.0f / 3, false});
    tcs.push_back({{512, 256, 28, 28}, 1.0f / 8, 1.0f / 3, false});

    return tcs;
}

inline std::vector<size_t> GetStrides(std::vector<size_t> lengths, bool contiguous)
{
    if(!contiguous)
        std::swap(lengths.front(), lengths.back());
    std::vector<size_t> strides(lengths.size());
    strides.back() = 1;
    for(int i = lengths.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * lengths[i + 1];
    if(!contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename T = float>
struct RReLUTestFwd : public ::testing::TestWithParam<RReLUTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        rrelu_config   = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto lengths    = rrelu_config.lengths;
        bool contiguous = rrelu_config.contiguous;

        auto input_strides = GetStrides(lengths, contiguous);
        input              = tensor<T>{lengths, input_strides}.generate(gen_value);

        auto out_strides = GetStrides(lengths, contiguous);
        output           = tensor<T>{lengths, out_strides};
        fill(output.begin(), output.end(), 1.0f);
        ref_output = tensor<T>{lengths, out_strides};

        noise     = tensor<float>{lengths};
        ref_noise = tensor<float>{lengths};

        state_sizeInBytes = miopen::GetRReLUStatesSize(handle);
        if(state_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();
        if(state_sizeInBytes != 0)
        {
            states     = std::vector<prngStates>(state_sizeInBytes / sizeof(prngStates));
            states_dev = handle.Write(states);
            auto status =
                miopen::RReLUStatesInit(handle, states_dev.get(), state_sizeInBytes, 2024);
            EXPECT_EQ(status, miopenStatusSuccess);
            states = handle.Read<prngStates>(states_dev, states.size());
        }

        ws_sizeInBytes = miopen::GetRReLUForwardWorkspaceSize(handle, input.desc, output.desc);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();
        if(ws_sizeInBytes != 0)
        {
            std::vector<size_t> workspace_dims;
            workspace_dims.push_back(ws_sizeInBytes / sizeof(T));
            workspace     = tensor<T>{workspace_dims};
            workspace_dev = handle.Write(workspace.data);
        }

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
        noise_dev  = handle.Write(noise.data);
    }

    void RunTest()
    {
        cpu_rrelu_forward5d<T>(
            states, input, ref_output, ref_noise, rrelu_config.lower, rrelu_config.upper);

        auto&& handle = get_handle();
        miopenStatus_t status;

        status = miopen::RReLUForward(handle,
                                      workspace_dev.get(),
                                      ws_sizeInBytes,
                                      states_dev.get(),
                                      state_sizeInBytes,
                                      input.desc,
                                      input_dev.get(),
                                      output.desc,
                                      output_dev.get(),
                                      noise.desc,
                                      noise_dev.get(),
                                      rrelu_config.lower,
                                      rrelu_config.upper);

        EXPECT_EQ(status, miopenStatusSuccess);
        output.data = handle.Read<T>(output_dev, output.data.size());
        noise.data  = handle.Read<float>(noise_dev, noise.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;

        auto error_noise = miopen::rms_range(ref_noise, noise);
        EXPECT_TRUE(miopen::range_distance(ref_noise) == miopen::range_distance(noise));
        EXPECT_TRUE(error_noise < tolerance)
            << "Error forward noise beyond tolerance Error: " << error_noise
            << ", Tolerance: " << tolerance;

        auto error = miopen::rms_range(ref_output, output);
        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < tolerance) << "Error forward output beyond tolerance Error: " << error
                                       << ", Tolerance: " << tolerance;
    }

    RReLUTestCase rrelu_config;

    tensor<T> input;
    tensor<T> output;
    tensor<float> noise;
    std::vector<prngStates> states;
    tensor<T> workspace;

    tensor<T> ref_output;
    tensor<float> ref_noise;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr noise_dev;
    miopen::Allocator::ManageDataPtr states_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t state_sizeInBytes;
    size_t ws_sizeInBytes;
};

template <typename T = float>
struct RReLUTestBwd : public ::testing::TestWithParam<RReLUTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        rrelu_config   = GetParam();
        auto gen_value = [](auto...) {
            return static_cast<float>(prng::gen_descreet_uniform_sign<T>(1e-2, 100)) + 3;
        };

        auto lengths    = rrelu_config.lengths;
        bool contiguous = rrelu_config.contiguous;

        noise = tensor<float>{lengths}.generate(gen_value);

        auto input_strides = GetStrides(lengths, contiguous);
        dinput             = tensor<T>{lengths, input_strides};
        ref_dinput         = tensor<T>{lengths, input_strides};

        auto out_strides = GetStrides(lengths, true);
        doutput          = tensor<T>{lengths, out_strides};
        fill(doutput.begin(), doutput.end(), 1.0f);

        noise_dev   = handle.Write(noise.data);
        doutput_dev = handle.Write(doutput.data);
        dinput_dev  = handle.Write(dinput.data);
    }

    void RunTest()
    {
        cpu_rrelu_backward5d<T>(noise, doutput, ref_dinput);

        auto&& handle = get_handle();
        miopenStatus_t status;

        status = miopen::RReLUBackward(handle,
                                       noise.desc,
                                       noise_dev.get(),
                                       doutput.desc,
                                       doutput_dev.get(),
                                       dinput.desc,
                                       dinput_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);
        dinput.data = handle.Read<T>(dinput_dev, dinput.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;

        auto error = miopen::rms_range(ref_dinput, dinput);
        EXPECT_TRUE(miopen::range_distance(ref_dinput) == miopen::range_distance(dinput));
        EXPECT_TRUE(error < tolerance) << "Error backward dinput beyond tolerance Error: " << error
                                       << ", Tolerance: " << tolerance;
    }

    RReLUTestCase rrelu_config;

    tensor<float> noise;
    tensor<T> doutput;
    tensor<T> dinput;

    tensor<T> ref_dinput;

    miopen::Allocator::ManageDataPtr noise_dev;
    miopen::Allocator::ManageDataPtr doutput_dev;
    miopen::Allocator::ManageDataPtr dinput_dev;
};
