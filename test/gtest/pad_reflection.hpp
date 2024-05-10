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
#include "cpu_pad_reflection.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/pad_reflection.hpp>

struct PadReflectionCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    size_t padding;
    int contiguous;
    friend std::ostream& operator<<(std::ostream& os, const PadReflectionCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " Padding:" << tc.padding << " Contiguous:" << tc.contiguous;
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

     std::vector<size_t> GetPadding() const
    {
        std::vector<size_t> paddingVector;
        paddingVector.push_back(padding);
        // for(int i = 0; i < 4; ++i)
        // {
        //     paddingVector.push_back(padding[i]);
        // }
        return paddingVector;
    }

    int GetContiguous() const 
    {
        return contiguous;
    }
};

std::vector<PadReflectionCase> PadReflectionTestFloatConfigs()
{ // n c d h w padding
    // clang-format off
    return {
        // {  1,   1,    0,    3,    3, {2}, 1},
        // { 48,   8,    0,  512,  512, {1}, 1},
        // { 48,   8,    0,  512,  512, {1, 1, 3, 3}, 1},
        // { 48,   8,    0,  512,  512, {0, 0, 2, 2}, 1},
        // { 16, 311,    0,   98,  512, {1}, 1},
        // { 16, 311,    0,   98,  512, {1, 1, 3, 3}, 1},
        // { 16, 311,    0,   98,  512, {0, 0, 2, 2}, 1},
        {  1,   1,    0,    0,    3, 2, 1},
        { 48,   8,    0,  0,  512, 1, 1},
        { 48,   8,    0,  0,  512, 3, 1},
        { 16, 311,    0,   0,  512, 1, 1},
        { 16, 311,    0,   0,  512, 3, 1},
        {  1,   1,    0,    0,    3, 2, 0},
        { 48,   8,    0,  0,  512, 1, 0},
        { 48,   8,    0,  0,  512, 3, 0},
        { 16, 311,    0,   0,  512, 1, 0},
        { 16, 311,    0,   0,  512, 3, 0},
      };
    // clang-format on
}

template <typename T>
inline std::vector<T> GetStrides(std::vector<T> input, bool contiguous)
{
    if(!contiguous)
        std::swap(input.front(), input.back());
    std::vector<T> strides(input.size());
    strides.back() = 1;
    for(int i = input.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * input[i + 1];
    if(!contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename T = float>
struct PadReflectionFwdTest : public ::testing::TestWithParam<PadReflectionCase>
{
protected:
    void SetUp() override
    {
        auto&& handle         = get_handle();
        pad_reflection_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto in_dims = pad_reflection_config.GetInput();
        auto padding = pad_reflection_config.GetPadding();
        auto contiguous = pad_reflection_config.GetContiguous();
        auto input_strides = GetStrides(in_dims, contiguous == 1);
        input        = tensor<T>{in_dims, input_strides}.generate(gen_value);
        std::vector<size_t> out_dims;

        for(int i = 0; i < in_dims.size(); i++)
        {
            // i == W dim
            if(i == 2)
            {
                out_dims.push_back(in_dims[i] + 2 * padding[0]);
            }
            // else if(i == 3)
            // {
            //     out_dims.push_back(in_dims[i] + 2 * padding[0]);
            // }
            else
            {
                out_dims.push_back(in_dims[i]);
            }
        }
        auto output_strides = GetStrides(out_dims, contiguous == 1);
        output = tensor<T>{out_dims, output_strides};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dims, output_strides};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();
        auto padding  = pad_reflection_config.GetPadding();
        auto contiguous = pad_reflection_config.GetContiguous();

        cpu_pad_reflection_fwd<T>(input, ref_output, contiguous, padding);
        miopenStatus_t status;
        if (contiguous == 1)
        {
            status = miopen::PadReflection1dFwdContiguous(handle,
                                                input.desc,
                                                input_dev.get(),
                                                output.desc,
                                                output_dev.get(),
                                                padding.data(),
                                                padding.size());
        }
        else if (contiguous == 0)
        {
            status = miopen::PadReflection1dFwd(handle,
                                                input.desc,
                                                input_dev.get(),
                                                output.desc,
                                                output_dev.get(),
                                                padding.data(),
                                                padding.size());
        }
        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        for(int i = 0; i < output.data.size() - 1; ++i)
        {
            EXPECT_EQ(output.data[i], ref_output.data[i]);
        }
    }
    PadReflectionCase pad_reflection_config;

    tensor<T> input;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

template <typename T = float>
struct PadReflectionBwdTest : public ::testing::TestWithParam<PadReflectionCase>
{
protected:
    void SetUp() override
    {
        auto&& handle         = get_handle();
        pad_reflection_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto in_dims = pad_reflection_config.GetInput();
        auto padding = pad_reflection_config.GetPadding();
        auto contiguous = pad_reflection_config.GetContiguous();
        auto input_strides = GetStrides(in_dims, contiguous == 1);
        input        = tensor<T>{in_dims, input_strides};
        std::fill(input.begin(), input.end(), 0.5);

        std::vector<size_t> out_dims;
        for(int i = 0; i < in_dims.size(); i++)
        {
            if(i == 2)
            {
                out_dims.push_back(in_dims[i] + 2 * padding[0]);
            }
            else
            {
                out_dims.push_back(in_dims[i]);
            }
        }
        auto output_strides = GetStrides(out_dims, contiguous == 1);
        output = tensor<T>{out_dims, output_strides}.generate(gen_value);
        
        ref_input = tensor<T>{in_dims, input_strides};
        std::fill(ref_input.begin(), ref_input.end(), 0.5);

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();
        auto padding  = pad_reflection_config.GetPadding();
        auto contiguous = pad_reflection_config.GetContiguous();

        cpu_pad_reflection_bwd<T>(ref_input, output, contiguous, padding);
        miopenStatus_t status;
        if (contiguous == 1)
        {
            status = miopen::PadReflection1dBwdContiguous(handle,
                                                input.desc,
                                                input_dev.get(),
                                                output.desc,
                                                output_dev.get(),
                                                padding.data(),
                                                padding.size());
        }
        else if (contiguous == 0)
        {
            status = miopen::PadReflection1dBwd(handle,
                                                input.desc,
                                                input_dev.get(),
                                                output.desc,
                                                output_dev.get(),
                                                padding.data(),
                                                padding.size());
        }
        EXPECT_EQ(status, miopenStatusSuccess);
        input.data = handle.Read<T>(input_dev, input.data.size());
    }

    void Verify()
    {\
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;
        auto error = miopen::rms_range(input, ref_input);
        EXPECT_TRUE(miopen::range_distance(input) == miopen::range_distance(ref_input));
        EXPECT_TRUE(error < tolerance) << "Outputs do not match each other. Error:" << error;
    }
    PadReflectionCase pad_reflection_config;

    tensor<T> input;
    tensor<T> output;

    tensor<T> ref_input;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};
