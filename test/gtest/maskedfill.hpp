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

#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <miopen/maskedfill.hpp>
#include <miopen/maskedfill/solvers.hpp>

#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "random.hpp"
#include "verify.hpp"
#include "../driver/tensor_driver.hpp"

#include "cpu_maskedfill.hpp"

struct MaskedFillTestCase /* MaskedFillTestParameters */
{
    std::vector<size_t> const size; // or "dims"
    std::vector<size_t> const& GetSize() const { return size; };
    std::vector<size_t> strides;
    std::vector<size_t> const& GetStrides() const { return strides; };
    MaskedFillTestCase(std::vector<size_t> const size_) : size{size_}, strides(size_.size(), 1)
    {
        auto stride = 1;
        for(signed i = size.size() - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= size[i];
        }
    }
    MaskedFillTestCase(std::vector<size_t> const size_, std::vector<size_t> const strides_)
        : size{size_}, strides{strides_}
    {
    }
    friend std::ostream& operator<<(std::ostream& os, MaskedFillTestCase const& parameters)
    {
        assert(parameters.size.size() == parameters.strides.size());
        std::stringstream sizestringstream, stridesstringstream;
        sizestringstream << "{";
        stridesstringstream << "{";
        for(auto dimension = 0; dimension < parameters.size.size(); ++dimension)
        {
            sizestringstream << parameters.size[dimension];
            stridesstringstream << parameters.strides[dimension];
            if(dimension < parameters.size.size() - 1)
            {
                sizestringstream << ", ";
                stridesstringstream << ", ";
            }
        }
        sizestringstream << "}";
        stridesstringstream << "}";
        return os << "{" << sizestringstream.str() << ", " << stridesstringstream.str() << "}";
    }
};
std::vector<MaskedFillTestCase> const
MaskedFillTestConfigs(bool const is_backward)
{
    if (!is_backward) return {
        {{1}},
        {{2, 2}},
        {{1323, 12, 12, 20}},
        {{2, 2, 2}},
        {{2, 2, 2}, {1, 4, 2}},
    };
    else return {
        {{1}},
        {{2, 2}},
        {{1323, 12, 12, 20}},
        {{2, 2, 2}},
        {{2, 2, 2}, {1, 4, 2}},
    };
}

inline int SetTensorLayout(miopen::TensorDescriptor& desc)
{
    return SetTensorNd(&desc, desc.GetLengths(), desc.GetStrides(), desc.GetType());
}

template <typename T = float>
class MaskedFillTest : public testing::TestWithParam<MaskedFillTestCase>
{
    bool const is_backward;
    tensor<T> input, output, ref_output;
    tensor<int8_t>
        mask; // `tensor<bool>`s aren't implemented (because `miopen_type<bool>` isn't implemented)
    miopen::Allocator::ManageDataPtr input_dev, output_dev, mask_dev;
    float value;

    void SetUp() override
    {
        auto&& handle      = get_handle();
        auto const size    = GetParam().GetSize();
        auto const strides = GetParam().GetStrides();

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        std::mt19937 generator;
        std::uniform_int_distribution<unsigned int> distribution{0, 1};

        input = tensor<T>(size, strides);
        input.generate(gen_value);
        SetTensorLayout(input.desc);
        input_dev = handle.Write(input.data);

        output = tensor<T>(size, strides);
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());
        SetTensorLayout(output.desc);
        output_dev = handle.Write(output.data);

        ref_output = tensor<T>(size, strides);
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());
        SetTensorLayout(ref_output.desc);

        mask = tensor<int8_t>(size, strides);
        mask.generate([&](auto...) { return distribution(generator); });
        SetTensorLayout(mask.desc);
        mask_dev = handle.Write(mask.data);

        value = prng::gen_descreet_uniform_sign<float>(1e-2, 100);
    }

public:
    MaskedFillTest(bool const is_backward_) : is_backward {is_backward_} {}

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;
        if (!is_backward) {
            status = miopen::MaskedFillForward(handle,

                                               input.desc,
                                               input_dev.get(),
                                               output.desc,
                                               output_dev.get(),

                                               mask.desc,
                                               mask_dev.get(),

                                               value);
            EXPECT_EQ(status, miopenStatusSuccess);
            output.data = handle.Read<T>(output_dev, output.data.size());
            cpu_maskedfill_forward<T, 5>(input, ref_output, mask, value);
        } else {
            status = miopen::MaskedFillBackward(handle,

                                                input.desc,
                                                input_dev.get(),
                                                output.desc,
                                                output_dev.get(),

                                                mask.desc,
                                                mask_dev.get(),

                                                value);
            EXPECT_EQ(status, miopenStatusSuccess);
            output.data = handle.Read<T>(output_dev, output.data.size());
            cpu_maskedfill_backward<T, 5>(input, ref_output, mask);
        }
    }
    void Verify() const
    {
        EXPECT_TRUE(miopen::range_distance(output) == miopen::range_distance(ref_output));
        auto const error =
            miopen::range_product(output, ref_output, 0, miopen::sum, miopen::abs_diff);
        EXPECT_TRUE(error == 0) << "Outputs do not match each other: `error` = " << error;
    }
};

template <typename T = float>
struct MaskedFillForwardTest : MaskedFillTest<T>
{
    MaskedFillForwardTest() : MaskedFillTest<T>{false} {}
};

template <typename T = float>
struct MaskedFillBackwardTest : MaskedFillTest<T>
{
    MaskedFillBackwardTest() : MaskedFillTest<T>{true} {}
};
