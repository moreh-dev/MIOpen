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
#include "cpu_logsumexp.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/logsumexp.hpp>

struct LogsumexpTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    int32_t* dims;
    int32_t num_dims;
    bool keepdim;

    friend std::ostream& operator<<(std::ostream& os, const LogsumexpTestCase& tc)
    {
        return os << "N: " << tc.N << " C: " << tc.C << " D: " << tc.D << " H: " << tc.H
                  << " W: " << tc.W << " num_dims: " << tc.num_dims << " keepdim: " << tc.keepdim;
    }

    std::vector<size_t> GetInput()
    {
        if((N != 0) && (C != 0) && (D != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>{N, C, D, H, W};
        }
        else if((N != 0) && (C != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>{N, C, H, W};
        }
        else if((N != 0) && (C != 0) && (W != 0))
        {
            return std::vector<size_t>{N, C, W};
        }
        else if((N != 0) && (C != 0))
        {
            return std::vector<size_t>{N, C};
        }
        else if((N != 0))
        {
            return std::vector<size_t>{N};
        }
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }

    std::vector<int32_t> GetDims()
    {
        if((N != 0) && (C != 0) && (D != 0) && (H != 0) && (W != 0))
        {
            std::vector<int32_t> adjusted_dims;
            for(int i = 0; i < num_dims; i++)
            {
                adjusted_dims.push_back(dims[i]);
            }
            return adjusted_dims;
        }
        else if((N != 0) && (C != 0) && (H != 0) && (W != 0))
        {
            std::vector<int32_t> adjusted_dims;
            for(int i = 0; i < num_dims; i++)
            {
                int32_t dim = dims[i];
                if(dim == 3 || dim == 4)
                {
                    adjusted_dims.push_back(dim - 1);
                }
                else if(dim == 2)
                {
                    std::cout << "Incorrect Dims\n" << std::endl;
                    return std::vector<int32_t>({0});
                }
                else
                {
                    adjusted_dims.push_back(dim);
                }
            }
            return adjusted_dims;
        }
        else if((N != 0) && (C != 0) && (W != 0))
        {
            std::vector<int32_t> adjusted_dims;
            for(int i = 0; i < num_dims; i++)
            {
                int32_t dim = dims[i];
                if(dim == 4)
                {
                    adjusted_dims.push_back(dim - 2);
                }
                else if(dim == 3 || dim == 2)
                {
                    std::cout << "Incorrect Dims\n" << std::endl;
                    return std::vector<int32_t>({0});
                }
                else
                {
                    adjusted_dims.push_back(dim);
                }
            }
            return adjusted_dims;
        }
        else if((N != 0) && (C != 0))
        {
            std::vector<int32_t> adjusted_dims;
            for(int i = 0; i < num_dims; i++)
            {
                int32_t dim = dims[i];
                if(dim == 2 || dim == 3 || dim == 4)
                {
                    std::cout << "Incorrect Dims\n" << std::endl;
                    return std::vector<int32_t>({0});
                }
                else
                {
                    adjusted_dims.push_back(dim);
                }
            }
            return adjusted_dims;
        }
        else if((N != 0))
        {
            std::vector<int32_t> adjusted_dims;
            for(int i = 0; i < num_dims; i++)
            {
                int32_t dim = dims[i];
                if(dim == 1 || dim == 2 || dim == 3 || dim == 4)
                {
                    std::cout << "Incorrect Dims\n" << std::endl;
                    return std::vector<int32_t>({0});
                }
                else
                {
                    adjusted_dims.push_back(dim);
                }
            }
            return adjusted_dims;
        }
        else
        {
            std::cout << "Incorrect Dims\n" << std::endl;
            return std::vector<int32_t>({0});
        }
    }
};

std::vector<LogsumexpTestCase> LogsumexpTestConfigs()
{
    return {
        {17, 40, 0, 0, 0, new int32_t[1]{0}, 1, true},
    };
}

template <typename T = float>
struct LogsumexpForwardTest : public ::testing::TestWithParam<LogsumexpTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle    = get_handle();
        logsumexp_config = GetParam();
        auto gen_value   = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        std::vector<int32_t> dims_vector = logsumexp_config.GetDims();

        dims     = logsumexp_config.dims;
        num_dims = logsumexp_config.num_dims;

        for(int i = 0; i < num_dims; i++)
        {
            dims[i] = dims_vector[i];
        }

        keepdim = logsumexp_config.keepdim;

        auto input_dims = logsumexp_config.GetInput();

        std::vector<size_t> output_dims(input_dims.size());
        std::copy(input_dims.begin(), input_dims.end(), output_dims.begin());

        for(const auto& dim : dims_vector)
        {
            output_dims[dim] = 1;
        }

        input = tensor<T>{input_dims}.generate(gen_value);

        output = tensor<T>{output_dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{output_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_logsumexp_forward(input, ref_output);
        miopenStatus_t status;

        status = miopen::LogsumexpForward(handle,
                                          input.desc,
                                          input_dev.get(),
                                          output.desc,
                                          output_dev.get(),
                                          dims,
                                          num_dims,
                                          keepdim);

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
            << "Error output beyond tolerance Error: " << error << ",   Threshold " << threshold;
    }

    LogsumexpTestCase logsumexp_config;

    tensor<T> input;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    int32_t* dims;
    int32_t num_dims;

    bool keepdim;
};
