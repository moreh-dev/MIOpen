/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include "cpu_indexselect.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/indexselect.hpp>

struct IndexSelectTestCase
{
    std::vector<int> inputs;
    int dim;
    std::vector<int> indices;

    friend std::ostream& operator<<(std::ostream& os, const IndexSelectTestCase& tc)
    {
        os << "inputs:";
        for(size_t i = 0; i < tc.inputs.size(); i++)
        {
            os << tc.inputs[i];
            if(i != tc.inputs.size() - 1)
            {
                os << ",";
            }
        }
        os << "dim:";
        os << tc.dim;
        os << "indices:";
        for(size_t i = 0; i < tc.indices.size(); i++)
        {
            os << tc.indices[i];
            if(i != tc.indices.size() - 1)
            {
                os << ",";
            }
        }
        return os;
    }

    std::vector<int> GetInput() { return inputs; }
    int GetDim() { return dim; }
    std::vector<int> GetIndices() { return indices; };
    std::vector<int> GetOutput()
    {
        auto ret = inputs;
        ret[dim] = indices.size();
        return ret;
    }
};

std::vector<IndexSelectTestCase> IndexSelectFwdTestConfigs()
{
    return {{{32, 32, 32, 1}, 0, {2, 6, 8, 17, 19, 22, 26, 30}},
            {{32, 32, 32, 1}, 1, {0, 5, 6, 8, 9, 11, 13, 20}},
            {{32, 32, 32, 1}, 2, {3, 9, 13, 14, 23, 26, 27, 28}},
            {{32, 32, 512, 1}, 0, {0, 6, 8, 13, 21, 22, 25, 27}},
            {{32, 32, 512, 1}, 1, {6, 14, 21, 23, 24, 27, 30, 31}},
            {{32, 32, 512, 1}, 2, {19, 93, 171, 179, 200, 239, 442, 499}},
            {{32, 512, 32, 1}, 0, {3, 12, 16, 17, 19, 20, 24, 28}},
            {{32, 512, 32, 1}, 1, {60, 84, 97, 192, 258, 425, 474, 485}},
            {{32, 512, 32, 1}, 2, {7, 11, 13, 14, 16, 17, 22, 27}},
            {{32, 512, 512, 1}, 0, {5, 9, 11, 15, 18, 22, 24, 30}},
            {{32, 512, 512, 1}, 1, {8, 68, 166, 207, 297, 393, 428, 482}},
            {{32, 512, 512, 1}, 2, {32, 47, 54, 134, 345, 353, 408, 452}},
            {{512, 32, 32, 1}, 0, {75, 199, 220, 234, 241, 302, 348, 353}},
            {{512, 32, 32, 1}, 1, {1, 7, 10, 12, 14, 16, 24, 25}},
            {{512, 32, 32, 1}, 2, {1, 10, 11, 13, 14, 18, 26, 30}},
            {{512, 32, 512, 1}, 0, {89, 188, 225, 278, 299, 354, 385, 435}},
            {{512, 32, 512, 1}, 1, {4, 5, 8, 17, 19, 23, 25, 29}},
            {{512, 32, 512, 1}, 2, {7, 29, 57, 178, 216, 347, 382, 390}},
            {{512, 512, 32, 1}, 0, {39, 62, 66, 156, 273, 277, 324, 465}},
            {{512, 512, 32, 1}, 1, {175, 244, 307, 341, 357, 466, 491, 497}},
            {{512, 512, 32, 1}, 2, {1, 11, 17, 22, 25, 27, 30, 31}},
            {{512, 512, 512, 1}, 0, {104, 249, 342, 343, 359, 381, 503, 510}},
            {{512, 512, 512, 1}, 1, {21, 63, 142, 205, 274, 359, 370, 433}},
            {{512, 512, 512, 1}, 2, {0, 100, 139, 234, 353, 368, 464, 489}}};
}

std::vector<IndexSelectTestCase> IndexSelectBwdTestConfigs()
{
    return {{{32, 32, 32, 1}, 0, {2, 6, 8, 17, 19, 22, 26, 30}},
            {{32, 32, 32, 1}, 1, {0, 5, 6, 8, 9, 11, 13, 20}},
            {{32, 32, 32, 1}, 2, {3, 9, 13, 14, 23, 26, 27, 28}},
            {{32, 32, 512, 1}, 0, {0, 6, 8, 13, 21, 22, 25, 27}},
            {{32, 32, 512, 1}, 1, {6, 14, 21, 23, 24, 27, 30, 31}},
            {{32, 32, 512, 1}, 2, {19, 93, 171, 179, 200, 239, 442, 499}},
            {{32, 512, 32, 1}, 0, {3, 12, 16, 17, 19, 20, 24, 28}},
            {{32, 512, 32, 1}, 1, {60, 84, 97, 192, 258, 425, 474, 485}},
            {{32, 512, 32, 1}, 2, {7, 11, 13, 14, 16, 17, 22, 27}},
            {{32, 512, 512, 1}, 0, {5, 9, 11, 15, 18, 22, 24, 30}},
            {{32, 512, 512, 1}, 1, {8, 68, 166, 207, 297, 393, 428, 482}},
            {{32, 512, 512, 1}, 2, {32, 47, 54, 134, 345, 353, 408, 452}},
            {{512, 32, 32, 1}, 0, {75, 199, 220, 234, 241, 302, 348, 353}},
            {{512, 32, 32, 1}, 1, {1, 7, 10, 12, 14, 16, 24, 25}},
            {{512, 32, 32, 1}, 2, {1, 10, 11, 13, 14, 18, 26, 30}},
            {{512, 32, 512, 1}, 0, {89, 188, 225, 278, 299, 354, 385, 435}},
            {{512, 32, 512, 1}, 1, {4, 5, 8, 17, 19, 23, 25, 29}},
            {{512, 32, 512, 1}, 2, {7, 29, 57, 178, 216, 347, 382, 390}},
            {{512, 512, 32, 1}, 0, {39, 62, 66, 156, 273, 277, 324, 465}},
            {{512, 512, 32, 1}, 1, {175, 244, 307, 341, 357, 466, 491, 497}},
            {{512, 512, 32, 1}, 2, {1, 11, 17, 22, 25, 27, 30, 31}},
            {{512, 512, 512, 1}, 0, {104, 249, 342, 343, 359, 381, 503, 510}},
            {{512, 512, 512, 1}, 1, {21, 63, 142, 205, 274, 359, 370, 433}},
            {{512, 512, 512, 1}, 2, {0, 100, 139, 234, 353, 368, 464, 489}}};
}

template <typename T = float>
struct IndexSelectFwdTest : public ::testing::TestWithParam<IndexSelectTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle      = get_handle();
        indexselect_config = GetParam();
        auto gen_value     = [](auto...) { return prng::gen_descreet_uniform_sign<T>(0, 1); };

        auto in_dims      = indexselect_config.GetInput();
        dim               = indexselect_config.GetDim();
        auto indices_para = indexselect_config.GetIndices();
        auto out_dims     = indexselect_config.GetOutput();

        input      = tensor<T>{in_dims}.generate(gen_value);
        output     = tensor<T>{out_dims};
        indices    = tensor<int>{std::vector<size_t>({indices_para.size()})};
        outputhost = tensor<T>{out_dims};

        for(size_t i = 0; i < indices_para.size(); i++)
        {
            indices[i] = indices_para[i];
        }

        std::fill(output.begin(), output.end(), 0);
        std::fill(outputhost.begin(), outputhost.end(), 0);

        input_dev   = handle.Write(input.data);
        output_dev  = handle.Write(output.data);
        indices_dev = handle.Write(indices.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_indexselect_forward<T>(input, indices, output, dim, outputhost);

        miopenStatus_t status;

        status = miopen::IndexSelectForward(handle,
                                            input.desc,
                                            input_dev.get(),
                                            indices.desc,
                                            indices_dev.get(),
                                            output.desc,
                                            output_dev.get(),
                                            dim);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(outputhost, output);

        EXPECT_TRUE(miopen::range_distance(outputhost) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error:" << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    IndexSelectTestCase indexselect_config;

    tensor<T> input;
    tensor<int> indices;
    tensor<T> output;
    int dim;

    tensor<T> outputhost;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr indices_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

template <typename T = float>
struct IndexSelectBwdTest : public ::testing::TestWithParam<IndexSelectTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle      = get_handle();
        indexselect_config = GetParam();
        auto gen_value     = [](auto...) { return prng::gen_descreet_uniform_sign<T>(0, 1); };

        auto in_dims      = indexselect_config.GetInput();
        dim               = indexselect_config.GetDim();
        auto indices_para = indexselect_config.GetIndices();
        auto out_dims     = indexselect_config.GetOutput();

        inputGrad     = tensor<T>{in_dims};
        outputGrad    = tensor<T>{out_dims}.generate(gen_value);
        indices       = tensor<int>{std::vector<size_t>({indices_para.size()})};
        inputGradhost = tensor<T>{in_dims};

        for(size_t i = 0; i < indices_para.size(); i++)
        {
            indices[i] = indices_para[i];
        }

        std::fill(inputGrad.begin(), inputGrad.end(), 0);
        std::fill(inputGradhost.begin(), inputGradhost.end(), 0);

        inputGrad_dev  = handle.Write(inputGrad.data);
        outputGrad_dev = handle.Write(outputGrad.data);
        indices_dev    = handle.Write(indices.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_indexselect_backward<T>(inputGradhost, indices, dim, outputGrad);

        miopenStatus_t status;

        status = miopen::IndexSelectBackward(handle,
                                             inputGrad.desc,
                                             inputGrad_dev.get(),
                                             indices.desc,
                                             indices_dev.get(),
                                             outputGrad.desc,
                                             outputGrad_dev.get(),
                                             dim);

        EXPECT_EQ(status, miopenStatusSuccess);

        inputGrad.data = handle.Read<T>(inputGrad_dev, inputGrad.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(inputGradhost, inputGrad);

        EXPECT_TRUE(miopen::range_distance(inputGradhost) == miopen::range_distance(inputGrad));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error:" << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    IndexSelectTestCase indexselect_config;

    tensor<T> inputGrad;
    tensor<int> indices;
    tensor<T> outputGrad;
    int dim;

    tensor<T> inputGradhost;

    miopen::Allocator::ManageDataPtr inputGrad_dev;
    miopen::Allocator::ManageDataPtr indices_dev;
    miopen::Allocator::ManageDataPtr outputGrad_dev;
};