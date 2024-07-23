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
        ret[dim] -= indices.size();
        return ret;
    }
};

std::vector<IndexSelectTestCase> IndexSelectFwdTestConfigs() { return {{{2, 2, 2, 2}, 1, {1}}}; }

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

        input.data   = handle.Read<T>(input_dev, input.data.size());
        output.data  = handle.Read<T>(output_dev, output.data.size());
        indices.data = handle.Read<int>(indices_dev, indices.data.size());
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