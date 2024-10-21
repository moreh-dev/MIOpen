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

#include "cpu_gather.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>
#include <miopen/allocator.hpp>
#include <miopen/miopen.h>
#include <miopen/gather.hpp>

struct GatherV2TestCase
{
    std::vector<size_t> paramGradDims;
    std::vector<size_t> indicesDims;
    uint32_t dim;
    uint32_t batch_dims;

    friend std::ostream& operator<<(std::ostream& os, const GatherV2TestCase& tc)
    {
        os << "Param dim: ";
        for(auto paramDim : tc.paramGradDims)
        {
            os << paramDim << " ";
        }
        os << std::endl;

        os << "Indices dim: ";
        for(auto indexDim : tc.indicesDims)
        {
            os << indexDim << " ";
        }
        os << std::endl;

        os << "dim: " << tc.dim << " batch_dims: " << tc.batch_dims;

        return os;
    }

    std::vector<size_t> GetParamGradDim() const { return paramGradDims; }
    std::vector<size_t> GetIndicesDim() const { return indicesDims; }
    uint32_t GetDim() const { return dim; }
    uint32_t GetBatchDims() const { return batch_dims; }

    GatherV2TestCase() {}

    GatherV2TestCase(std::vector<size_t> param_grad_dim,
                     std::vector<size_t> indices_dim,
                     uint32_t dim_,
                     uint32_t batch_dims_)
        : paramGradDims(param_grad_dim),
          indicesDims(indices_dim),
          dim(dim_),
          batch_dims(batch_dims_)
    {
    }
};

inline std::vector<GatherV2TestCase> GenFullTestCases()
{
    return {GatherV2TestCase({2, 2, 3}, {2, 2}, 1, 0),         // non-batched
            GatherV2TestCase({16, 16, 3}, {24, 2}, 2, 0),      // non-batched large size
            GatherV2TestCase({2, 2, 3}, {2, 2}, 1, 1),         // batched
            GatherV2TestCase({16, 256, 256}, {16, 256}, 1, 1), // batched large size
            GatherV2TestCase({16, 256, 768}, {16, 256}, 2, 2),
            GatherV2TestCase({32, 400, 400}, {32, 8}, 2, 1)};
}

template <typename T, typename I>
struct GatherV2BwdTest : public ::testing::TestWithParam<GatherV2TestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto param_grad_dims = config.GetParamGradDim();
        auto indices_dims    = config.GetIndicesDim();
        dim                  = config.GetDim();
        batch_dims           = config.GetBatchDims();

        std::vector<size_t> out_grad_dims;
        for(uint32_t i = 0; i < dim; i++)
        {
            out_grad_dims.push_back(param_grad_dims[i]);
        }

        for(uint32_t i = batch_dims; i < indices_dims.size(); i++)
        {
            out_grad_dims.push_back(indices_dims[i]);
        }

        for(uint32_t i = dim + 1; i < param_grad_dims.size(); i++)
        {
            out_grad_dims.push_back(param_grad_dims[i]);
        }

        gatherDesc.setMode(MIOPEN_GATHER_V2);
        gatherDesc.setDim(dim);
        gatherDesc.setBatchDims(batch_dims);

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gather_dim_size = param_grad_dims[dim];
        auto gen_index       = [gather_dim_size](auto...) {
            return prng::gen_0_to_B(static_cast<I>(gather_dim_size));
        };

        indices    = tensor<I>{indices_dims}.generate(gen_index);
        outputGrad = tensor<T>{out_grad_dims}.generate(gen_value);

        paramGrad = tensor<T>{param_grad_dims};
        std::fill(paramGrad.begin(), paramGrad.end(), static_cast<T>(0));

        ref_paramGrad = tensor<T>{param_grad_dims};
        std::fill(ref_paramGrad.begin(), ref_paramGrad.end(), static_cast<T>(0));

        paramGrad_dev  = handle.Write(paramGrad.data);
        indices_dev    = handle.Write(indices.data);
        outputGrad_dev = handle.Write(outputGrad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_gatherv2_backward<T, I>(outputGrad, indices, ref_paramGrad, dim, batch_dims);

        miopenStatus_t status;
        status = gatherDesc.Backward(handle,
                                     outputGrad.desc,
                                     outputGrad_dev.get(),
                                     indices.desc,
                                     indices_dev.get(),
                                     paramGrad.desc,
                                     paramGrad_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        paramGrad.data = handle.Read<T>(paramGrad_dev, paramGrad.data.size());
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
        return tolerance;
    }

    void Verify()
    {
        double threshold = GetTolerance();
        auto error       = miopen::rms_range(ref_paramGrad, paramGrad);

        EXPECT_TRUE(miopen::range_distance(ref_paramGrad) == miopen::range_distance(paramGrad));

        EXPECT_TRUE(error < threshold)
            << "Error output (param grad) beyond tolerance Error:" << error
            << ",  Threshold: " << threshold << std::endl;
    }

    GatherV2TestCase config;

    miopen::GatherDescriptor gatherDesc;
    tensor<T> paramGrad;
    tensor<T> outputGrad;
    tensor<I> indices;

    tensor<T> ref_paramGrad;

    uint32_t dim;
    uint32_t batch_dims;

    miopen::Allocator::ManageDataPtr paramGrad_dev;
    miopen::Allocator::ManageDataPtr outputGrad_dev;
    miopen::Allocator::ManageDataPtr indices_dev;
};
