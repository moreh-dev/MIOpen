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
#include "cpu_l1loss.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/l1loss.hpp>

struct L1LossTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    miopenL1LossReduction_t reduction;
    bool contiguous;

    friend std::ostream& operator<<(std::ostream& os, const L1LossTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " reducion mode:" << tc.reduction
                  << " contiguous:" << tc.contiguous;
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
};

inline std::vector<L1LossTestCase> L1LossTestConfigs()
{ // n c d h w dim
    // clang-format off
    return {
        {1, 1, 1, 1, 1, MIOPEN_L1LOSS_SUM_REDUCTION, false},
        {1, 2, 3, 4, 1, MIOPEN_L1LOSS_SUM_REDUCTION, false},
        {1, 1, 1, 257, 1, MIOPEN_L1LOSS_SUM_REDUCTION, false},
        {2, 10, 128, 128, 1, MIOPEN_L1LOSS_SUM_REDUCTION, false},
        {5, 13, 17, 11, 1, MIOPEN_L1LOSS_MEAN_REDUCTION, false},
        {256, 4, 8723, 1, 1, MIOPEN_L1LOSS_SUM_REDUCTION, false},
        {256, 4, 8723, 1, 1, MIOPEN_L1LOSS_SUM_REDUCTION, true},
        {1, 1, 1, 1, 1, MIOPEN_L1LOSS_SUM_REDUCTION, true},
        {34, 4, 5, 1, 1, MIOPEN_L1LOSS_SUM_REDUCTION, true},
        {4, 7, 5, 1, 1, MIOPEN_L1LOSS_SUM_REDUCTION, true},
        {15, 4, 5, 1, 1, MIOPEN_L1LOSS_SUM_REDUCTION, true}
    };
    // clang-format on
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
struct L1LossFwdTest : public ::testing::TestWithParam<L1LossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle   = get_handle();
        l1loss_config   = GetParam();
        auto gen_value1 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 1); };
        auto gen_value2 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 2); };

        reduction       = l1loss_config.reduction;
        auto in_dims    = l1loss_config.GetInput();
        auto contiguous = l1loss_config.contiguous;

        auto in_strides = GetStrides(in_dims, contiguous);
        input           = tensor<T>{in_dims, in_strides}.generate(gen_value1);

        auto tar_strides = GetStrides(in_dims, contiguous);
        target           = tensor<T>{in_dims, tar_strides}.generate(gen_value2);

        auto out_lengths =
            (reduction == MIOPEN_L1LOSS_NONE_REDUCTION) ? in_dims : std::vector<size_t>{1};
        auto out_strides = GetStrides(out_lengths, contiguous);

        output = tensor<T>{out_lengths, out_strides};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_lengths, out_strides};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        std::vector<size_t> workspace_lengths;
        ws_sizeInBytes = (reduction == MIOPEN_L1LOSS_NONE_REDUCTION)
                             ? 0
                             : miopen::GetL1LossForwardWorkspaceSize(
                                   handle, reduction, input.desc, target.desc, output.desc);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();

        if(ws_sizeInBytes != 0)
        {
            std::vector<size_t> workspace_dims;
            workspace_dims.push_back(ws_sizeInBytes / sizeof(T));

            workspace = tensor<T>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), static_cast<T>(0));

            ref_workspace = tensor<T>{workspace_dims};
            std::fill(ref_workspace.begin(), ref_workspace.end(), static_cast<T>(0));

            workspace_dev = handle.Write(workspace.data);
        }

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        if(reduction != MIOPEN_L1LOSS_NONE_REDUCTION)
        {
            cpu_l1loss_reduced_forward<T>(input, target, ref_output, ref_workspace, reduction);
            status         = miopen::L1LossForward(handle,
                                           reduction,
                                           workspace_dev.get(),
                                           ws_sizeInBytes,
                                           input.desc,
                                           input_dev.get(),
                                           target.desc,
                                           target_dev.get(),
                                           output.desc,
                                           output_dev.get());
            workspace.data = handle.Read<T>(workspace_dev, workspace.data.size());
        }

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    double GetTolerance()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;
        return tolerance;
    }

    void Verify()
    {
        double threshold = GetTolerance();

        // auto error_w = miopen::rms_range(ref_workspace, workspace);
        //
        // EXPECT_TRUE(miopen::range_distance(ref_workspace) == miopen::range_distance(workspace));
        // EXPECT_TRUE(error_w < tolerance) << "Error workspace beyond tolerance Error: " << error_w
        //                                 << ",  Tolerance: " << tolerance;

        auto error = miopen::rms_range(ref_output, output);
        std::cout << "ref output = " << ref_output[0] << " output = " << output[0] << std::endl;

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error: " << error
                                            << ",  Tolerance: " << threshold * 10;
    }

    L1LossTestCase l1loss_config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> output;
    tensor<T> workspace;
    miopenL1LossReduction_t reduction;

    tensor<T> ref_workspace;
    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;
};

template <typename T = float>
struct L1LossBwdTest : public ::testing::TestWithParam<L1LossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle        = get_handle();
        l1loss_config = GetParam();
        auto gen_value1 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_value2 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 101); };

        reduction         = l1loss_config.reduction;
        auto in_dims    = l1loss_config.GetInput();
        auto contiguous = l1loss_config.contiguous;

        //if(contiguous)
        //    GTEST_SKIP();

        auto in_strides = GetStrides(in_dims, contiguous);
        input           = tensor<T>{in_dims, in_strides}.generate(gen_value1);

        auto tar_strides = GetStrides(in_dims, contiguous);
        target           = tensor<T>{in_dims, tar_strides}.generate(gen_value2);

        auto out_lengths = (reduction == MIOPEN_L1LOSS_NONE_REDUCTION) ? in_dims : std::vector<size_t>{1};
        auto out_strides = GetStrides(out_lengths, contiguous);

        dO = tensor<T>{out_lengths, out_strides};
        std::fill(dO.begin(), dO.end(), 0.5);

        dI = tensor<T>{in_dims, in_strides};
        std::fill(dI.begin(), dI.end(), std::numeric_limits<T>::quiet_NaN());
        dT = tensor<T>{in_dims, tar_strides};
        std::fill(dT.begin(), dT.end(), std::numeric_limits<T>::quiet_NaN());

        ref_dI = tensor<T>{in_dims, in_strides};
        std::fill(ref_dI.begin(), ref_dI.end(), std::numeric_limits<T>::quiet_NaN());
        ref_dT = tensor<T>{in_dims, tar_strides};
        std::fill(ref_dT.begin(), ref_dT.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        dO_dev     = handle.Write(dO.data);
        dI_dev     = handle.Write(dI.data);
        dT_dev     = handle.Write(dT.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        if(reduction != MIOPEN_L1LOSS_NONE_REDUCTION)
        {
            cpu_l1loss_reduced_backward<T>(input, target, dO, ref_dI, ref_dT, reduction);
            status = miopen::L1LossBackward(handle,
                                                         input.desc,
                                                         input_dev.get(),
                                                         target.desc,
                                                         target_dev.get(),
                                                         dO.desc,
                                                         dO_dev.get(),
                                                         dI.desc,
                                                         dI_dev.get(),
                                                         dT.desc,
                                                         dT_dev.get(),
                                                         reduction);
        }

        EXPECT_EQ(status, miopenStatusSuccess);

        dI.data = handle.Read<T>(dI_dev, dI.data.size());
        dT.data = handle.Read<T>(dT_dev, dT.data.size());
    }

    double GetTolerance()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;
        return tolerance;
    }

    void Verify()
    {
        double threshold = GetTolerance();

        auto error_dI = miopen::rms_range(ref_dI, dI);
        auto error_dT = miopen::rms_range(ref_dT, dT);

        EXPECT_TRUE(miopen::range_distance(ref_dI) == miopen::range_distance(dI));
        EXPECT_TRUE(miopen::range_distance(ref_dT) == miopen::range_distance(dT));
        EXPECT_TRUE(error_dI < threshold * 10)
            << "Error output beyond tolerance Error: " << error_dI
            << ",  Tolerance: " << threshold * 10;
        EXPECT_TRUE(error_dT < threshold * 10)
            << "Error output beyond tolerance Error: " << error_dT
            << ",  Tolerance: " << threshold * 10;
    }

    L1LossTestCase l1loss_config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> dO;
    tensor<T> dI;
    tensor<T> dT;

    tensor<T> ref_dI;
    tensor<T> ref_dT;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr dO_dev;
    miopen::Allocator::ManageDataPtr dI_dev;
    miopen::Allocator::ManageDataPtr dT_dev;

    miopenL1LossReduction_t reduction;
};
