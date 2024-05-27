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

#include "cpu_tripletmarginloss.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/tripletmarginloss.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
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

struct TripletMarginLossTestCase
{
    std::vector<size_t> lengths;
    float margin;
    int p;
    float eps;
    bool swap;
    float divisor;
    bool contiguous;

    friend std::ostream& operator<<(std::ostream& os, const TripletMarginLossTestCase& tc)
    {
        return os << " Lengths:" << tc.lengths << " Margin:" << tc.margin << " p:" << tc.p
                  << " eps:" << tc.eps << " Swap:" << (tc.swap ? "True" : "False")
                  << " Divisor:" << tc.divisor
                  << " Contiguous:" << (tc.contiguous ? "True" : "False");
    }
};

inline std::vector<TripletMarginLossTestCase> TripletMarginLossTestConfigs()
{
    std::vector<TripletMarginLossTestCase> tcs;
    tcs.push_back({{65536, 1}, 1, 2, 1e-6, false, std::numeric_limits<float>::quiet_NaN(), true});
    tcs.push_back({{1, 2}, 1, 2, 1e-6, false, std::numeric_limits<float>::quiet_NaN(), true});
    tcs.push_back({{1, 1024}, 1, 2, 1e-6, false, std::numeric_limits<float>::quiet_NaN(), true});
    tcs.push_back({{1, 65536}, 1, 2, 1e-6, false, std::numeric_limits<float>::quiet_NaN(), true});
    tcs.push_back({{993, 80}, 1, 2, 1e-6, false, std::numeric_limits<float>::quiet_NaN(), true});
    tcs.push_back({{80, 993}, 1, 2, 1e-6, false, std::numeric_limits<float>::quiet_NaN(), true});

    tcs.push_back({{65536, 1}, 1, 2, 1e-6, false, std::numeric_limits<float>::quiet_NaN(), false});
    tcs.push_back({{1, 2}, 1, 2, 1e-6, false, std::numeric_limits<float>::quiet_NaN(), false});
    tcs.push_back({{1, 1024}, 1, 2, 1e-6, false, std::numeric_limits<float>::quiet_NaN(), false});
    tcs.push_back({{1, 65536}, 1, 2, 1e-6, false, std::numeric_limits<float>::quiet_NaN(), false});
    tcs.push_back({{993, 80}, 1, 2, 1e-6, false, std::numeric_limits<float>::quiet_NaN(), false});
    tcs.push_back({{80, 993}, 1, 2, 1e-6, false, std::numeric_limits<float>::quiet_NaN(), false});

    tcs.push_back({{65536, 1}, 1, 2, 1e-6, false, 1, true});
    tcs.push_back({{1, 2}, 1, 2, 1e-6, false, 1, true});
    tcs.push_back({{1, 1024}, 1, 2, 1e-6, false, 1, true});
    tcs.push_back({{1, 65536}, 1, 2, 1e-6, false, 1, true});
    tcs.push_back({{993, 80}, 1, 2, 1e-6, false, 1, true});
    tcs.push_back({{80, 993}, 1, 2, 1e-6, false, 1, true});

    tcs.push_back({{65536, 1}, 1, 2, 1e-6, false, 1, false});
    tcs.push_back({{1, 2}, 1, 2, 1e-6, false, 1, false});
    tcs.push_back({{1, 1024}, 1, 2, 1e-6, false, 1, false});
    tcs.push_back({{1, 65536}, 1, 2, 1e-6, false, 1, false});
    tcs.push_back({{993, 80}, 1, 2, 1e-6, false, 1, false});
    tcs.push_back({{80, 993}, 1, 2, 1e-6, false, 1, false});

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
struct TripletMarginLossTest : public ::testing::TestWithParam<TripletMarginLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle            = get_handle();
        tripletmarginloss_config = GetParam();
        auto gen_value1 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 10); };
        auto gen_value2 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 9); };
        auto gen_value3 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 11); };

        margin          = tripletmarginloss_config.margin;
        p               = tripletmarginloss_config.p;
        eps             = tripletmarginloss_config.eps;
        swap            = tripletmarginloss_config.swap;
        divisor         = tripletmarginloss_config.divisor;
        auto lengths    = tripletmarginloss_config.lengths;
        auto contiguous = tripletmarginloss_config.contiguous;

        auto anchor_strides = GetStrides(lengths, contiguous);
        anchor              = tensor<T>{lengths, anchor_strides}.generate(gen_value1);
        dA                  = tensor<T>{lengths, anchor_strides};
        ref_dA              = tensor<T>{lengths, anchor_strides};

        auto positive_strides = GetStrides(lengths, true);
        positive              = tensor<T>{lengths, positive_strides}.generate(gen_value2);
        dP                    = tensor<T>{lengths, positive_strides};
        ref_dP                = tensor<T>{lengths, positive_strides};

        auto negative_strides = GetStrides(lengths, true);
        negative              = tensor<T>{lengths, negative_strides}.generate(gen_value3);
        dN                    = tensor<T>{lengths, negative_strides};
        ref_dN                = tensor<T>{lengths, negative_strides};

        auto out_lengths =
            std::isnan(divisor) ? std::vector<size_t>{lengths[0]} : std::vector<size_t>{1};
        auto out_strides = GetStrides(out_lengths, true);

        output     = tensor<T>{out_lengths, out_strides};
        ref_output = tensor<T>{out_lengths, out_strides};
        dO         = tensor<T>{out_lengths, out_strides};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(dO.begin(), dO.end(), static_cast<T>(0.5f));

        std::vector<size_t> workspace_lengths;
        ws_sizeInBytes = max(
            miopen::GetTripletMarginLossForwardWorkspaceSize(handle, anchor.desc, output.desc),
            miopen::GetTripletMarginLossBackwardWorkspaceSize(handle, anchor.desc, output.desc));
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();

        if(ws_sizeInBytes != 0)
        {
            std::vector<size_t> workspace_dims;
            workspace_dims.push_back(ws_sizeInBytes / sizeof(T));

            workspace = tensor<T>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), 0.0f);

            workspace_dev = handle.Write(workspace.data);
        }

        anchor_dev   = handle.Write(anchor.data);
        positive_dev = handle.Write(positive.data);
        negative_dev = handle.Write(negative.data);
        output_dev   = handle.Write(output.data);
        dO_dev       = handle.Write(dO.data);
        dA_dev       = handle.Write(dA.data);
        dP_dev       = handle.Write(dP.data);
        dN_dev       = handle.Write(dN.data);
    }

    void RunTest()
    {
        if(std::isnan(divisor))
        {
            cpu_tripletmarginloss_unreduced_forward<T>(
                anchor, positive, negative, ref_output, margin, p, eps, swap);
            cpu_tripletmarginloss_unreduced_backward<T>(
                anchor, positive, negative, dO, ref_dA, ref_dP, ref_dN, margin, p, eps, swap);
        }
        else
        {
            cpu_tripletmarginloss_forward<T>(
                anchor, positive, negative, ref_output, margin, p, eps, swap, divisor);
            cpu_tripletmarginloss_backward<T>(anchor,
                                              positive,
                                              negative,
                                              dO,
                                              ref_dA,
                                              ref_dP,
                                              ref_dN,
                                              margin,
                                              p,
                                              eps,
                                              swap,
                                              divisor);
        }

        auto&& handle = get_handle();
        miopenStatus_t status;

        status = miopen::TripletMarginLossForward(handle,
                                                  workspace_dev.get(),
                                                  ws_sizeInBytes,
                                                  anchor.desc,
                                                  anchor_dev.get(),
                                                  positive.desc,
                                                  positive_dev.get(),
                                                  negative.desc,
                                                  negative_dev.get(),
                                                  output.desc,
                                                  output_dev.get(),
                                                  margin,
                                                  p,
                                                  eps,
                                                  swap,
                                                  divisor);
        EXPECT_EQ(status, miopenStatusSuccess);
        output.data = handle.Read<T>(output_dev, output.data.size());

        status = miopen::TripletMarginLossBackward(handle,
                                                   workspace_dev.get(),
                                                   ws_sizeInBytes,
                                                   anchor.desc,
                                                   anchor_dev.get(),
                                                   positive.desc,
                                                   positive_dev.get(),
                                                   negative.desc,
                                                   negative_dev.get(),
                                                   dO.desc,
                                                   dO_dev.get(),
                                                   dA.desc,
                                                   dA_dev.get(),
                                                   dP.desc,
                                                   dP_dev.get(),
                                                   dN.desc,
                                                   dN_dev.get(),
                                                   margin,
                                                   p,
                                                   eps,
                                                   swap,
                                                   divisor);
        EXPECT_EQ(status, miopenStatusSuccess);
        dA.data = handle.Read<T>(dA_dev, dA.data.size());
        dP.data = handle.Read<T>(dP_dev, dP.data.size());
        dN.data = handle.Read<T>(dN_dev, dN.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;

        auto error = miopen::rms_range(ref_output, output);
        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < tolerance) << "Error forward output beyond tolerance Error: " << error
                                       << ",  Tolerance: " << tolerance;

        auto error_dA = miopen::rms_range(ref_dA, dA);
        auto error_dP = miopen::rms_range(ref_dP, dP);
        auto error_dN = miopen::rms_range(ref_dN, dN);
        EXPECT_TRUE(miopen::range_distance(ref_dA) == miopen::range_distance(dA));
        EXPECT_TRUE(miopen::range_distance(ref_dP) == miopen::range_distance(dP));
        EXPECT_TRUE(miopen::range_distance(ref_dN) == miopen::range_distance(dN));
        EXPECT_TRUE(error_dA < tolerance && error_dP < tolerance && error_dN < tolerance)
            << "Error backward output beyond tolerance Error: {" << error_dA << "," << error_dP
            << "," << error_dN << "},  Tolerance: " << tolerance;
    }

    TripletMarginLossTestCase tripletmarginloss_config;

    tensor<T> anchor;
    tensor<T> positive;
    tensor<T> negative;
    tensor<T> output;
    tensor<T> dO;
    tensor<T> dA;
    tensor<T> dP;
    tensor<T> dN;
    tensor<T> workspace;

    tensor<T> ref_output;
    tensor<T> ref_dA;
    tensor<T> ref_dP;
    tensor<T> ref_dN;

    miopen::Allocator::ManageDataPtr anchor_dev;
    miopen::Allocator::ManageDataPtr positive_dev;
    miopen::Allocator::ManageDataPtr negative_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr dO_dev;
    miopen::Allocator::ManageDataPtr dA_dev;
    miopen::Allocator::ManageDataPtr dP_dev;
    miopen::Allocator::ManageDataPtr dN_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;

    float margin;
    int p;
    float eps;
    bool swap;
    float divisor;
};
