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

#include <cstddef>
#include <limits>
#include <ostream>
#include <vector>
#define MIOPEN_BETA_API 1

#include "cpu_sgd.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/sgd.hpp>

struct SGDTestCase
{
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    size_t D;
    double lr;
    double momentum;
    double dampening;
    double weightDecay;
    char nesterov;
    char momentumInitialized;
    std::vector<size_t> strides;
    friend std::ostream& operator<<(std::ostream& os, const SGDTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " LearningRate:" << tc.lr << " Momentum:" << tc.momentum
                  << " Dampening:" << tc.dampening << " WeightDecay:" << tc.weightDecay
                  << " Nesterov:" << (int)tc.nesterov
                  << " MomentumInitialized:" << (int)tc.momentumInitialized;
    }

    std::vector<size_t> GetDims()
    {
        std::vector<size_t> dims;
        if(N != 0)
            dims.push_back(N);
        if(C != 0)
            dims.push_back(C);
        if(H != 0)
            dims.push_back(H);
        if(W != 0)
            dims.push_back(W);
        if(D != 0)
            dims.push_back(D);

        if(dims.empty())
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }

        return dims;
    }

    std::vector<size_t> GetStrides()
    {
        size_t n_dims = GetDims().size();
        if(!strides.empty() && strides.size() != n_dims)
        {
            std::cout << "Error Input Tensor Strides\n" << std::endl;
            return std::vector<size_t>({0});
        }

        return strides;
    }
};

std::vector<SGDTestCase> SGDTestConfigs()
{ // n c d h w lr momentum dampening weightDecay nesterov momentumInitialized
    // clang-format off
    return {
        {32, 0, 0, 0, 0,     0.004, 0.9, 0,       0,      1, 0, {}},
        {32, 3, 0, 3, 3,     0.004, 0.9, 0,       0,      1, 0, {}},
        {32, 32, 3, 3, 3,    0.004, 0.9, 0,       0,      1, 0, {}},
        {64, 32, 3, 3, 3,    0.004, 0.9, 0,       0,      1, 0, {}},
        {32, 0, 0, 0, 0,     0.004, 0.9, 0,       0,      1, 1, {}},
        {32, 3, 0, 3, 3,     0.004, 0.9, 0,       0,      1, 1, {}},
        {32, 32, 3, 3, 3,    0.004, 0.9, 0,       0,      1, 1, {}},
        {64, 32, 3, 3, 3,    0.004, 0.9, 0,       0,      1, 1, {}},
        {61, 3, 0, 11, 11,   0.01,  0.9, 0,       0.0005, 0, 0, {}},
        {192, 64, 0, 5, 5,   0.01,  0.9, 0,       0.0005, 0, 0, {}},
        {61, 3, 0, 11, 11,   0.01,  0.9, 0,       0.0005, 0, 1, {}},
        {192, 64, 0, 5, 5,   0.01,  0.9, 0,       0.0005, 0, 1, {}},
        {64, 3, 0, 3, 3,     0.01,  0.9, 0.0005,  0,      0, 0, {}},
        {64, 64, 0, 1, 1,    0.01,  0.9, 0.0005,  0,      0, 0, {}},
        {64, 3, 0, 3, 3,     0.01,  0.9, 0.0005,  0,      0, 1, {}},
        {64, 64, 0, 1, 1,    0.01,  0.9, 0.0005,  0,      0, 1, {}},
        {29, 0, 0, 0, 0,     0.001, 0.9, 0,       0,      1, 0, {1}},
        {31, 0, 0, 0, 0,     0.001, 0.9, 0,       0,      1, 0, {4}},
        {64, 32, 0, 0, 0,    0.001, 0.9, 0,       0,      1, 0, {32, 4}},
        {61, 32, 0, 0, 0,    0.001, 0.9, 0,       0,      1, 0, {32, 3}},
        {127, 61, 0, 0, 0,   0.001, 0.9, 0,       0,      1, 0, {61, 3}},
        {11, 16, 32, 0, 0,   0.001, 0.9, 0,       0,      1, 0, {1024, 64, 4}},
        {31, 31, 31, 0, 0,   0.001, 0.9, 0,       0,      1, 0, {961, 31, 5}},
        {31, 31, 31, 0, 0,   0.001, 0.9, 0,       0,      1, 0, {961, 155, 5}},
        {15, 4, 4, 4, 4,     0.001, 0.9, 0,       0,      1, 0, {256, 64, 16, 4, 4}},
        {15, 4, 4, 4, 4,     0.001, 0.9, 0,       0,      1, 0, {256, 64, 16, 16, 1}},
        {15, 4, 4, 4, 4,     0.001, 0.9, 0,       0,      1, 0, {256, 64, 64, 4, 1}},
        {15, 4, 4, 4, 4,     0.001, 0.9, 0,       0,      1, 0, {256, 256, 16, 4, 1}},
        {15, 4, 4, 4, 4,     0.001, 0.9, 0,       0,      1, 0, {1024, 64, 16, 4, 1}},
        {15, 4, 4, 4, 4,     0.001, 0.9, 0,       0,      1, 0, {256, 64, 32, 8, 2}},
        {15, 4, 4, 4, 4,     0.001, 0.9, 0,       0,      1, 0, {256, 128, 32, 8, 1}},
        {15, 4, 4, 4, 4,     0.001, 0.9, 0,       0,      1, 0, {512, 128, 32, 4, 1}},
    };
    // clang-format on
}

template <typename T = float>
struct SGDTest : public ::testing::TestWithParam<SGDTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        SGD_config     = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        lr                   = SGD_config.lr;
        momentum             = SGD_config.momentum;
        dampening            = SGD_config.dampening;
        weight_decay         = SGD_config.weightDecay;
        nesterov             = SGD_config.nesterov;
        momentum_initialized = SGD_config.momentumInitialized;

        auto dims    = SGD_config.GetDims();
        auto strides = SGD_config.GetStrides();

        if(strides.empty())
        {
            param_input                = tensor<T>{dims}.generate(gen_value);
            grad                       = tensor<T>{dims}.generate(gen_value);
            momentum_buffer_input      = tensor<T>{dims}.generate(gen_value);
            param_output               = tensor<T>{dims}.generate(gen_value);
            momentum_buffer_output     = tensor<T>{dims}.generate(gen_value);
            ref_param_output           = tensor<T>(param_output);
            ref_momentum_buffer_output = tensor<T>(momentum_buffer_output);
        }
        else
        {
            param_input                = tensor<T>{dims, strides}.generate(gen_value);
            grad                       = tensor<T>{dims, strides}.generate(gen_value);
            momentum_buffer_input      = tensor<T>{dims, strides}.generate(gen_value);
            param_output               = tensor<T>{dims, strides}.generate(gen_value);
            momentum_buffer_output     = tensor<T>{dims, strides}.generate(gen_value);
            ref_param_output           = tensor<T>(param_output);
            ref_momentum_buffer_output = tensor<T>(momentum_buffer_output);
        }

        param_input_dev            = handle.Write(param_input.data);
        param_output_dev           = handle.Write(param_output.data);
        grad_dev                   = handle.Write(grad.data);
        momentum_buffer_input_dev  = handle.Write(momentum_buffer_input.data);
        momentum_buffer_output_dev = handle.Write(momentum_buffer_output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        cpu_SGD_forward<T>(param_input,
                           ref_param_output,
                           grad,
                           momentum_buffer_input,
                           ref_momentum_buffer_output,
                           lr,
                           momentum,
                           dampening,
                           weight_decay,
                           nesterov,
                           momentum_initialized);
        miopenStatus_t status;

        status = miopen::SGDForward(handle,
                                    param_input.desc,
                                    param_input_dev.get(),
                                    param_output.desc,
                                    param_output_dev.get(),
                                    grad.desc,
                                    grad_dev.get(),
                                    momentum_buffer_input.desc,
                                    momentum_buffer_input_dev.get(),
                                    momentum_buffer_output.desc,
                                    momentum_buffer_output_dev.get(),
                                    lr,
                                    momentum,
                                    dampening,
                                    weight_decay,
                                    nesterov,
                                    momentum_initialized);

        EXPECT_EQ(status, miopenStatusSuccess);

        param_output.data = handle.Read<T>(param_output_dev, param_output.data.size());
        momentum_buffer_output.data =
            handle.Read<T>(momentum_buffer_output_dev, momentum_buffer_output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto param_error = miopen::rms_range(ref_param_output, param_output);
        auto momentum_buffer_error =
            miopen::rms_range(ref_momentum_buffer_output, momentum_buffer_output);

        for(int i = 0; i < 29; ++i)
        {
            fprintf(stderr, "ref:%f mine:%f\n", float(ref_param_output[i]), float(param_output[i]));
        }

        EXPECT_TRUE(miopen::range_distance(ref_param_output) ==
                    miopen::range_distance(param_output));
        EXPECT_TRUE(param_error < threshold * 10)
            << "Error param output beyond tolerance Error:" << param_error
            << ",  Thresholdx10: " << threshold * 10;

        EXPECT_TRUE(miopen::range_distance(ref_momentum_buffer_output) ==
                    miopen::range_distance(momentum_buffer_output));
        EXPECT_TRUE(momentum_buffer_error < threshold * 10)
            << "Error momentum buffer output beyond tolerance Error:" << momentum_buffer_error
            << ",  Thresholdx10: " << threshold * 10;
    }
    SGDTestCase SGD_config;

    tensor<T> param_input;
    tensor<T> param_output;
    tensor<T> grad;
    tensor<T> momentum_buffer_input;
    tensor<T> momentum_buffer_output;

    tensor<T> ref_param_output;
    tensor<T> ref_momentum_buffer_output;

    miopen::Allocator::ManageDataPtr param_input_dev;
    miopen::Allocator::ManageDataPtr param_output_dev;
    miopen::Allocator::ManageDataPtr grad_dev;
    miopen::Allocator::ManageDataPtr momentum_buffer_input_dev;
    miopen::Allocator::ManageDataPtr momentum_buffer_output_dev;

    double lr;
    double momentum;
    double dampening;
    double weight_decay;
    char nesterov;
    char momentum_initialized;
};
