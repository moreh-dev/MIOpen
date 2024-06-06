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

#include "sigmoid_focal_loss.hpp"
#include "miopen/bfloat16.hpp"
#include "tensor_holder.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace sigmoidfocalloss {
struct SigmoidFocalLossForwardTestFloat32 : SigmoidFocalLossFwdTest<float>
{
};

struct SigmoidFocalLossForwardTestFloat16 : SigmoidFocalLossFwdTest<half>
{
};

struct SigmoidFocalLossForwardTestBFloat16 : SigmoidFocalLossFwdTest<bfloat16>
{
};

struct SigmoidFocalLossBackwardTestFloat32 : SigmoidFocalLossBwdTest<float>
{
};

struct SigmoidFocalLossBackwardTestFloat16 : SigmoidFocalLossBwdTest<half>
{
};

struct SigmoidFocalLossBackwardTestBFloat16 : SigmoidFocalLossBwdTest<bfloat16>
{
};

struct SigmoidFocalLossUnreducedForwardTestFloat32 : SigmoidFocalLossUnreducedFwdTest<float>
{
};

struct SigmoidFocalLossUnreducedForwardTestFloat16 : SigmoidFocalLossUnreducedFwdTest<half>
{
};

struct SigmoidFocalLossUnreducedForwardTestBFloat16 : SigmoidFocalLossUnreducedFwdTest<bfloat16>
{
};

struct SigmoidFocalLossUnreducedBackwardTestFloat32 : SigmoidFocalLossUnreducedBwdTest<float>
{
};

struct SigmoidFocalLossUnreducedBackwardTestFloat16 : SigmoidFocalLossUnreducedBwdTest<half>
{
};

struct SigmoidFocalLossUnreducedBackwardTestBFloat16 : SigmoidFocalLossUnreducedBwdTest<bfloat16>
{
};
}; // namespace sigmoidfocalloss

using namespace sigmoidfocalloss;
// =========================== Reduced Forward Test ===========================
TEST_P(SigmoidFocalLossForwardTestFloat32, SigmoidFocalLossForwardTest)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SigmoidFocalLossForwardTestSet,
                         SigmoidFocalLossForwardTestFloat32,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));

TEST_P(SigmoidFocalLossForwardTestFloat16, SigmoidFocalLossForwardTest)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SigmoidFocalLossForwardTestSet,
                         SigmoidFocalLossForwardTestFloat16,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));

TEST_P(SigmoidFocalLossForwardTestBFloat16, SigmoidFocalLossForwardTest)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SigmoidFocalLossForwardTestSet,
                         SigmoidFocalLossForwardTestBFloat16,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));

// =========================== Reduced Backward Test ===========================
TEST_P(SigmoidFocalLossBackwardTestFloat32, SigmoidFocalLossBackwardTest)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SigmoidFocalLossBackwardTestSet,
                         SigmoidFocalLossBackwardTestFloat32,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));

TEST_P(SigmoidFocalLossBackwardTestFloat16, SigmoidFocalLossBackwardTest)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SigmoidFocalLossBackwardTestSet,
                         SigmoidFocalLossBackwardTestFloat16,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));

TEST_P(SigmoidFocalLossBackwardTestBFloat16, SigmoidFocalLossBackwardTest)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SigmoidFocalLossBackwardTestSet,
                         SigmoidFocalLossBackwardTestBFloat16,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));

// =========================== Unreduced Forward Test ===========================
TEST_P(SigmoidFocalLossUnreducedForwardTestFloat32, SigmoidFocalLossUnreducedForwardTest)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SigmoidFocalLossUnreducedForwardTestSet,
                         SigmoidFocalLossUnreducedForwardTestFloat32,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));

TEST_P(SigmoidFocalLossUnreducedForwardTestFloat16, SigmoidFocalLossUnreducedForwardTest)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SigmoidFocalLossUnreducedForwardTestSet,
                         SigmoidFocalLossUnreducedForwardTestFloat16,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));

TEST_P(SigmoidFocalLossUnreducedForwardTestBFloat16, SigmoidFocalLossUnreducedForwardTest)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SigmoidFocalLossUnreducedForwardTestSet,
                         SigmoidFocalLossUnreducedForwardTestBFloat16,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));

// =========================== Unreduced Backward Test ===========================
TEST_P(SigmoidFocalLossUnreducedBackwardTestFloat32, SigmoidFocalLossUnreducedBackwardTest)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SigmoidFocalLossUnreducedBackwardTestSet,
                         SigmoidFocalLossUnreducedBackwardTestFloat32,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));

TEST_P(SigmoidFocalLossUnreducedBackwardTestFloat16, SigmoidFocalLossUnreducedBackwardTest)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SigmoidFocalLossUnreducedBackwardTestSet,
                         SigmoidFocalLossUnreducedBackwardTestFloat16,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));

TEST_P(SigmoidFocalLossUnreducedBackwardTestBFloat16, SigmoidFocalLossUnreducedBackwardTest)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SigmoidFocalLossUnreducedBackwardTestSet,
                         SigmoidFocalLossUnreducedBackwardTestBFloat16,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));
