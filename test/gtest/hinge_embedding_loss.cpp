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

#include "hinge_embedding_loss.hpp"
#include "miopen/bfloat16.hpp"
#include "tensor_holder.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace loss {
struct HingeEmbeddingLossUnreducedForwardTestFloat32
    : HingeEmbeddingLossUnreducedFwdTest<float, int>
{
};

struct HingeEmbeddingLossUnreducedForwardTestFloat16 : HingeEmbeddingLossUnreducedFwdTest<half, int>
{
};

struct HingeEmbeddingLossUnreducedForwardTestBFloat16
    : HingeEmbeddingLossUnreducedFwdTest<bfloat16, int>
{
};

struct HingeEmbeddingLossUnreducedBackwardTestFloat32
    : HingeEmbeddingLossUnreducedBwdTest<float, int>
{
};

struct HingeEmbeddingLossUnreducedBackwardTestFloat16
    : HingeEmbeddingLossUnreducedBwdTest<half, int>
{
};

struct HingeEmbeddingLossUnreducedBackwardTestBFloat16
    : HingeEmbeddingLossUnreducedBwdTest<bfloat16, int>
{
};
}; // namespace loss
using namespace loss;

TEST_P(HingeEmbeddingLossUnreducedForwardTestFloat32, HingeEmbeddingLossUnreducedForwardTest)
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

INSTANTIATE_TEST_SUITE_P(HingeEmbeddingLossUnreducedForwardTestSet,
                         HingeEmbeddingLossUnreducedForwardTestFloat32,
                         testing::ValuesIn(HingeEmbeddingLossTestConfigs()));

TEST_P(HingeEmbeddingLossUnreducedForwardTestFloat16, HingeEmbeddingLossUnreducedForwardTest)
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

INSTANTIATE_TEST_SUITE_P(HingeEmbeddingLossUnreducedForwardTestSet,
                         HingeEmbeddingLossUnreducedForwardTestFloat16,
                         testing::ValuesIn(HingeEmbeddingLossTestConfigs()));

TEST_P(HingeEmbeddingLossUnreducedForwardTestBFloat16, HingeEmbeddingLossUnreducedForwardTest)
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

INSTANTIATE_TEST_SUITE_P(HingeEmbeddingLossUnreducedForwardTestSet,
                         HingeEmbeddingLossUnreducedForwardTestBFloat16,
                         testing::ValuesIn(HingeEmbeddingLossTestConfigs()));

TEST_P(HingeEmbeddingLossUnreducedBackwardTestFloat32, HingeEmbeddingLossUnreducedBackwardTest)
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

INSTANTIATE_TEST_SUITE_P(HingeEmbeddingLossUnreducedBackwardTestSet,
                         HingeEmbeddingLossUnreducedBackwardTestFloat32,
                         testing::ValuesIn(HingeEmbeddingLossTestConfigs()));

TEST_P(HingeEmbeddingLossUnreducedBackwardTestFloat16, HingeEmbeddingLossUnreducedBackwardTest)
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

INSTANTIATE_TEST_SUITE_P(HingeEmbeddingLossUnreducedBackwardTestSet,
                         HingeEmbeddingLossUnreducedBackwardTestFloat16,
                         testing::ValuesIn(HingeEmbeddingLossTestConfigs()));

TEST_P(HingeEmbeddingLossUnreducedBackwardTestBFloat16, HingeEmbeddingLossUnreducedBackwardTest)
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

INSTANTIATE_TEST_SUITE_P(HingeEmbeddingLossUnreducedBackwardTestSet,
                         HingeEmbeddingLossUnreducedBackwardTestBFloat16,
                         testing::ValuesIn(HingeEmbeddingLossTestConfigs()));
