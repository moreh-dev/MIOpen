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
#include "miopen/bfloat16.hpp"
#include <miopen/env.hpp>
#include "softmaxcrossentropywithlogits.hpp"

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace softmaxcrossentropywithlogits {

std::string GetFloatArg()
{
    const auto& tmp = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct SoftmaxCrossEntropyWithLogitsTestFloat : SoftmaxCrossEntropyWithLogitsTest<float>
{
};

struct SoftmaxCrossEntropyWithLogitsTestHalf : SoftmaxCrossEntropyWithLogitsTest<half>
{
};

struct SoftmaxCrossEntropyWithLogitsTestBFloat16 : SoftmaxCrossEntropyWithLogitsTest<bfloat16>
{
};

struct SoftmaxCrossEntropyWithLogitsTestFloatBwd : SoftmaxCrossEntropyWithLogitsTestBwd<float>
{
};

struct SoftmaxCrossEntropyWithLogitsTestHalfBwd : SoftmaxCrossEntropyWithLogitsTestBwd<half>
{
};

struct SoftmaxCrossEntropyWithLogitsTestBFloat16Bwd : SoftmaxCrossEntropyWithLogitsTestBwd<bfloat16>
{
};

} // namespace softmaxcrossentropywithlogits
using namespace softmaxcrossentropywithlogits;

// FORWARD TEST
TEST_P(SoftmaxCrossEntropyWithLogitsTestFloat, SoftmaxCrossEntropyWithLogitsTest)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--float") ||
       GetFloatArg() == "--testall")
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(SoftmaxCrossEntropyWithLogitsTestHalf, SoftmaxCrossEntropyWithLogitsTest)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--half") ||
       GetFloatArg() == "--testall")
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(SoftmaxCrossEntropyWithLogitsTestBFloat16, SoftmaxCrossEntropyWithLogitsTest)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--bfloat16") ||
       GetFloatArg() == "--testall")
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SoftmaxCrossEntropyWithLogitsTestSet,
                         SoftmaxCrossEntropyWithLogitsTestFloat,
                         testing::ValuesIn(SoftmaxCrossEntropyWithLogitsTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SoftmaxCrossEntropyWithLogitsTestSet,
                         SoftmaxCrossEntropyWithLogitsTestHalf,
                         testing::ValuesIn(SoftmaxCrossEntropyWithLogitsTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SoftmaxCrossEntropyWithLogitsTestSet,
                         SoftmaxCrossEntropyWithLogitsTestBFloat16,
                         testing::ValuesIn(SoftmaxCrossEntropyWithLogitsTestConfigs()));

// BACKWARD TEST
TEST_P(SoftmaxCrossEntropyWithLogitsTestFloatBwd, SoftmaxCrossEntropyWithLogitsTestBwd)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--float") ||
       GetFloatArg() == "--testall")
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(SoftmaxCrossEntropyWithLogitsTestHalfBwd, SoftmaxCrossEntropyWithLogitsTestBwd)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--half") ||
       GetFloatArg() == "--testall")
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(SoftmaxCrossEntropyWithLogitsTestBFloat16Bwd, SoftmaxCrossEntropyWithLogitsTestBwd)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--bfloat16") ||
       GetFloatArg() == "--testall")
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(SoftmaxCrossEntropyWithLogitsTestSet,
                         SoftmaxCrossEntropyWithLogitsTestFloatBwd,
                         testing::ValuesIn(SoftmaxCrossEntropyWithLogitsTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SoftmaxCrossEntropyWithLogitsTestSet,
                         SoftmaxCrossEntropyWithLogitsTestHalfBwd,
                         testing::ValuesIn(SoftmaxCrossEntropyWithLogitsTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SoftmaxCrossEntropyWithLogitsTestSet,
                         SoftmaxCrossEntropyWithLogitsTestBFloat16Bwd,
                         testing::ValuesIn(SoftmaxCrossEntropyWithLogitsTestConfigs()));
