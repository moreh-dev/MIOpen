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

#include "indexselect.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace env = miopen::env;

namespace indexselect {

std::string GetFloatArg()
{
    const auto tmp = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct IndexSelectFwdTestFloat : IndexSelectFwdTest<float>
{
};

struct IndexSelectBwdTestFloat : IndexSelectBwdTest<float>
{
};

struct IndexSelectFwdTestHalf : IndexSelectFwdTest<half_float::half>
{
};

struct IndexSelectBwdTestHalf : IndexSelectBwdTest<half_float::half>
{
};

struct IndexSelectFwdTestBFloat16 : IndexSelectFwdTest<bfloat16>
{
};

struct IndexSelectBwdTestBFloat16 : IndexSelectBwdTest<bfloat16>
{
};

} // namespace indexselect
using namespace indexselect;

TEST_P(IndexSelectFwdTestFloat, IndexSelectFwdTest)
{
    if(env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--float"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(IndexSelectBwdTestFloat, IndexSelectBwdTest)
{
    if(env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--float"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(IndexSelectFwdTestHalf, IndexSelectFwdTest)
{
    if(env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--half"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(IndexSelectBwdTestHalf, IndexSelectBwdTest)
{
    if(env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--half"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(IndexSelectFwdTestBFloat16, IndexSelectFwdTest)
{
    if(env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--bfloat16"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(IndexSelectBwdTestBFloat16, IndexSelectBwdTest)
{
    if(env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--bfloat16"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(IndexSelectTestSet,
                         IndexSelectFwdTestFloat,
                         testing::ValuesIn(IndexSelectFwdTestConfigs()));

INSTANTIATE_TEST_SUITE_P(IndexSelectTestSet,
                         IndexSelectBwdTestFloat,
                         testing::ValuesIn(IndexSelectBwdTestConfigs()));

INSTANTIATE_TEST_SUITE_P(IndexSelectTestSet,
                         IndexSelectFwdTestHalf,
                         testing::ValuesIn(IndexSelectFwdTestConfigs()));
INSTANTIATE_TEST_SUITE_P(IndexSelectTestSet,
                         IndexSelectBwdTestHalf,
                         testing::ValuesIn(IndexSelectBwdTestConfigs()));
INSTANTIATE_TEST_SUITE_P(IndexSelectTestSet,
                         IndexSelectFwdTestBFloat16,
                         testing::ValuesIn(IndexSelectFwdTestConfigs()));
INSTANTIATE_TEST_SUITE_P(IndexSelectTestSet,
                         IndexSelectBwdTestBFloat16,
                         testing::ValuesIn(IndexSelectBwdTestConfigs()));