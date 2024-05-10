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

#include "pad_reflection.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace pad_reflection {

std::string GetFloatArg()
{
    const auto& tmp = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct PadReflectionFwdTestFloat : PadReflectionFwdTest<float>
{
};

struct PadReflectionFwdTestHalf : PadReflectionFwdTest<half_float::half>
{
};

struct PadReflectionFwdTestBF16 : PadReflectionFwdTest<bfloat16>
{
};

struct PadReflectionBwdTestFloat : PadReflectionBwdTest<float>
{
};

struct PadReflectionBwdTestHalf : PadReflectionBwdTest<half_float::half>
{
};

struct PadReflectionBwdTestBF16 : PadReflectionBwdTest<bfloat16>
{
};

} // namespace pad_reflection
using namespace pad_reflection;

TEST_P(PadReflectionFwdTestFloat, PadReflectionFw)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--float"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(PadReflectionFwdTestHalf, PadReflectionFw)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--half"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(PadReflectionFwdTestBF16, PadReflectionFw)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--bfloat16"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(PadReflectionBwdTestFloat, PadReflectionBw)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--float"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(PadReflectionBwdTestHalf, PadReflectionBw)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--half"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(PadReflectionBwdTestBF16, PadReflectionBw)
{
    if(miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && (GetFloatArg() == "--bfloat16"))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(PadReflectionTestSet,
                         PadReflectionFwdTestFloat,
                         testing::ValuesIn(PadReflectionTestFloatConfigs()));

INSTANTIATE_TEST_SUITE_P(PadReflectionTestSet,
                         PadReflectionFwdTestHalf,
                         testing::ValuesIn(PadReflectionTestFloatConfigs()));

INSTANTIATE_TEST_SUITE_P(PadReflectionTestSet,
                         PadReflectionFwdTestBF16,
                         testing::ValuesIn(PadReflectionTestFloatConfigs()));
INSTANTIATE_TEST_SUITE_P(PadReflectionTestSet,
                         PadReflectionBwdTestFloat,
                         testing::ValuesIn(PadReflectionTestFloatConfigs()));

INSTANTIATE_TEST_SUITE_P(PadReflectionTestSet,
                         PadReflectionBwdTestHalf,
                         testing::ValuesIn(PadReflectionTestFloatConfigs()));

INSTANTIATE_TEST_SUITE_P(PadReflectionTestSet,
                         PadReflectionBwdTestBF16,
                         testing::ValuesIn(PadReflectionTestFloatConfigs()));
