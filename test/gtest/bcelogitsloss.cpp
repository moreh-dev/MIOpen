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

#include "bcelogitsloss.hpp"
#include "miopen/bfloat16.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace smoothl1loss {

std::string GetFloatArg()
{
    const auto& tmp = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct BCELogitsLossTestForwardFloat : BCELogitsLossTestForward<float>
{
};

struct BCELogitsLossTestForwardHalf : BCELogitsLossTestForward<half>
{
};

struct BCELogitsLossTestForwardBfloat16 : BCELogitsLossTestForward<bfloat16>
{
};

struct BCELogitsLossTestBackwardFloat : BCELogitsLossTestBackward<float>
{
};

struct BCELogitsLossTestBackwardHalf : BCELogitsLossTestBackward<half>
{
};

struct BCELogitsLossTestBackwardBfloat16 : BCELogitsLossTestBackward<bfloat16>
{
};

} // namespace smoothl1loss
using namespace smoothl1loss;

TEST_P(BCELogitsLossTestForwardFloat, BCELogitsLossTestFw)
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

TEST_P(BCELogitsLossTestForwardHalf, BCELogitsLossTestFw)
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

TEST_P(BCELogitsLossTestForwardBfloat16, BCELogitsLossTestFw)
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

TEST_P(BCELogitsLossTestBackwardFloat, BCELogitsLossTestBw)
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

TEST_P(BCELogitsLossTestBackwardHalf, BCELogitsLossTestBw)
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

TEST_P(BCELogitsLossTestBackwardBfloat16, BCELogitsLossTestBw)
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

INSTANTIATE_TEST_SUITE_P(BCELogitsLossTestSet,
                         BCELogitsLossTestForwardFloat,
                         testing::ValuesIn(BCELogitsLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(BCELogitsLossTestSet,
                         BCELogitsLossTestForwardHalf,
                         testing::ValuesIn(BCELogitsLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(BCELogitsLossTestSet,
                         BCELogitsLossTestForwardBfloat16,
                         testing::ValuesIn(BCELogitsLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(BCELogitsLossTestSet,
                         BCELogitsLossTestBackwardFloat,
                         testing::ValuesIn(BCELogitsLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(BCELogitsLossTestSet,
                         BCELogitsLossTestBackwardHalf,
                         testing::ValuesIn(BCELogitsLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(BCELogitsLossTestSet,
                         BCELogitsLossTestBackwardBfloat16,
                         testing::ValuesIn(BCELogitsLossTestConfigs()));
