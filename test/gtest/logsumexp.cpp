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

#include "logsumexp.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace env = miopen::env;

namespace logsumexp {

std::string GetFloatArg()
{
    const auto tmp = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(tmp.empty())
        return "";
    return tmp;
}

struct LogsumexpForwardTestFloat : LogsumexpForwardTest<float>
{
};

struct LogsumexpForwardTestHalf : LogsumexpForwardTest<half_float::half>
{
};

struct LogsumexpForwardTestBFloat16 : LogsumexpForwardTest<bfloat16>
{
};

struct LogsumexpBackwardTestFloat : LogsumexpBackwardTest<float>
{
};

struct LogsumexpBackwardTestHalf : LogsumexpBackwardTest<half_float::half>
{
};

struct LogsumexpBackwardTestBFloat16 : LogsumexpBackwardTest<bfloat16>
{
};

} // namespace logsumexp
using namespace logsumexp;

TEST_P(LogsumexpForwardTestFloat, LogsumexpTestFW)
{
    if(!MIOPEN_TEST_ALL || (env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--float")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
}

TEST_P(LogsumexpForwardTestHalf, LogsumexpTestFW)
{
    if(!MIOPEN_TEST_ALL || (env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--half")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
}

TEST_P(LogsumexpForwardTestBFloat16, LogsumexpTestFW)
{
    if(!MIOPEN_TEST_ALL || (env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--bfloat16")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
}

TEST_P(LogsumexpBackwardTestFloat, LogsumexpTestBW)
{
    if(!MIOPEN_TEST_ALL || (env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--float")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
}

TEST_P(LogsumexpBackwardTestHalf, LogsumexpTestBW)
{
    if(!MIOPEN_TEST_ALL || (env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--half")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
}

TEST_P(LogsumexpBackwardTestBFloat16, LogsumexpTestBW)
{
    if(!MIOPEN_TEST_ALL || (env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--bfloat16")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
}

INSTANTIATE_TEST_SUITE_P(LogsumexpTestSet, LogsumexpForwardTestFloat, testing::ValuesIn(LogsumexpTestConfigs()));
INSTANTIATE_TEST_SUITE_P(LogsumexpTestSet, LogsumexpForwardTestHalf, testing::ValuesIn(LogsumexpTestConfigs()));
INSTANTIATE_TEST_SUITE_P(LogsumexpTestSet, LogsumexpForwardTestBFloat16, testing::ValuesIn(LogsumexpTestConfigs()));
INSTANTIATE_TEST_SUITE_P(LogsumexpTestSet, LogsumexpBackwardTestFloat, testing::ValuesIn(LogsumexpTestConfigs()));
INSTANTIATE_TEST_SUITE_P(LogsumexpTestSet, LogsumexpBackwardTestHalf, testing::ValuesIn(LogsumexpTestConfigs()));
INSTANTIATE_TEST_SUITE_P(LogsumexpTestSet, LogsumexpBackwardTestBFloat16, testing::ValuesIn(LogsumexpTestConfigs()));
