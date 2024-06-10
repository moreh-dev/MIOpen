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

#include "marginrankingloss.hpp"
#include "../tensor_holder.hpp"
#include <miopen/env.hpp>


MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace marginrankingloss {

std::string GetFloatArg()
{
    const auto& tmp = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct MarginRankingLossTestFloat : MarginRankingLossFwdTest<float>
{
};

struct MarginRankingLossTestHalf : MarginRankingLossFwdTest<half>
{
};

struct MarginRankingLossTestBFloat16 : MarginRankingLossFwdTest<bfloat16>
{
};

struct MarginRankingLossTestFloatBwd : MarginRankingLossBwdTest<float>
{
};

struct MarginRankingLossTestHalfBwd : MarginRankingLossBwdTest<half>
{
};

struct MarginRankingLossTestBFloat16Bwd : MarginRankingLossBwdTest<bfloat16>
{
};

} // namespace marginrankingloss
using namespace marginrankingloss;

// FORWARD TEST
TEST_P(MarginRankingLossTestFloat, MarginRankingLossTest)
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

TEST_P(MarginRankingLossTestHalf, MarginRankingLossTest)
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

TEST_P(MarginRankingLossTestBFloat16, MarginRankingLossTest)
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

INSTANTIATE_TEST_SUITE_P(MarginRankingLossTestSet, MarginRankingLossTestFloat, testing::ValuesIn(MarginRankingLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(MarginRankingLossTestSet, MarginRankingLossTestHalf, testing::ValuesIn(MarginRankingLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(MarginRankingLossTestSet, MarginRankingLossTestBFloat16, testing::ValuesIn(MarginRankingLossTestConfigs()));

// BACKWARD TEST
TEST_P(MarginRankingLossTestFloatBwd, MarginRankingLossTestBwd)
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

TEST_P(MarginRankingLossTestHalfBwd, MarginRankingLossTestBwd)
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

TEST_P(MarginRankingLossTestBFloat16Bwd, MarginRankingLossTestBwd)
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

INSTANTIATE_TEST_SUITE_P(MarginRankingLossTestSet, MarginRankingLossTestFloatBwd, testing::ValuesIn(MarginRankingLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(MarginRankingLossTestSet, MarginRankingLossTestHalfBwd, testing::ValuesIn(MarginRankingLossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(MarginRankingLossTestSet, MarginRankingLossTestBFloat16Bwd, testing::ValuesIn(MarginRankingLossTestConfigs()));
