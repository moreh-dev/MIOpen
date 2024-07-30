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
#include <miopen/env.hpp>
#include "interpolate.hpp"

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace interpolate {

std::string GetFloatArg()
{
    const auto& tmp = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct GPU_Interpolate_fwd_FP32 : InterpolateTestFwd<float>
{
};

struct GPU_Interpolate_fwd_FP16 : InterpolateTestFwd<half>
{
};

struct GPU_Interpolate_fwd_BFP16 : InterpolateTestFwd<bfloat16>
{
};

struct GPU_Interpolate_bwd_FP32 : InterpolateTestBwd<float>
{
};

struct GPU_Interpolate_bwd_FP16 : InterpolateTestBwd<half>
{
};

struct GPU_Interpolate_bwd_BFP16 : InterpolateTestBwd<bfloat16>
{
};

} // namespace interpolate
using namespace interpolate;

// FORWARD TEST
TEST_P(GPU_Interpolate_fwd_FP32, InterpolateTest)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--float") ||
       miopen::IsUnset(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(GPU_Interpolate_fwd_FP16, InterpolateTest)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--half") ||
       miopen::IsUnset(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(GPU_Interpolate_fwd_BFP16, InterpolateTest)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--bfloat16") ||
       miopen::IsUnset(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(InterpolateTestSet,
                         GPU_Interpolate_fwd_FP32,
                         testing::ValuesIn(InterpolateTestConfigs()));
INSTANTIATE_TEST_SUITE_P(InterpolateTestSet,
                         GPU_Interpolate_fwd_FP16,
                         testing::ValuesIn(InterpolateTestConfigs()));
INSTANTIATE_TEST_SUITE_P(InterpolateTestSet,
                         GPU_Interpolate_fwd_BFP16,
                         testing::ValuesIn(InterpolateTestConfigs()));

// BACKWARD TEST
TEST_P(GPU_Interpolate_bwd_FP32, InterpolateTestBwd)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--float") ||
       miopen::IsUnset(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(GPU_Interpolate_bwd_FP16, InterpolateTestBwd)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--half") ||
       miopen::IsUnset(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(GPU_Interpolate_bwd_BFP16, InterpolateTestBwd)
{
    if((miopen::IsEnabled(ENV(MIOPEN_TEST_ALL)) && GetFloatArg() == "--bfloat16") ||
       miopen::IsUnset(ENV(MIOPEN_TEST_ALL)))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(InterpolateTestSet,
                         GPU_Interpolate_bwd_FP32,
                         testing::ValuesIn(InterpolateTestConfigs()));
INSTANTIATE_TEST_SUITE_P(InterpolateTestSet,
                         GPU_Interpolate_bwd_FP16,
                         testing::ValuesIn(InterpolateTestConfigs()));
INSTANTIATE_TEST_SUITE_P(InterpolateTestSet,
                         GPU_Interpolate_bwd_BFP16,
                         testing::ValuesIn(InterpolateTestConfigs()));
