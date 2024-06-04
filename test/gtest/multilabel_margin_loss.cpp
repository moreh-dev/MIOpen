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

#include "multilabel_margin_loss.hpp"
#include "miopen/bfloat16.hpp"
#include "tensor_holder.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace multilabel_margin_loss {

struct MultilabelMarginLossForwardTestFloat32 : MultilabelMarginLossFwdTest<float, int>
{
};

struct MultilabelMarginLossForwardTestFloat16 : MultilabelMarginLossFwdTest<half, int>
{
};

struct MultilabelMarginLossForwardTestBFloat16 : MultilabelMarginLossFwdTest<bfloat16, int>
{
};

}; // namespace multilabel_margin_loss
using namespace multilabel_margin_loss;
// =========================== Reduced Forward Test ===========================
TEST_P(MultilabelMarginLossForwardTestFloat32, MultilabelMarginLossFwdTest)
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

INSTANTIATE_TEST_SUITE_P(MultilabelMarginLossForwardTestSet,
                         MultilabelMarginLossForwardTestFloat32,
                         testing::ValuesIn(MultilabelMarginLossTestFloatConfigs()));

TEST_P(MultilabelMarginLossForwardTestFloat16, MultilabelMarginLossFwdTest)
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

INSTANTIATE_TEST_SUITE_P(MultilabelMarginLossForwardTestSet,
                         MultilabelMarginLossForwardTestFloat16,
                         testing::ValuesIn(MultilabelMarginLossTestFloatConfigs()));

TEST_P(MultilabelMarginLossForwardTestBFloat16, MultilabelMarginLossFwdTest)
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

INSTANTIATE_TEST_SUITE_P(MultilabelMarginLossForwardTestSet,
                         MultilabelMarginLossForwardTestBFloat16,
                         testing::ValuesIn(MultilabelMarginLossTestFloatConfigs()));
// =========================== Reduced Forward Test End ===========================