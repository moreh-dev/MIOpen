/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>
#include <serialize.hpp>
#include <fusionHost.hpp>
#include <miopen/fusion.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"
#include "cba.hpp"

struct ConvBiasActivInferTestFloat : ConvBiasActivInferTest<float>
{
};
//

template <typename Solver, typename TestCase>
void RunSolver(miopen::FusionPlanDescriptor& fusePlanDesc,
               const miopen::fusion::FusionInvokeParams plan_params,
               const TestCase& conv_config,
               bool& test_skipped)
{
    auto& handle = get_handle();
    Solver solv{};
    auto fusion_ctx = miopen::FusionContext{&fusePlanDesc, handle};
    fusion_ctx.DetectRocm();
    if(!solv.IsApplicable(fusion_ctx))
    {
        test_skipped = true;
        GTEST_SKIP() << solv.SolverDbId() << " Not Applicable" << conv_config;
    }
    ASSERT_TRUE(solv.IsApplicable(fusion_ctx));
    auto sol = solv.GetSolution(fusion_ctx);
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, plan_params);
    handle.Finish();
}
template <typename Solver>
void RunTunableSolver(miopen::FusionPlanDescriptor& fusePlanDesc,
                      const miopen::fusion::FusionInvokeParams plan_params,
                      const ConvTestCase& conv_config,
                      bool& test_skipped)
{
    auto& handle = get_handle();
    Solver solv{};
    auto fusion_ctx = miopen::FusionContext{&fusePlanDesc, handle};
    fusion_ctx.DetectRocm();
    if(!solv.IsApplicable(fusion_ctx))
    {
        test_skipped = true;
        GTEST_SKIP() << solv.SolverDbId() << " Not Applicable" << conv_config;
    }
    ASSERT_TRUE(solv.IsApplicable(fusion_ctx));
    auto sol = solv.GetSolution(fusion_ctx, solv.GetDefaultPerformanceConfig(fusion_ctx));
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, plan_params);
    handle.Finish();
}

// TEST_P(ConvBiasActivInferTestFloat, ConvBiasActivAsm1x1UFloat)
// {
//     RunTunableSolver<miopen::solver::fusion::ConvBiasActivAsm1x1U>(fusePlanDesc, plan_params,
//     conv_config, test_skipped);
// }
// TEST_P(ConvBiasActivInferTestFloat, ConvOclDirectFwdFused)
// {
//     RunTunableSolver<miopen::solver::fusion::ConvOclDirectFwdFused>(fusePlanDesc, plan_params,
//     conv_config, test_skipped);
// }
TEST_P(ConvBiasActivInferTestFloat, ConvBinWinogradRxSFused)
{
    RunSolver<miopen::solver::fusion::ConvBinWinogradRxSFused>(
        fusePlanDesc, plan_params, conv_config, test_skipped);
}
TEST_P(ConvBiasActivInferTestFloat, ConvBinWinogradRxSf2x3g1Fused)
{
    RunSolver<miopen::solver::fusion::ConvBinWinogradRxSf2x3g1Fused>(
        fusePlanDesc, plan_params, conv_config, test_skipped);
}
INSTANTIATE_TEST_SUITE_P(CBAInferSolverTest,
                         ConvBiasActivInferTestFloat,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(GetNetwork1())));