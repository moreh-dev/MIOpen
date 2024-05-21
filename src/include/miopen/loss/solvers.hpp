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
#pragma once

#include <miopen/loss/problem_description.hpp>
#include <miopen/solver.hpp>

#include <utility>

namespace miopen {

namespace solver {

namespace loss {

using HingeEmbeddingLossFwdSolverBase =
    NonTunableSolverBase<ExecutionContext, miopen::loss::HingeEmbeddingLossFwdProblemDescription>;

struct HingeEmbeddingLossFwd final : HingeEmbeddingLossFwdSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<HingeEmbeddingLossFwd>();
    }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::loss::HingeEmbeddingLossFwdProblemDescription& problem) const override;

    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::loss::HingeEmbeddingLossFwdProblemDescription& problem) const override;

    std::size_t GetWorkspaceSize(
        const ExecutionContext& context,
        const miopen::loss::HingeEmbeddingLossFwdProblemDescription& problem) const override;

    bool MayNeedWorkspace() const override { return true; }
};

using HingeEmbeddingLossBwdSolverBase =
    NonTunableSolverBase<ExecutionContext, miopen::loss::HingeEmbeddingLossBwdProblemDescription>;

struct HingeEmbeddingLossBwd final : HingeEmbeddingLossBwdSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<HingeEmbeddingLossBwd>();
    }

    bool IsApplicable(
        const ExecutionContext& context,
        const miopen::loss::HingeEmbeddingLossBwdProblemDescription& problem) const override;

    ConvSolution GetSolution(
        const ExecutionContext& context,
        const miopen::loss::HingeEmbeddingLossBwdProblemDescription& problem) const override;
};

using HingeEmbeddingLossUnreducedFwdSolverBase =
    NonTunableSolverBase<ExecutionContext,
                         miopen::loss::HingeEmbeddingLossUnreducedFwdProblemDescription>;

struct HingeEmbeddingLossUnreducedFwd final : HingeEmbeddingLossUnreducedFwdSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<HingeEmbeddingLossUnreducedFwd>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::loss::HingeEmbeddingLossUnreducedFwdProblemDescription& problem)
        const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::loss::HingeEmbeddingLossUnreducedFwdProblemDescription&
                                 problem) const override;
};

using HingeEmbeddingLossUnreducedBwdSolverBase =
    NonTunableSolverBase<ExecutionContext,
                         miopen::loss::HingeEmbeddingLossUnreducedBwdProblemDescription>;

struct HingeEmbeddingLossUnreducedBwd final : HingeEmbeddingLossUnreducedBwdSolverBase
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<HingeEmbeddingLossUnreducedBwd>();
    }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::loss::HingeEmbeddingLossUnreducedBwdProblemDescription& problem)
        const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::loss::HingeEmbeddingLossUnreducedBwdProblemDescription&
                                 problem) const override;
};

} // namespace loss

} // namespace solver

} // namespace miopen
