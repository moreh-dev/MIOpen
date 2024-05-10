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

#include <miopen/pad_reflection/problem_description.hpp>
#include <miopen/solver.hpp>
#include <utility>

namespace miopen {

namespace solver {

namespace pad_reflection {

// using PadReflectionSolver =
//     NonTunableSolverBase<ExecutionContext, miopen::pad_reflection::ProblemDescription>;

// struct PadReflection final : PadReflectionSolver
// {
//     const std::string& SolverDbId() const override { return GetSolverDbId<PadReflection>(); }

//     bool IsApplicable(const ExecutionContext& context,
//                       const miopen::pad_reflection::ProblemDescription& problem) const override;

//     ConvSolution
//     GetSolution(const ExecutionContext& context,
//                 const miopen::pad_reflection::ProblemDescription& problem) const override;
// };

using PadReflection1dFwdContiguousSolver =
    NonTunableSolverBase<ExecutionContext, miopen::pad_reflection::PadReflection1dFwdContiguousProblemDescription>;

struct PadReflection1dFwdContiguous final : PadReflection1dFwdContiguousSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<PadReflection1dFwdContiguous>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pad_reflection::PadReflection1dFwdContiguousProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::pad_reflection::PadReflection1dFwdContiguousProblemDescription& problem) const override;
};

using PadReflection1dFwdSolver =
    NonTunableSolverBase<ExecutionContext, miopen::pad_reflection::PadReflection1dFwdProblemDescription>;

struct PadReflection1dFwd final : PadReflection1dFwdSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<PadReflection1dFwd>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pad_reflection::PadReflection1dFwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::pad_reflection::PadReflection1dFwdProblemDescription& problem) const override;
};

using PadReflection1dBwdContiguousSolver =
    NonTunableSolverBase<ExecutionContext, miopen::pad_reflection::PadReflection1dBwdContiguousProblemDescription>;

struct PadReflection1dBwdContiguous final : PadReflection1dBwdContiguousSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<PadReflection1dBwdContiguous>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pad_reflection::PadReflection1dBwdContiguousProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::pad_reflection::PadReflection1dBwdContiguousProblemDescription& problem) const override;
};

using PadReflection1dBwdSolver =
    NonTunableSolverBase<ExecutionContext, miopen::pad_reflection::PadReflection1dBwdProblemDescription>;

struct PadReflection1dBwd final : PadReflection1dBwdSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<PadReflection1dBwd>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pad_reflection::PadReflection1dBwdProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::pad_reflection::PadReflection1dBwdProblemDescription& problem) const override;
};

} // namespace pad_reflection

} // namespace solver

} // namespace miopen
