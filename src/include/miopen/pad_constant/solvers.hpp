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

#include "miopen/solver.hpp"
#include "problem_description.hpp"
#include <cstddef>
#include <string>

namespace miopen {
namespace solver {
namespace pad_constant_fwd {
using PadConstantFwdSolver =
    NonTunableSolverBase<ExecutionContext, miopen::pad_constant_fwd::ProblemDescription>;

struct PadConstantFwd final : PadConstantFwdSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<PadConstantFwd>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pad_constant_fwd::ProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::pad_constant_fwd::ProblemDescription& problem) const override;
};
} // namespace pad_constant_fwd

namespace pad_constant_bwd {
using PadConstantBwdSolver =
    NonTunableSolverBase<ExecutionContext, miopen::pad_constant_bwd::ProblemDescription>;

struct PadConstantBwd final : PadConstantBwdSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<PadConstantBwd>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::pad_constant_bwd::ProblemDescription& problem) const override;

    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::pad_constant_bwd::ProblemDescription& problem) const override;
};
} // namespace pad_constant_bwd
} // namespace solver
} // namespace miopen
