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

#include <miopen/solver.hpp>
#include <miopen/maskedfill/problem_description.hpp>

namespace miopen::solver::maskedfill {

constexpr auto float32contiguousfwdminimumnonimprovementnumel  = 524288;
constexpr auto float16contiguousfwdminimumnonimprovementnumel  = 4194304;
constexpr auto bfloat16contiguousfwdminimumnonimprovementnumel = 4194304;
constexpr auto noncontiguousfwdminimumnonimprovementnumel      = 1089000;
using MaskedFillSolver =
    NonTunableSolverBase<ExecutionContext, miopen::maskedfill::ProblemDescription>;
struct MaskedFill : MaskedFillSolver
{
    std::string const& SolverDbId() const override { return GetSolverDbId<MaskedFill>(); }
    bool IsApplicable(ExecutionContext const& context,
                      miopen::maskedfill::ProblemDescription const& problem) const override;
    ConvSolution GetSolution(ExecutionContext const& context,
                             miopen::maskedfill::ProblemDescription const& problem) const override;
};

} // namespace miopen::solver::maskedfill
