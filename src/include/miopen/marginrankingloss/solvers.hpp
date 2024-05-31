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

#include "miopen/conv_solution.hpp"
#include "miopen/execution_context.hpp"
#include "miopen/mlo_internal.hpp"
#include "miopen/solver.hpp"
#include "miopen/marginrankingloss/problem_description.hpp"
#include "miopen/kernel_build_params.hpp"
#include "miopen/kernel_info.hpp"

#include <utility>
#include <vector>

namespace miopen {

namespace solver {

const auto make_hip_kernel = [](std::vector<size_t> local_size,
                                std::vector<size_t> grid_size,
                                std::string kernel_file,
                                std::string kernel_name,
                                KernelBuildParameters build_params) {

    while(local_size.size() < 3)
        local_size.push_back(1);
    while(grid_size.size() < 3)
        grid_size.push_back(1);
    for(int i = 0; i < local_size.size(); ++i)
        grid_size[i] = AlignUp(grid_size[i], local_size[i]);
    return KernelInfo{build_params.GenerateFor(kbp::HIP{}), local_size, grid_size, kernel_file, kernel_name};
};

namespace marginrankingloss {

using MarginRankingLossForwardSolver = NonTunableSolverBase<ExecutionContext, miopen::marginrankingloss::ProblemDescriptionForward>;

using MarginRankingLossBackwardSolver = NonTunableSolverBase<ExecutionContext, miopen::marginrankingloss::ProblemDescriptionBackward>;

struct MarginRankingLossForward : MarginRankingLossForwardSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<MarginRankingLossForwardSolver>();
    }

    bool IsApplicable(const ExecutionContext& context, const miopen::marginrankingloss::ProblemDescriptionForward& problem) const override;

    ConvSolution GetSolution(const ExecutionContext& context, const miopen::marginrankingloss::ProblemDescriptionForward& problem) const override;

    std::size_t GetWorkspaceSize(const ExecutionContext& context, const miopen::marginrankingloss::ProblemDescriptionForward& problem) const override;

    bool MayNeedWorkspace() const override;
};

struct MarginRankingLossBackward : MarginRankingLossBackwardSolver
{
    const std::string& SolverDbId() const override
    {
        return GetSolverDbId<MarginRankingLossBackwardSolver>();
    }

    bool IsApplicable(const ExecutionContext& context, const miopen::marginrankingloss::ProblemDescriptionBackward& problem) const override;

    ConvSolution GetSolution(const ExecutionContext& context, const miopen::marginrankingloss::ProblemDescriptionBackward& problem) const override;

    std::size_t GetWorkspaceSize([[maybe_unused]] const ExecutionContext& context,
                                 [[maybe_unused]] const miopen::marginrankingloss::ProblemDescriptionBackward& problem) const override
    {
        return 0;
    }
    
    bool MayNeedWorkspace() const override { return false; }
};

} // namespace marginrankingloss

} // namespace solver

} // namespace miopen
