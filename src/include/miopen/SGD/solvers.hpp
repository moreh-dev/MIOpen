#pragma once

#include <miopen/solver.hpp>
#include <miopen/SGD/problem_description.hpp>

#include <utility>

namespace miopen {
namespace solver {
namespace SGD {

using SGDSolver = NonTunableSolverBase<ExecutionContext, miopen::SGD::ProblemDescription>;

struct SGDForward final : SGDSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<SGDForward>();}

    bool IsApplicable(const ExecutionContext& constext,
                      const miopen::SGD::ProblemDescription& problem) const override;
    ConvSolution GetSolution(const ExecutionContext& context,
                             const miopen::SGD::ProblemDescription& problem) const override;
    std::size_t GetWorkspaceSize(const ExecutionContext& context,
                                 const miopen::SGD::ProblemDescription& problem) const override
    {
        return 0;
    }
    bool MayNeedWorkspace() const override { return false; }
};

} // namespace SGD
} // namespace solver
} // namespace miopen
