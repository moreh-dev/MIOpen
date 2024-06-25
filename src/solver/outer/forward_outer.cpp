/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/outer/solvers.hpp>

#include <miopen/outer/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/outer.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>

namespace miopen {

namespace solver {

namespace outer {

static bool IsImprovementOverROCm(const miopen::outer::ProblemDescription& problem)
{
    std::cout << "outer IsImprovementOverROCm is called" << std::endl;
    return true;
}

bool OuterForward::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                              const miopen::outer::ProblemDescription& problem) const
{
    std::cout << "outer IsApplicable is called" << std::endl;
    if(!problem.IsAllPacked())
        return false;
    if(!IsImprovementOverROCm(problem))
        return false;
    return true;
}

ConvSolution OuterForward::GetSolution(const ExecutionContext& context,
                                     const miopen::outer::ProblemDescription& problem) const
{
    std::cout << "outer GetSolution is called" << std::endl;
    auto result = ConvSolution{miopenStatusSuccess};

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {

        };
    };

    return result;
}

std::size_t OuterForward::GetWorkspaceSize(const ExecutionContext& context,
                                         const miopen::outer::ProblemDescription& problem) const
{
    std::cout << "outer GetWorkspaceSize is called" << std::endl;
    return 0;
}

} // namespace outer

} // namespace solver

} // namespace miopen
