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

#include <cstddef>
#include <cstdint>

#include <miopen/errors.hpp>
#include "miopen/gather.hpp"
#include <miopen/tensor.hpp>
#include <miopen/problem_description_base.hpp>
#include "../src/kernels/tensor_view.hpp"

namespace miopen {

struct NetworkConfig;

namespace gather {

template <int M>
tensor_view_t<M> reshape(const TensorDescriptor& tensorDes, const std::vector<size_t>& shape);

extern template tensor_view_t<3> reshape<3>(const TensorDescriptor& tensorDes,
                                            const std::vector<size_t>& shape);

extern template tensor_view_t<4> reshape<4>(const TensorDescriptor& tensorDes,
                                            const std::vector<size_t>& shape);

struct BwdProblemDescription : ProblemDescriptionBase
{
    // Backward constructor
    BwdProblemDescription(const GatherDescriptor& gatherDesc_,
                          const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& indicesDesc_,
                          const TensorDescriptor& paramGradDesc_)
        : gatherDesc(gatherDesc_),
          outputGradDesc(outputGradDesc_),
          indicesDesc(indicesDesc_),
          paramGradDesc(paramGradDesc_)
    {
        auto dim        = gatherDesc.getDim();
        auto batch_dims = gatherDesc.getBatchDims();

        if(indicesDesc.GetType() != miopenInt32 && indicesDesc.GetType() != miopenInt64)
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "GatherV2::BwdProblemDescription: Indices tensor must be int32 or int64 tensor.");
        }

        if(!IsSameType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GatherV2::BwdProblemDescription: Output grad and param grad tensors must "
                         "be same type.");
        }

        if(!IsAllContiguous())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GatherV2::BwdProblemDescription: All tensors must be contiguous.");
        }

        if(paramGradDesc.GetNumDims() < dim + 1)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GatherV2::BwdProblemDescription: Param grad tensor dimension must be "
                         "larger than dim value.");
        }

        if(batch_dims > 0)
        {
            if(batch_dims > indicesDesc.GetNumDims())
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "GatherV2::BwdProblemDescription: Batch dims must be less than or "
                             "equal to indices tensor dimension.");
            }

            if(batch_dims >= paramGradDesc.GetNumDims())
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "GatherV2::BwdProblemDescription: Batch dims must be less than or "
                             "equal to param grad tensor dimension.");
            }

            if(dim < batch_dims)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "GatherV2::BwdProblemDescription: Batch dims must be less than or "
                             "equal to dim value.");
            }

            for(uint32_t i = 0; i < batch_dims; i++)
            {
                if(paramGradDesc.GetLengths()[i] != indicesDesc.GetLengths()[i])
                {
                    MIOPEN_THROW(miopenStatusBadParm,
                                 "GatherV2::BwdProblemDescription: Param grad tensor dim should be "
                                 "equal to indices dim.");
                }
            }
        }
    }

    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetIndicesDesc() const { return indicesDesc; }
    const TensorDescriptor& GetParamGradDesc() const { return paramGradDesc; }
    const GatherDescriptor& GetGatherDesc() const { return gatherDesc; }

    bool IsSameType() const
    {
        if(outputGradDesc.GetType() != paramGradDesc.GetType())
        {
            return false;
        }

        return true;
    }

    bool IsAllContiguous() const
    {
        if(outputGradDesc.IsContiguous() && indicesDesc.IsContiguous() &&
           paramGradDesc.IsContiguous())
        {
            return true;
        }
        return false;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    GatherDescriptor gatherDesc;
    TensorDescriptor outputGradDesc;
    TensorDescriptor indicesDesc;
    TensorDescriptor paramGradDesc;
};

} // namespace gather

} // namespace miopen
