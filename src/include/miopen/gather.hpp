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

#include <miopen/common.hpp>
#include <miopen/export_internals.h>
#include <miopen/handle.hpp>
#include "miopen/object.hpp"
#include <miopen/miopen.h>

#include <ostream>

namespace miopen {

struct Handle;
struct TensorDescriptor;

struct MIOPEN_INTERNALS_EXPORT GatherDescriptor : miopenGatherDescriptor
{
    GatherDescriptor();
    GatherDescriptor(miopenGatherMode_t m, uint32_t dim, uint32_t batch_dims);

    miopenGatherMode_t getMode() const { return mode; }
    uint32_t getDim() const { return dim; }
    uint32_t getBatchDims() const { return batch_dims; }

    void setMode(miopenGatherMode_t m) { mode = m; }
    void setDim(uint32_t d) { dim = d; }
    void setBatchDims(uint32_t b) { batch_dims = b; }

    miopenStatus_t Backward(Handle& handle,
                            const TensorDescriptor& outputGradDesc,
                            ConstData_t outputgrad,
                            const TensorDescriptor& indiceDesc,
                            ConstData_t indices,
                            const TensorDescriptor& paramGradDesc,
                            Data_t paramGrad) const;

    friend std::ostream& operator<<(std::ostream& stream, const GatherDescriptor& x);

private:
    miopenGatherMode_t mode = MIOPEN_GATHER_V2;
    uint32_t dim;
    uint32_t batch_dims;
};

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenGatherDescriptor, miopen::GatherDescriptor);
