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
#include "miopen/tensor.hpp"
#include <miopen/miopen.h>
#ifndef MIOPEN_MARGINRANKINGLOSS_HPP_
#define MIOPEN_MARGINRANKINGLOSS_HPP_

#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

miopenStatus_t MarginRankingLossForward(Handle& handle,
                                        Data_t workspace,
                                        size_t workspaceSizeInBytes,
                                        const TensorDescriptor& input1Desc,
                                        ConstData_t input1,
                                        const TensorDescriptor& input2Desc,
                                        ConstData_t input2,
                                        const TensorDescriptor& targetDesc,
                                        ConstData_t target,
                                        const TensorDescriptor& outputDesc,
                                        Data_t output,
                                        float margin,
                                        float divisor);

miopenStatus_t MarginRankingLossBackward(Handle& handle,
                                         const TensorDescriptor& input1Desc,
                                         ConstData_t input1,
                                         const TensorDescriptor& input2Desc,
                                         ConstData_t input2,
                                         const TensorDescriptor& targetDesc,
                                         ConstData_t target,
                                         const TensorDescriptor& outGradDesc,
                                         Data_t outGrad,
                                         const TensorDescriptor& in1GradDesc,
                                         Data_t in1Grad,
                                         const TensorDescriptor& in2GradDesc,
                                         Data_t in2Grad,
                                         float margin,
                                         float divisor);

miopenStatus_t MarginRankingLossUnreducedForward(Handle& handle,
                                                 const TensorDescriptor& input1Desc,
                                                 ConstData_t input1,
                                                 const TensorDescriptor& input2Desc,
                                                 ConstData_t input2,
                                                 const TensorDescriptor& targetDesc,
                                                 ConstData_t target,
                                                 const TensorDescriptor& outputDesc,
                                                 Data_t output,
                                                 float margin);

miopenStatus_t MarginRankingLossUnreducedBackward(Handle& handle,
                                                  const TensorDescriptor& input1Desc,
                                                  ConstData_t input1,
                                                  const TensorDescriptor& input2Desc,
                                                  ConstData_t input2,
                                                  const TensorDescriptor& targetDesc,
                                                  ConstData_t target,
                                                  const TensorDescriptor& outGradDesc,
                                                  Data_t outGrad,
                                                  const TensorDescriptor& in1GradDesc,
                                                  Data_t in1Grad,
                                                  const TensorDescriptor& in2GradDesc,
                                                  Data_t in2Grad,
                                                  float margin);

} // namespace miopen
#endif // MIOPEN_MARGINRANKINGLOSS_HPP_
