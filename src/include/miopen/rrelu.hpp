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

namespace miopen {

struct Handle;
struct TensorDescriptor;

size_t GetRReLUStatesSize(Handle& handle);

miopenStatus_t
RReLUStatesInit(Handle& handle, Data_t states, size_t stateSizeInBytes, uint64_t seed);

size_t GetRReLUForwardWorkspaceSize(Handle& handle,
                                    const TensorDescriptor& inputDesc,
                                    const TensorDescriptor& outputDesc);

miopenStatus_t RReLUForward(Handle& handle,
                            Data_t workspace,
                            size_t workspaceSizeInBytes,
                            ConstData_t states,
                            size_t stateSizeInBytes,
                            const TensorDescriptor& inputDesc,
                            ConstData_t input,
                            const TensorDescriptor& outputDesc,
                            Data_t output,
                            const TensorDescriptor& noiseDesc,
                            Data_t noise,
                            float lower,
                            float upper);

miopenStatus_t RReLUBackward(Handle& handle,
                             const TensorDescriptor& noiseDesc,
                             ConstData_t noise,
                             const TensorDescriptor& doutputDesc,
                             ConstData_t doutput,
                             const TensorDescriptor& dinputDesc,
                             Data_t dinput);

} // namespace miopen
