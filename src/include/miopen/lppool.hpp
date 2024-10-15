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

namespace lppool {

MIOPEN_INTERNALS_EXPORT miopenStatus_t LPPoolForward(Handle& handle,
                                                     const TensorDescriptor& inputDesc,
                                                     ConstData_t input,
                                                     const TensorDescriptor& outputDesc,
                                                     Data_t output,
                                                     int64_t KD,
                                                     int64_t KH,
                                                     int64_t SD,
                                                     int64_t SH,
                                                     float norm_type);

MIOPEN_INTERNALS_EXPORT miopenStatus_t LPPoolBackward(Handle& handle,
                                                      const TensorDescriptor& inputDesc,
                                                      ConstData_t input,
                                                      const TensorDescriptor& outputDesc,
                                                      ConstData_t output,
                                                      const TensorDescriptor& outputGradDesc,
                                                      ConstData_t output_grad,
                                                      const TensorDescriptor& inputGradDesc,
                                                      Data_t input_grad,
                                                      int64_t KD,
                                                      int64_t KH,
                                                      int64_t SD,
                                                      int64_t SH,
                                                      float norm_type);
} // namespace lppool

} // namespace miopen