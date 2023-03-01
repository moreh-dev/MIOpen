/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef MIOPEN_GUARD_MIOPEN_EXT_H_
#define MIOPEN_GUARD_MIOPEN_EXT_H_

/* Put modnn extension APIs here. */
/* If used, should be included after miopen_internal.h. */

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wextern-c-compat"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*!
  @brief Check if the convolution forward will use pre-compiled kernel.
*/
MIOPEN_EXPORT miopenStatus_t
miopenCheckConvFwdUsePreCompiledKernel(miopenHandle_t handle,
                                       const miopenTensorDescriptor_t xDesc,
                                       const void* x,
                                       const miopenTensorDescriptor_t wDesc,
                                       const void* w,
                                       const miopenConvolutionDescriptor_t convDesc,
                                       const miopenTensorDescriptor_t yDesc,
                                       void* y,
                                       bool* kernelBuildHappen);

/*!
  @brief Check if the convolution backward data will use pre-compiled kernel.
*/
MIOPEN_EXPORT miopenStatus_t
miopenCheckConvolutionBackwardDataUsePreCompiledKernel(miopenHandle_t handle,
                                                       const miopenTensorDescriptor_t dyDesc,
                                                       const void* dy,
                                                       const miopenTensorDescriptor_t wDesc,
                                                       const void* w,
                                                       const miopenConvolutionDescriptor_t convDesc,
                                                       const miopenTensorDescriptor_t dxDesc,
                                                       void* dx,
                                                       bool* kernelBuildHappen);

/*!
  @brief Check if the convolution backward weights will use pre-compiled kernel.
*/
MIOPEN_EXPORT miopenStatus_t miopenCheckConvolutionBackwardWeightsUsePreCompiledKernel(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t dyDesc,
    const void* dy,
    const miopenTensorDescriptor_t xDesc,
    const void* x,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dwDesc,
    void* dw,
    bool* kernelBuildHappen);

#ifdef __cplusplus
}
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif // MIOPEN_GUARD_MIOPEN_EXT_H_
