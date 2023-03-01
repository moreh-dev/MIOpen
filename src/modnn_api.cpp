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
#include <miopen/miopen.h>
#include <miopen/miopen_internal.h>
#include <miopen/miopen_ext.h>

#include <miopen/convolution.hpp>
#include <miopen/errors.hpp>
#include <miopen/find_controls.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>
#include <algorithm>

extern "C" miopenStatus_t
miopenCheckConvolutionForwardUsePreCompiledKernel(miopenHandle_t handle,
                                                  const miopenTensorDescriptor_t xDesc,
                                                  const void* x,
                                                  const miopenTensorDescriptor_t wDesc,
                                                  const void* w,
                                                  const miopenConvolutionDescriptor_t convDesc,
                                                  const miopenTensorDescriptor_t yDesc,
                                                  void* y,
                                                  bool* kernelBuildHappen)
{
    MIOPEN_LOG_FUNCTION(handle, xDesc, x, wDesc, w, convDesc, yDesc, y, kernelBuildHappen);

    if(miopen::deref(convDesc).mode == miopenTranspose)
        return miopen::try_([&] {
            miopen::deref(convDesc).CheckConvBwdDataUsePreCompiledKernel(miopen::deref(handle),
                                                                         miopen::deref(xDesc),
                                                                         DataCast(x),
                                                                         miopen::deref(wDesc),
                                                                         DataCast(w),
                                                                         miopen::deref(yDesc),
                                                                         DataCast(y),
                                                                         kernelBuildHappen);
        });

    return miopen::try_([&] {
        miopen::deref(convDesc).CheckConvFwdUsePreCompiledKernel(miopen::deref(handle),
                                                                 miopen::deref(xDesc),
                                                                 DataCast(x),
                                                                 miopen::deref(wDesc),
                                                                 DataCast(w),
                                                                 miopen::deref(yDesc),
                                                                 DataCast(y),
                                                                 kernelBuildHappen);
    });
}

extern "C" miopenStatus_t
miopenCheckConvolutionBackwardDataUsePreCompiledKernel(miopenHandle_t handle,
                                                       const miopenTensorDescriptor_t dyDesc,
                                                       const void* dy,
                                                       const miopenTensorDescriptor_t wDesc,
                                                       const void* w,
                                                       const miopenConvolutionDescriptor_t convDesc,
                                                       const miopenTensorDescriptor_t dxDesc,
                                                       void* dx,
                                                       bool* kernelBuildHappen)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, dy, wDesc, w, convDesc, dxDesc, dx, kernelBuildHappen);

    if(miopen::deref(convDesc).mode == miopenTranspose)
        return miopen::try_([&] {
            miopen::deref(convDesc).CheckConvFwdUsePreCompiledKernel(miopen::deref(handle),
                                                                     miopen::deref(dyDesc),
                                                                     DataCast(dy),
                                                                     miopen::deref(wDesc),
                                                                     DataCast(w),
                                                                     miopen::deref(dxDesc),
                                                                     DataCast(dx),
                                                                     kernelBuildHappen);
        });

    return miopen::try_([&] {
        miopen::deref(convDesc).CheckConvBwdDataUsePreCompiledKernel(miopen::deref(handle),
                                                                     miopen::deref(dyDesc),
                                                                     DataCast(dy),
                                                                     miopen::deref(wDesc),
                                                                     DataCast(w),
                                                                     miopen::deref(dxDesc),
                                                                     DataCast(dx),
                                                                     kernelBuildHappen);
    });
}

extern "C" miopenStatus_t miopenCheckConvolutionBackwardWeightsUsePreCompiledKernel(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t dyDesc,
    const void* dy,
    const miopenTensorDescriptor_t xDesc,
    const void* x,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dwDesc,
    void* dw,
    bool* kernelBuildHappen)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, dy, xDesc, x, convDesc, dwDesc, dw, kernelBuildHappen);

    if(miopen::deref(convDesc).mode == miopenTranspose)
        return miopen::try_([&] {
            miopen::deref(convDesc).CheckConvBwdWeightsUsePreCompiledKernel(miopen::deref(handle),
                                                                            miopen::deref(xDesc),
                                                                            DataCast(x),
                                                                            miopen::deref(dyDesc),
                                                                            DataCast(dy),
                                                                            miopen::deref(dwDesc),
                                                                            DataCast(dw),
                                                                            kernelBuildHappen);
        });

    return miopen::try_([&] {
        miopen::deref(convDesc).CheckConvBwdWeightsUsePreCompiledKernel(miopen::deref(handle),
                                                                        miopen::deref(dyDesc),
                                                                        DataCast(dy),
                                                                        miopen::deref(xDesc),
                                                                        DataCast(x),
                                                                        miopen::deref(dwDesc),
                                                                        DataCast(dw),
                                                                        kernelBuildHappen);
    });
}
