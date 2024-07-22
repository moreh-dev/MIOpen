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

#include <miopen/indexselect.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdIndexSelect(const miopenTensorDescriptor_t Desc)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(Desc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "indexselectfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "indexselectfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "indexselectbfp16";
        }
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenIndexSelectForward(miopenHandle_t handle,
                                                   const miopenTensorDescriptor_t xDesc,
                                                   const void* x,
                                                   const miopenTensorDescriptor_t indicesDesc,
                                                   const void* indices,
                                                   const miopenTensorDescriptor_t yDesc,
                                                   void* y,
                                                   size_t dim)
{
    MIOPEN_LOG_FUNCTION(handle, xDesc, x, indicesDesc, indices, yDesc, y);

    LogCmdIndexSelect(xDesc);

    return miopen::try_([&] {
        miopen::IndexSelectForward(miopen::deref(handle),
                                   miopen::deref(xDesc),
                                   DataCast(x),
                                   miopen::deref(indicesDesc),
                                   DataCast(indices),
                                   miopen::deref(yDesc),
                                   DataCast(y),
                                   dim);
    });
}

extern "C" miopenStatus_t miopenIndexSelectBackward(miopenHandle_t handle,
                                                    const miopenTensorDescriptor_t xGradDesc,
                                                    void* xGrad,
                                                    const miopenTensorDescriptor_t indicesDesc,
                                                    const void* indices,
                                                    const miopenTensorDescriptor_t yGradDesc,
                                                    const void* yGrad,
                                                    size_t dim)
{
    MIOPEN_LOG_FUNCTION(handle, xGradDesc, xGrad, indicesDesc, indices, yGradDesc, yGrad);

    LogCmdIndexSelect(xGradDesc);

    return miopen::try_([&] {
        miopen::IndexSelectBackward(miopen::deref(handle),
                                    miopen::deref(xGradDesc),
                                    DataCast(xGrad),
                                    miopen::deref(indicesDesc),
                                    DataCast(indices),
                                    miopen::deref(yGradDesc),
                                    DataCast(yGrad),
                                    dim);
    });
}
