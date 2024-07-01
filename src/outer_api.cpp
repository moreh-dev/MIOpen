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

#include <miopen/outer.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdOuter(
                    const miopenTensorDescriptor_t x1Desc,
                    const miopenTensorDescriptor_t x2Desc,
                    bool is_fwd
)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(x1Desc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "sumfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "sumfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "sumbfp16";
        }

        ss 
        << " -N " << miopen::deref(x1Desc).GetLengths()[1] 
        << " -M " << miopen::deref(x2Desc).GetLengths()[2];

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenOuterForward(miopenHandle_t handle,
                                           const miopenTensorDescriptor_t x1Desc,
                                           const void* x1,
                                           const miopenTensorDescriptor_t x2Desc,
                                           const void* x2,
                                           const miopenTensorDescriptor_t yDesc,
                                           void* y)
{
    MIOPEN_LOG_FUNCTION(
        handle, x1Desc, x1, x2Desc, x2, yDesc, yDesc, y);

    LogCmdOuter(x1Desc, x2Desc, true);

    std::cout << "miopenOuterForward is called" << std::endl;

    
    return miopen::try_([&] {
        miopen::OuterForward(miopen::deref(handle),
                           miopen::deref(x1Desc),
                           DataCast(x1),
                           miopen::deref(x2Desc),
                           DataCast(x2),
                           miopen::deref(yDesc),
                           DataCast(y)
                        );
    });
}

extern "C" miopenStatus_t miopenOuterBackward(miopenHandle_t handle,
                                                const miopenTensorDescriptor_t x1Desc,
                                                const void* x1,
                                                const miopenTensorDescriptor_t x2Desc,
                                                const void* x2,
                                                const miopenTensorDescriptor_t yGradDesc,
                                                const void* yGrad,
                                                const miopenTensorDescriptor_t x1GradDesc,
                                                void* x1Grad,
                                                const miopenTensorDescriptor_t x2GradDesc,
                                                void* x2Grad)
{
    MIOPEN_LOG_FUNCTION(
        handle, x1Desc, x1, x2Desc, x2, yGradDesc, yGrad, 
        x1GradDesc, x1Grad, x2GradDesc, x2Grad);

    LogCmdOuter(x1Desc, x2Desc, true);

    std::cout << "miopenOuterBackward is called" << std::endl;
    
    return miopen::try_([&] {
        miopen::OuterBackward(miopen::deref(handle),
                           miopen::deref(x1Desc),
                           DataCast(x1),
                           miopen::deref(x2Desc),
                           DataCast(x2),
                           miopen::deref(yGradDesc),
                           DataCast(yGrad),
                           miopen::deref(x1GradDesc),
                           DataCast(x1Grad),
                           miopen::deref(x2GradDesc),
                           DataCast(x2Grad)
                        );
    });
}
