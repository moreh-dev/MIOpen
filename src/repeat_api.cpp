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

#include <miopen/repeat.hpp>
#include <miopen/sum.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void
LogCmdRepeat(const miopenTensorDescriptor_t xDesc, const int* sizes, int num_sizes, bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(xDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "repeatfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "repeatfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "repeatbfp16";
        }

        int32_t size = {0};
        miopenGetTensorDescriptorSize(xDesc, &size);
        ss << " -n " << miopen::deref(xDesc).GetLengths()[0];
        if(size == 5)
        {
            ss << " -c " << miopen::deref(xDesc).GetLengths()[1] << " -D "
               << miopen::deref(xDesc).GetLengths()[2] << " -H "
               << miopen::deref(xDesc).GetLengths()[3] << " -W "
               << miopen::deref(xDesc).GetLengths()[4];
        }
        else if(size == 4)
        {
            ss << " -c " << miopen::deref(xDesc).GetLengths()[1] << " -H "
               << miopen::deref(xDesc).GetLengths()[2] << " -W "
               << miopen::deref(xDesc).GetLengths()[3];
        }
        else if(size == 3)
        {
            ss << " -c " << miopen::deref(xDesc).GetLengths()[1] << " -W "
               << miopen::deref(xDesc).GetLengths()[2];
        }
        else if(size == 2)
        {
            ss << " -c " << miopen::deref(xDesc).GetLengths()[1];
        }

        ss << " -sizes ";
        for(int i = 0; i < num_sizes; ++i)
        {
            ss << sizes[i] << " ";
        }

        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenRepeatForward(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t xDesc,
                                              const void* x,
                                              const int* sizes,
                                              const int num_sizes,
                                              const miopenTensorDescriptor_t yDesc,
                                              void* y)
{
    MIOPEN_LOG_FUNCTION(handle, xDesc, x, sizes, num_sizes, yDesc, y);
    LogCmdRepeat(xDesc, sizes, num_sizes, true);
    return miopen::try_([&] {
        miopen::RepeatForward(miopen::deref(handle),
                              miopen::deref(xDesc),
                              DataCast(x),
                              sizes,
                              num_sizes,
                              miopen::deref(yDesc),
                              DataCast(y));
    });
}

extern "C" miopenStatus_t miopenRepeatBackward(miopenHandle_t handle,
                                               const miopenTensorDescriptor_t dyDesc,
                                               const void* dy,
                                               const int* sizes,
                                               const int num_sizes,
                                               const miopenTensorDescriptor_t dxDesc,
                                               void* dx)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, dy, sizes, num_sizes, dxDesc, dx);
    LogCmdRepeat(dyDesc, sizes, num_sizes, false);
    return miopen::try_([&] {
        miopen::RepeatBackward(miopen::deref(handle),
                               miopen::deref(dyDesc),
                               DataCast(dy),
                               sizes,
                               num_sizes,
                               miopen::deref(dxDesc),
                               DataCast(dx));
    });
}
