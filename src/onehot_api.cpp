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

#include "miopen/miopen.h"
#include <__clang_hip_math.h>
#include <miopen/onehot.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdOneHot(const miopenTensorDescriptor_t inDesc)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;

        int32_t size = {0};
        miopenGetTensorDescriptorSize(inDesc, &size);
        ss << " -n " << miopen::deref(inDesc).GetLengths()[0];
        if(size == 5)
        {
            ss << " -c " << miopen::deref(inDesc).GetLengths()[1] << " -D "
               << miopen::deref(inDesc).GetLengths()[2] << " -H "
               << miopen::deref(inDesc).GetLengths()[3] << " -W "
               << miopen::deref(inDesc).GetLengths()[4];
        }
        else if(size == 4)
        {
            ss << " -c " << miopen::deref(inDesc).GetLengths()[1] << " -H "
               << miopen::deref(inDesc).GetLengths()[2] << " -W "
               << miopen::deref(inDesc).GetLengths()[3];
        }
        else if(size == 3)
        {
            ss << " -c " << miopen::deref(inDesc).GetLengths()[1] << " -W "
               << miopen::deref(inDesc).GetLengths()[2];
        }
        else if(size == 2)
        {
            ss << " -c " << miopen::deref(inDesc).GetLengths()[1];
        }

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenOneHot(miopenHandle_t handle,
                                       const miopenTensorDescriptor_t inDesc,
                                       const void* input,
                                       const long inputSize,
                                       const miopenTensorDescriptor_t outDesc,
                                       void* output,
                                       int numClasses)
{
    MIOPEN_LOG_FUNCTION(handle, inDesc, input, inputSize, outDesc, output, numClasses);

    LogCmdOneHot(inDesc);
    return miopen::try_([&] {
        miopen::OneHot(miopen::deref(handle),
                       miopen::deref(inDesc),
                       DataCast(input),
                       inputSize,
                       miopen::deref(outDesc),
                       DataCast(output),
                       numClasses);
    });
}
