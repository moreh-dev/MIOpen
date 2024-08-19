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

#include <miopen/logsumexp.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdLogsumexp(const miopenTensorDescriptor_t inputDesc,
                            const int* dims,
                            const int num_dims,
                            bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(inputDesc).GetType();
        if(dtype == miopenFloat)
        {
            ss << "logsumexpfp32";
        }
        else if(dtype == miopenHalf)
        {
            ss << "logsumexpfp16";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "logsumexpf16";
        }

        int32_t size = {0};
        miopenGetTensorDescriptorSize(inputDesc, &size);
        ss << " -n " << miopen::deref(inputDesc).GetLengths()[0];
        if(size == 5)
        {
            ss << " - c " << miopen::deref(inputDesc).GetLengths()[1] << " -D "
               << miopen::deref(inputDesc).GetLengths()[2] << " -H "
               << miopen::deref(inputDesc).GetLengths()[3] << " -W "
               << miopen::deref(inputDesc).GetLengths()[4];
        }
        else if(size == 4)
        {
            ss << " - c " << miopen::deref(inputDesc).GetLengths()[1] << " -D "
               << miopen::deref(inputDesc).GetLengths()[2] << " -H "
               << miopen::deref(inputDesc).GetLengths()[3];
        }
        else if(size == 3)
        {
            ss << " - c " << miopen::deref(inputDesc).GetLengths()[1] << " -D "
               << miopen::deref(inputDesc).GetLengths()[2];
        }
        else if(size == 2)
        {
            ss << " - c " << miopen::deref(inputDesc).GetLengths()[1];
        }

        ss << " -dims ";
        for(int i = 0; i < num_dims; i++)
        {
            ss << dims[i] << " ";
        }

        ss << " -F " << ((is_fwd) ? "true" : "false");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
};

extern "C" miopenStatus_t miopenLogsumexpForward(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t inputDesc,
                                                 const void* input,
                                                 const miopenTensorDescriptor_t outputDesc,
                                                 void* output,
                                                 const int* dims,
                                                 const int num_dims)
{
    MIOPEN_LOG_FUNCTION(handle, inputDesc, input, outputDesc, output, dims, num_dims);

    LogCmdLogsumexp(inputDesc, dims, num_dims, true);

    return miopen::try_([&] {
        miopen::LogsumexpForward(miopen::deref(handle),
                                 miopen::deref(inputDesc),
                                 DataCast(input),
                                 miopen::deref(outputDesc),
                                 DataCast(output),
                                 dims,
                                 num_dims);
    });
}

extern "C" miopenStatus_t miopenLogsumexpBackward(miopenHandle_t handle,
                                                  const miopenTensorDescriptor_t inputDesc,
                                                  const void* input,
                                                  const miopenTensorDescriptor_t inputGradDesc,
                                                  void* inputGrad,
                                                  const miopenTensorDescriptor_t outputDesc,
                                                  const void* output,
                                                  const miopenTensorDescriptor_t outputGradDesc,
                                                  const void* outputGrad,
                                                  const int* dims,
                                                  const int num_dims)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        inputGradDesc,
                        inputGrad,
                        outputDesc,
                        output,
                        outputGradDesc,
                        outputGrad,
                        dims,
                        num_dims);

    LogCmdLogsumexp(inputDesc, dims, num_dims, false);

    return miopen::try_([&] {
        miopen::LogsumexpBackward(miopen::deref(handle),
                                  miopen::deref(inputDesc),
                                  DataCast(input),
                                  miopen::deref(inputGradDesc),
                                  DataCast(inputGrad),
                                  miopen::deref(outputDesc),
                                  DataCast(output),
                                  miopen::deref(outputGradDesc),
                                  DataCast(outputGrad),
                                  dims,
                                  num_dims);
    });
}
