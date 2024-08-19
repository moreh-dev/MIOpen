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

#include <miopen/maskedfill.hpp>

#include <miopen/logger.hpp>
#include <miopen/errors.hpp>
#include <miopen/tensor.hpp>
#include <miopen/handle.hpp>

static void LogCmdMaskedFill(miopenTensorDescriptor_t const inputDesc, bool const is_fwd)
{
    if(miopen ::IsLoggingCmd())
    {
        std ::stringstream ss;
        auto const dtype = miopen ::deref(inputDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "maskedfillfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "maskedfillfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "maskedfillfp16";
        }
        int32_t size{0};
        miopenGetTensorDescriptorSize(inputDesc, &size);
        ss << " -n " << miopen ::deref(inputDesc).GetLengths()[0] << " -c "
           << miopen ::deref(inputDesc).GetLengths()[1];
        if(size == 5)
        {
            ss << " -D " << miopen ::deref(inputDesc).GetLengths()[2] << " -H "
               << miopen ::deref(inputDesc).GetLengths()[3] << " -W "
               << miopen ::deref(inputDesc).GetLengths()[4];
        }
        else if(size == 4)
        {
            ss << " -H " << miopen ::deref(inputDesc).GetLengths()[2] << " -W "
               << miopen ::deref(inputDesc).GetLengths()[3];
        }
        else if(size == 3)
        {
            ss << " -W " << miopen ::deref(inputDesc).GetLengths()[2];
        }
        ss << " -F " << (is_fwd ? "1" : "2");
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenMaskedFillForward(const miopenHandle_t handle,

                                                  const miopenTensorDescriptor_t inputDesc,
                                                  const void* input,
                                                  const miopenTensorDescriptor_t outputDesc,
                                                  void* output,

                                                  const miopenTensorDescriptor_t maskDesc,
                                                  const void* mask,

                                                  const float value)
{
    MIOPEN_LOG_FUNCTION(handle, inputDesc, input, outputDesc, output, maskDesc, mask, value);
    LogCmdMaskedFill(inputDesc, true);
    return miopen ::try_([&] {
        miopen ::MaskedFillForward(miopen ::deref(handle),
                                   miopen ::deref(inputDesc),
                                   DataCast(input),
                                   miopen ::deref(outputDesc),
                                   DataCast(output),
                                   miopen ::deref(maskDesc),
                                   DataCast(mask),
                                   value);
    });
}

extern "C" miopenStatus_t
miopenMaskedFillBackward(const miopenHandle_t handle,

                         const miopenTensorDescriptor_t outputGradientDesc,
                         const void* outputGradient,
                         const miopenTensorDescriptor_t inputGradientDesc,
                         void* inputGradient,

                         const miopenTensorDescriptor_t maskDesc,
                         const void* mask,

                         const float value)
{
    MIOPEN_LOG_FUNCTION(handle,
                        outputGradientDesc,
                        outputGradient,
                        inputGradientDesc,
                        inputGradient,
                        maskDesc,
                        mask,
                        value);
    LogCmdMaskedFill(outputGradientDesc, false);
    return miopen ::try_([&] {
        miopen ::MaskedFillBackward(miopen ::deref(handle),
                                    miopen ::deref(outputGradientDesc),
                                    DataCast(outputGradient),
                                    miopen ::deref(inputGradientDesc),
                                    DataCast(inputGradient),
                                    miopen ::deref(maskDesc),
                                    DataCast(mask),
                                    value);
    });
}
