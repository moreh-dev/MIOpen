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
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/image_transform.hpp>

extern "C" miopenStatus_t miopenImageAdjustHue(miopenHandle_t handle,
                                               miopenTensorDescriptor_t inputTensorDesc,
                                               miopenTensorDescriptor_t outputTensorDesc,
                                               const void* input_buf,
                                               void* output_buf,
                                               float hue)
{
    MIOPEN_LOG_FUNCTION(handle, inputTensorDesc, outputTensorDesc, input_buf, output_buf, hue);

    return miopen::try_([&] {
        miopen::miopenImageAdjustHue(miopen::deref(handle),
                                     miopen::deref(inputTensorDesc),
                                     miopen::deref(outputTensorDesc),
                                     DataCast(input_buf),
                                     DataCast(output_buf),
                                     hue);
    });
}

extern "C" miopenStatus_t miopenImageAdjustBrightness(miopenHandle_t handle,
                                                      miopenTensorDescriptor_t inputTensorDesc,
                                                      miopenTensorDescriptor_t outputTensorDesc,
                                                      const void* input_buf,
                                                      void* output_buf,
                                                      float brightness_factor)
{
    MIOPEN_LOG_FUNCTION(
        handle, inputTensorDesc, outputTensorDesc, input_buf, output_buf, brightness_factor);

    return miopen::try_([&] {
        miopen::miopenImageAdjustBrightness(miopen::deref(handle),
                                            miopen::deref(inputTensorDesc),
                                            miopen::deref(outputTensorDesc),
                                            DataCast(input_buf),
                                            DataCast(output_buf),
                                            brightness_factor);
    });
}