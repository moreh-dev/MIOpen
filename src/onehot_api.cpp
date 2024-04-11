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

#include <miopen/onehot.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

extern "C" miopenStatus_t miopenGetOneHostWorkspaceSize(miopenHandle_t handle,
                                                       const miopenTensorDescriptor_t inDesc,
                                                       const long inputSize,
                                                       const miopenTensorDescriptor_t outDesc,
                                                       const long numClasses,
                                                       size_t* sizeInBytes)
{

    MIOPEN_LOG_FUNCTION(handle, inDesc, inputSize, outDesc, numClasses);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) = miopen::GetOneHotWorkspaceSize(
            miopen::deref(handle), miopen::deref(inDesc), inputSize, miopen::deref(outDesc), numClasses);
    });
};

extern "C" miopenStatus_t miopenOneHotForward(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t inDesc,
                                              const void* input, const long inputSize, 
                                              const miopenTensorDescriptor_t outDesc,
                                              void* output, const long numClasses)
{
    MIOPEN_LOG_FUNCTION(handle, inDesc, input, inputSize, outDesc, output, numClasses);

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
