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

#include <miopen/hinge_embedding_loss.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
{
    os << '{';
    for(int i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

static void LogCmdHingeEmbeddingLoss(const miopenTensorDescriptor_t inputDesc,
                                     const miopenTensorDescriptor_t targetDesc,
                                     bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(inputDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "hingeEmbeddingLossfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "hingeEmbeddingLossfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "hingeEmbeddingLossbfp16";
        }

        MIOPEN_LOG_FUNCTION(inputDesc, targetDesc);
        ss << " -n " << miopen::deref(inputDesc).GetLengths()[0];
        ss << " -T " << miopen::deref(inputDesc).GetLengths();
        ss << " -Si " << miopen::deref(inputDesc).GetStrides();
        ss << " -St " << miopen::deref(targetDesc).GetStrides();
        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t
miopenGetHingeEmbeddingLossForwardWorkspaceSize(miopenHandle_t handle,
                                                const miopenTensorDescriptor_t inputDesc,
                                                const miopenTensorDescriptor_t targetDesc,
                                                const miopenTensorDescriptor_t outputDesc,
                                                size_t* sizeInBytes)
{

    MIOPEN_LOG_FUNCTION(handle, inputDesc, targetDesc, outputDesc, sizeInBytes);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::GetHingeEmbeddingLossForwardWorkspaceSize(miopen::deref(handle),
                                                              miopen::deref(inputDesc),
                                                              miopen::deref(targetDesc),
                                                              miopen::deref(outputDesc));
    });
}

extern "C" miopenStatus_t miopenHingeEmbeddingLossForward(miopenHandle_t handle,
                                                          void* workspace,
                                                          size_t workspaceSizeInBytes,
                                                          const miopenTensorDescriptor_t inputDesc,
                                                          const void* input,
                                                          const miopenTensorDescriptor_t targetDesc,
                                                          const void* target,
                                                          const miopenTensorDescriptor_t outputDesc,
                                                          void* output,
                                                          const float margin,
                                                          const miopenLossReductionMode_t reduction)
{
    MIOPEN_LOG_FUNCTION(handle,
                        workspace,
                        workspaceSizeInBytes,
                        inputDesc,
                        input,
                        targetDesc,
                        target,
                        outputDesc,
                        output,
                        margin,
                        reduction);

    LogCmdHingeEmbeddingLoss(inputDesc, targetDesc, true);
    return miopen::try_([&] {
        miopen::HingeEmbeddingLossForward(miopen::deref(handle),
                                          DataCast(workspace),
                                          workspaceSizeInBytes,
                                          miopen::deref(inputDesc),
                                          DataCast(input),
                                          miopen::deref(targetDesc),
                                          DataCast(target),
                                          miopen::deref(outputDesc),
                                          DataCast(output),
                                          margin,
                                          reduction);
    });
}

extern "C" miopenStatus_t miopenHingeEmbeddingLossBackward(miopenHandle_t handle,
                                                           miopenTensorDescriptor_t inputDesc,
                                                           const void* input,
                                                           miopenTensorDescriptor_t targetDesc,
                                                           const void* target,
                                                           miopenTensorDescriptor_t doutputDesc,
                                                           const void* doutput,
                                                           miopenTensorDescriptor_t dinputDesc,
                                                           void* dinput,
                                                           float margin,
                                                           float divisor)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        targetDesc,
                        target,
                        doutputDesc,
                        doutput,
                        dinputDesc,
                        dinput,
                        margin,
                        divisor);

    LogCmdHingeEmbeddingLoss(inputDesc, targetDesc, false);
    return miopen::try_([&] {
        miopen::HingeEmbeddingLossBackward(miopen::deref(handle),
                                           miopen::deref(inputDesc),
                                           DataCast(input),
                                           miopen::deref(targetDesc),
                                           DataCast(target),
                                           miopen::deref(doutputDesc),
                                           DataCast(doutput),
                                           miopen::deref(dinputDesc),
                                           DataCast(dinput),
                                           margin,
                                           divisor);
    });
}

extern "C" miopenStatus_t
miopenHingeEmbeddingLossUnreducedForward(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t inputDesc,
                                         const void* input,
                                         const miopenTensorDescriptor_t targetDesc,
                                         const void* target,
                                         const miopenTensorDescriptor_t outputDesc,
                                         void* output,
                                         const float margin)
{
    MIOPEN_LOG_FUNCTION(handle, inputDesc, input, targetDesc, target, outputDesc, output, margin);

    LogCmdHingeEmbeddingLoss(inputDesc, targetDesc, true);
    return miopen::try_([&] {
        miopen::HingeEmbeddingLossUnreducedForward(miopen::deref(handle),
                                                   miopen::deref(inputDesc),
                                                   DataCast(input),
                                                   miopen::deref(targetDesc),
                                                   DataCast(target),
                                                   miopen::deref(outputDesc),
                                                   DataCast(output),
                                                   margin);
    });
}

extern "C" miopenStatus_t
miopenHingeEmbeddingLossUnreducedBackward(miopenHandle_t handle,
                                          miopenTensorDescriptor_t inputDesc,
                                          const void* input,
                                          miopenTensorDescriptor_t targetDesc,
                                          const void* target,
                                          miopenTensorDescriptor_t doutputDesc,
                                          const void* doutput,
                                          miopenTensorDescriptor_t dinputDesc,
                                          void* dinput,
                                          float margin)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        targetDesc,
                        target,
                        doutputDesc,
                        doutput,
                        dinputDesc,
                        dinput,
                        margin);

    LogCmdHingeEmbeddingLoss(inputDesc, targetDesc, false);
    return miopen::try_([&] {
        miopen::HingeEmbeddingLossUnreducedBackward(miopen::deref(handle),
                                                    miopen::deref(inputDesc),
                                                    DataCast(input),
                                                    miopen::deref(targetDesc),
                                                    DataCast(target),
                                                    miopen::deref(doutputDesc),
                                                    DataCast(doutput),
                                                    miopen::deref(dinputDesc),
                                                    DataCast(dinput),
                                                    margin);
    });
}
