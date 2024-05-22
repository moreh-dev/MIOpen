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

#include <miopen/kldivloss.hpp>
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

static void LogCmdKLDivLoss(const miopenTensorDescriptor_t xDesc,
                            const miopenTensorDescriptor_t tDesc,
                            bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(xDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "kldivlossfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "kldivloss";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "kldivlossbfp16";
        }

        MIOPEN_LOG_FUNCTION(xDesc, tDesc);
        ss << " -N " << miopen::deref(xDesc).GetLengths()[0];
        ss << " -T " << miopen::deref(xDesc).GetLengths();
        ss << " -Si " << miopen::deref(xDesc).GetStrides();
        ss << " -St " << miopen::deref(tDesc).GetStrides();

        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenKLDivLossUnreducedForward(miopenHandle_t handle,
                                                          const miopenTensorDescriptor_t inputDesc,
                                                          const void* input,
                                                          const miopenTensorDescriptor_t targetDesc,
                                                          const void* target,
                                                          const miopenTensorDescriptor_t outputDesc,
                                                          void* output,
                                                          bool log_target)
{
    MIOPEN_LOG_FUNCTION(
        handle, inputDesc, input, targetDesc, target, outputDesc, output, log_target);

    LogCmdKLDivLoss(inputDesc, targetDesc, true);
    return miopen::try_([&] {
        miopen::KLDivLossUnreducedForward(miopen::deref(handle),
                                          miopen::deref(inputDesc),
                                          DataCast(input),
                                          miopen::deref(targetDesc),
                                          DataCast(target),
                                          miopen::deref(outputDesc),
                                          DataCast(output),
                                          log_target);
    });
}

extern "C" miopenStatus_t
miopenGetKLDivLossReducedForwardWorkspaceSize(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t inputDesc,
                                              const miopenTensorDescriptor_t targetDesc,
                                              const miopenTensorDescriptor_t outputDesc,
                                              float divisor,
                                              bool log_target,
                                              size_t* sizeInBytes)
{
    MIOPEN_LOG_FUNCTION(handle, inputDesc, targetDesc, outputDesc, sizeInBytes);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::GetKLDivLossReducedForwardWorkspaceSize(miopen::deref(handle),
                                                            miopen::deref(inputDesc),
                                                            miopen::deref(targetDesc),
                                                            miopen::deref(outputDesc),
                                                            divisor,
                                                            log_target);
    });
}

extern "C" miopenStatus_t miopenKLDivLossReducedForward(miopenHandle_t handle,
                                                        void* workspace,
                                                        size_t workspaceSizeInBytes,
                                                        const miopenTensorDescriptor_t inputDesc,
                                                        const void* input,
                                                        const miopenTensorDescriptor_t targetDesc,
                                                        const void* target,
                                                        const miopenTensorDescriptor_t outputDesc,
                                                        void* output,
                                                        float divisor,
                                                        bool log_target)
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
                        divisor,
                        log_target);

    LogCmdKLDivLoss(inputDesc, targetDesc, true);
    return miopen::try_([&] {
        miopen::KLDivLossReducedForward(miopen::deref(handle),
                                        DataCast(workspace),
                                        workspaceSizeInBytes,
                                        miopen::deref(inputDesc),
                                        DataCast(input),
                                        miopen::deref(targetDesc),
                                        DataCast(target),
                                        miopen::deref(outputDesc),
                                        DataCast(output),
                                        divisor,
                                        log_target);
    });
}

extern "C" miopenStatus_t
miopenKLDivLossUnreducedBackward(miopenHandle_t handle,
                                 const miopenTensorDescriptor_t inputDesc,
                                 const void* input,
                                 const miopenTensorDescriptor_t targetDesc,
                                 const void* target,
                                 const miopenTensorDescriptor_t outputGradDesc,
                                 const void* output_grad,
                                 const miopenTensorDescriptor_t inputGradDesc,
                                 void* input_grad,
                                 const miopenTensorDescriptor_t targetGradDesc,
                                 void* target_grad,
                                 bool log_target)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        targetDesc,
                        target,
                        outputGradDesc,
                        output_grad,
                        inputGradDesc,
                        input_grad,
                        targetGradDesc,
                        target_grad,
                        log_target);

    LogCmdKLDivLoss(inputDesc, targetDesc, true);
    return miopen::try_([&] {
        miopen::KLDivLossUnreducedBackward(miopen::deref(handle),
                                           miopen::deref(inputDesc),
                                           DataCast(input),
                                           miopen::deref(targetDesc),
                                           DataCast(target),
                                           miopen::deref(outputGradDesc),
                                           DataCast(output_grad),
                                           miopen::deref(inputGradDesc),
                                           DataCast(input_grad),
                                           miopen::deref(targetGradDesc),
                                           DataCast(target_grad),
                                           log_target);
    });
}

extern "C" miopenStatus_t
miopenKLDivLossReducedBackward(miopenHandle_t handle,
                               const miopenTensorDescriptor_t inputDesc,
                               const void* input,
                               const miopenTensorDescriptor_t targetDesc,
                               const void* target,
                               const miopenTensorDescriptor_t outputGradDesc,
                               const void* output_grad,
                               const miopenTensorDescriptor_t inputGradDesc,
                               void* input_grad,
                               const miopenTensorDescriptor_t targetGradDesc,
                               void* target_grad,
                               float divisor,
                               bool log_target)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        targetDesc,
                        target,
                        outputGradDesc,
                        output_grad,
                        inputGradDesc,
                        input_grad,
                        targetGradDesc,
                        target_grad,
                        divisor,
                        log_target);

    LogCmdKLDivLoss(inputDesc, targetDesc, true);
    return miopen::try_([&] {
        miopen::KLDivLossReducedBackward(miopen::deref(handle),
                                         miopen::deref(inputDesc),
                                         DataCast(input),
                                         miopen::deref(targetDesc),
                                         DataCast(target),
                                         miopen::deref(outputGradDesc),
                                         DataCast(output_grad),
                                         miopen::deref(inputGradDesc),
                                         DataCast(input_grad),
                                         miopen::deref(targetGradDesc),
                                         DataCast(target_grad),
                                         divisor,
                                         log_target);
    });
}