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

static void LogCmdHingeEmbeddingLoss(const miopenTensorDescriptor_t iDesc,
                                     const miopenTensorDescriptor_t tDesc,
                                     bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(iDesc).GetType();
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

        MIOPEN_LOG_FUNCTION(iDesc, tDesc);
        ss << " -n " << miopen::deref(iDesc).GetLengths()[0];
        ss << " -T " << miopen::deref(iDesc).GetLengths();
        ss << " -Si " << miopen::deref(iDesc).GetStrides();
        ss << " -St " << miopen::deref(tDesc).GetStrides();
        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t
miopenGetHingeEmbeddingLossForwardWorkspaceSize(miopenHandle_t handle,
                                                const miopenTensorDescriptor_t iDesc,
                                                const miopenTensorDescriptor_t tDesc,
                                                const miopenTensorDescriptor_t oDesc,
                                                size_t* sizeInBytes)
{

    MIOPEN_LOG_FUNCTION(handle, iDesc, tDesc, oDesc, sizeInBytes);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::GetHingeEmbeddingLossForwardWorkspaceSize(miopen::deref(handle),
                                                              miopen::deref(iDesc),
                                                              miopen::deref(tDesc),
                                                              miopen::deref(oDesc));
    });
}

extern "C" miopenStatus_t miopenHingeEmbeddingLossForward(miopenHandle_t handle,
                                                          void* workspace,
                                                          size_t workspaceSizeInBytes,
                                                          const miopenTensorDescriptor_t iDesc,
                                                          const void* i,
                                                          const miopenTensorDescriptor_t tDesc,
                                                          const void* t,
                                                          const miopenTensorDescriptor_t oDesc,
                                                          void* o,
                                                          const float margin,
                                                          const float divisor)
{
    MIOPEN_LOG_FUNCTION(
        handle, workspace, workspaceSizeInBytes, iDesc, i, tDesc, t, oDesc, o, margin);

    LogCmdHingeEmbeddingLoss(iDesc, tDesc, true);
    return miopen::try_([&] {
        miopen::HingeEmbeddingLossForward(miopen::deref(handle),
                                          DataCast(workspace),
                                          workspaceSizeInBytes,
                                          miopen::deref(iDesc),
                                          DataCast(i),
                                          miopen::deref(tDesc),
                                          DataCast(t),
                                          miopen::deref(oDesc),
                                          DataCast(o),
                                          margin,
                                          divisor);
    });
}

extern "C" miopenStatus_t
miopenHingeEmbeddingLossUnreducedForward(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t iDesc,
                                         const void* i,
                                         const miopenTensorDescriptor_t tDesc,
                                         const void* t,
                                         const miopenTensorDescriptor_t oDesc,
                                         void* o,
                                         const float margin)
{
    MIOPEN_LOG_FUNCTION(handle, iDesc, i, tDesc, t, oDesc, o, margin);

    LogCmdHingeEmbeddingLoss(iDesc, tDesc, true);
    return miopen::try_([&] {
        miopen::HingeEmbeddingLossUnreducedForward(miopen::deref(handle),
                                                   miopen::deref(iDesc),
                                                   DataCast(i),
                                                   miopen::deref(tDesc),
                                                   DataCast(t),
                                                   miopen::deref(oDesc),
                                                   DataCast(o),
                                                   margin);
    });
}

extern "C" miopenStatus_t miopenHingeEmbeddingLossUnreducedBackward(miopenHandle_t handle,
                                                                    miopenTensorDescriptor_t iDesc,
                                                                    const void* i,
                                                                    miopenTensorDescriptor_t tDesc,
                                                                    const void* t,
                                                                    miopenTensorDescriptor_t dODesc,
                                                                    const void* dO,
                                                                    miopenTensorDescriptor_t dIDesc,
                                                                    void* dI,
                                                                    float margin)
{
    MIOPEN_LOG_FUNCTION(handle, iDesc, i, tDesc, t, dODesc, dO, dIDesc, dI, margin);

    LogCmdHingeEmbeddingLoss(iDesc, tDesc, false);
    return miopen::try_([&] {
        miopen::HingeEmbeddingLossUnreducedBackward(miopen::deref(handle),
                                                    miopen::deref(iDesc),
                                                    DataCast(i),
                                                    miopen::deref(tDesc),
                                                    DataCast(t),
                                                    miopen::deref(dODesc),
                                                    DataCast(dO),
                                                    miopen::deref(dIDesc),
                                                    DataCast(dI),
                                                    margin);
    });
}
