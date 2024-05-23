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

#include <miopen/cosineembeddingloss.hpp>
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

static void LogCmdCosineEmbeddingLoss(const miopenTensorDescriptor_t x1Desc,
                                      const miopenTensorDescriptor_t x2Desc,
                                      const miopenTensorDescriptor_t tDesc,
                                      bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(x1Desc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "cosineembeddinglossfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "cosineembeddingloss";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "cosineembeddinglossbfp16";
        }

        MIOPEN_LOG_FUNCTION(x1Desc, x2Desc, tDesc);
        ss << " -N " << miopen::deref(x1Desc).GetLengths()[0];
        ss << " -D " << miopen::deref(x1Desc).GetLengths()[1];
        ss << " -Si1 " << miopen::deref(x1Desc).GetStrides();
        ss << " -Si2 " << miopen::deref(x2Desc).GetStrides();
        ss << " -St " << miopen::deref(tDesc).GetStrides();

        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t
miopenCosineEmbeddingLossUnreducedForward(miopenHandle_t handle,
                                          const miopenTensorDescriptor_t input1Desc,
                                          const void* input1,
                                          const miopenTensorDescriptor_t input2Desc,
                                          const void* input2,
                                          const miopenTensorDescriptor_t targetDesc,
                                          const void* target,
                                          const miopenTensorDescriptor_t outputDesc,
                                          void* output,
                                          const float margin)
{
    MIOPEN_LOG_FUNCTION(handle,
                        input1Desc,
                        input1,
                        input2Desc,
                        input2,
                        targetDesc,
                        target,
                        outputDesc,
                        output,
                        margin);

    LogCmdCosineEmbeddingLoss(input1Desc, input2Desc, targetDesc, true);
    return miopen::try_([&] {
        miopen::CosineEmbeddingLossUnreducedBackward(miopen::deref(handle),
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
