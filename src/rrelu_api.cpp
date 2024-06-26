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

#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/rrelu.hpp>

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

static void LogCmdRReLU(const miopenTensorDescriptor_t inputDesc, bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(inputDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "rrelufp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "rrelufp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "rrelubfp16";
        }

        MIOPEN_LOG_FUNCTION(inputDesc);
        ss << " -T " << miopen::deref(inputDesc).GetLengths();
        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenGetRReLUStatesSize(miopenHandle_t handle, size_t* stateSizeInBytes)
{
    MIOPEN_LOG_FUNCTION(handle, stateSizeInBytes);
    return miopen::try_([&] {
        miopen::deref(stateSizeInBytes) = miopen::GetRReLUStatesSize(miopen::deref(handle));
    });
}

extern "C" miopenStatus_t miopenRReLUStatesInit(miopenHandle_t handle,
                                                void* states,
                                                const size_t stateSizeInBytes,
                                                const uint64_t seed)
{
    MIOPEN_LOG_FUNCTION(handle, states, stateSizeInBytes, seed);
    return miopen::try_([&] {
        miopen::RReLUStatesInit(miopen::deref(handle), DataCast(states), stateSizeInBytes, seed);
    });
}

extern "C" miopenStatus_t miopenGetRReLUForwardWorkspaceSize(miopenHandle_t handle,
                                                             miopenTensorDescriptor_t inputDesc,
                                                             miopenTensorDescriptor_t outputDesc,
                                                             size_t* sizeInBytes)
{
    MIOPEN_LOG_FUNCTION(handle, inputDesc, outputDesc, sizeInBytes);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) = miopen::GetRReLUForwardWorkspaceSize(
            miopen::deref(handle), miopen::deref(inputDesc), miopen::deref(outputDesc));
    });
}

extern "C" miopenStatus_t miopenRReLUForward(miopenHandle_t handle,
                                             void* workspace,
                                             const size_t workspaceSizeInBytes,
                                             const void* states,
                                             const size_t stateSizeInBytes,
                                             const miopenTensorDescriptor_t inputDesc,
                                             const void* input,
                                             const miopenTensorDescriptor_t outputDesc,
                                             void* output,
                                             const miopenTensorDescriptor_t noiseDesc,
                                             void* noise,
                                             const float lower,
                                             const float upper)
{
    MIOPEN_LOG_FUNCTION(handle,
                        states,
                        stateSizeInBytes,
                        inputDesc,
                        input,
                        outputDesc,
                        output,
                        noiseDesc,
                        noise,
                        lower,
                        upper);
    LogCmdRReLU(inputDesc, true);
    return miopen::try_([&] {
        miopen::RReLUForward(miopen::deref(handle),
                             DataCast(workspace),
                             workspaceSizeInBytes,
                             DataCast(states),
                             stateSizeInBytes,
                             miopen::deref(inputDesc),
                             DataCast(input),
                             miopen::deref(outputDesc),
                             DataCast(output),
                             miopen::deref(noiseDesc),
                             DataCast(noise),
                             lower,
                             upper);
    });
}

extern "C" miopenStatus_t miopenRReLUBackward(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t noiseDesc,
                                              const void* noise,
                                              const miopenTensorDescriptor_t doutputDesc,
                                              const void* doutput,
                                              const miopenTensorDescriptor_t dinputDesc,
                                              void* dinput)
{
    MIOPEN_LOG_FUNCTION(handle, noiseDesc, noise, doutputDesc, doutput, dinputDesc, dinput);
    LogCmdRReLU(dinputDesc, false);
    return miopen::try_([&] {
        miopen::RReLUBackward(miopen::deref(handle),
                              miopen::deref(noiseDesc),
                              DataCast(noise),
                              miopen::deref(doutputDesc),
                              DataCast(doutput),
                              miopen::deref(dinputDesc),
                              DataCast(dinput));
    });
}
