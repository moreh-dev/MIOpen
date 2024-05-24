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
#include <miopen/tripletmarginloss.hpp>

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

static void LogCmdTripletMarginLoss(const miopenTensorDescriptor_t aDesc,
                                    const miopenTensorDescriptor_t oDesc,
                                    bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(aDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "tripletmarginlossfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "tripletmarginlossfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "tripletmarginlossbfp16";
        }

        MIOPEN_LOG_FUNCTION(aDesc, oDesc);
        ss << " -n " << miopen::deref(aDesc).GetLengths()[0];
        ss << " -T " << miopen::deref(aDesc).GetLengths();
        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t
miopenGetTripletMarginLossForwardWorkspaceSize(miopenHandle_t handle,
                                               const miopenTensorDescriptor_t aDesc,
                                               const miopenTensorDescriptor_t oDesc,
                                               size_t* sizeInBytes)
{

    MIOPEN_LOG_FUNCTION(handle, aDesc, oDesc, sizeInBytes);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) = miopen::GetTripletMarginLossForwardWorkspaceSize(
            miopen::deref(handle), miopen::deref(aDesc), miopen::deref(oDesc));
    });
}

extern "C" miopenStatus_t miopenTripletMarginLossForward(miopenHandle_t handle,
                                                         void* workspace,
                                                         const size_t workspaceSizeInBytes,
                                                         const miopenTensorDescriptor_t aDesc,
                                                         const void* anchor,
                                                         const miopenTensorDescriptor_t pDesc,
                                                         const void* positive,
                                                         const miopenTensorDescriptor_t nDesc,
                                                         const void* negative,
                                                         const miopenTensorDescriptor_t oDesc,
                                                         void* o,
                                                         const float margin,
                                                         const int p,
                                                         const float eps,
                                                         const bool swap,
                                                         const float divisor)
{
    MIOPEN_LOG_FUNCTION(handle,
                        workspace,
                        workspaceSizeInBytes,
                        aDesc,
                        anchor,
                        pDesc,
                        positive,
                        nDesc,
                        negative,
                        oDesc,
                        o,
                        margin,
                        p,
                        eps,
                        swap,
                        divisor);

    LogCmdTripletMarginLoss(aDesc, oDesc, true);
    return miopen::try_([&] {
        miopen::TripletMarginLossForward(miopen::deref(handle),
                                         DataCast(workspace),
                                         workspaceSizeInBytes,
                                         miopen::deref(aDesc),
                                         DataCast(anchor),
                                         miopen::deref(pDesc),
                                         DataCast(positive),
                                         miopen::deref(nDesc),
                                         DataCast(negative),
                                         miopen::deref(oDesc),
                                         DataCast(o),
                                         margin,
                                         p,
                                         eps,
                                         swap,
                                         divisor);
    });
}
