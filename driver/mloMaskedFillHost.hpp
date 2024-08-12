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

#ifndef MLO_MASKEDFILLHOST_H_
#define MLO_MASKEDFILLHOST_H_

#include <miopen/errors.hpp>
#include <miopen/tensor.hpp>

template <typename Tgpu, typename Tcheck>
int mloMaskedFillForwardRunHost(miopenTensorDescriptor_t const outputDesc,

                                Tgpu const* const input,
                                Tcheck* const hostoutput,

                                int8_t const* const mask,

                                Tgpu const value)
{
    auto const size  = miopen ::deref(outputDesc).GetLengths();
    auto const numel = std ::accumulate(size.begin(), size.end(), 1, std ::multiplies<>{});
    for(auto i = 0; i < numel; ++i)
    {
        hostoutput[i] = mask[i] ? value : input[i];
    }
    return 0;
}

template <typename Tgpu, typename Tcheck>
int mloMaskedFillBackwardRunHost(miopenTensorDescriptor_t const outputDesc,

                                 Tgpu const* const input,
                                 Tcheck* const hostoutput,

                                 int8_t const* const mask)
{
    auto const size  = miopen ::deref(outputDesc).GetLengths();
    auto const numel = std ::accumulate(size.begin(), size.end(), 1, std ::multiplies<>{});
    for(auto i = 0; i < numel; ++i)
    {
        hostoutput[i] = mask[i] ? 0 : input[i];
    }
    return 0;
}

#endif
