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

#include "miopen/pad_constant/problem_description.hpp"
#include "miopen/names.hpp"

namespace miopen {
namespace pad_constant_fwd_contiguous {
NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    // Should be the only thing we need? (After all paddings are all the same operation kind)
    auto dtype = xDesc.GetType();

    std::ostringstream ss;
    if(IsContiguous())
        ss << "contiguous-";
    ss << "fwd-";
    ss << "dtype" << dtype;
    ss << "yDesc" << yDesc.GetElementSize();

    return NetworkConfig{ss.str()};
}
} // namespace pad_constant_fwd_contiguous
namespace pad_constant_bwd {
NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    auto dtype = yDesc.GetType();

    std::ostringstream ss;
    if(IsContiguous())
        ss << "contiguous-";
    ss << "bwd-";
    ss << "dtype" << dtype;
    ss << "yDesc" << yDesc.GetElementSize();

    return NetworkConfig{ss.str()};
}
} // namespace pad_constant_bwd
} // namespace miopen
