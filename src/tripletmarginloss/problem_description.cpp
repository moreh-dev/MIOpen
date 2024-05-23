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

#include <miopen/tripletmarginloss/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace tripletmarginloss {

bool checkSameType(const TensorDescriptor& x, const TensorDescriptor& y)
{
    if(x.GetType() != y.GetType())
        return false;
    return true;
}

bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y)
{
    if(x.GetSize() != y.GetSize())
        return false;
    for(int32_t i = 0; i < x.GetSize(); ++i)
    {
        if(x.GetLengths()[i] != y.GetLengths()[i])
            return false;
    }
    return true;
}

bool checkSameStride(const TensorDescriptor& x, const TensorDescriptor& y)
{
    if(x.GetSize() != y.GetSize())
        return false;
    for(int32_t i = 0; i < x.GetSize(); ++i)
    {
        if(x.GetStrides()[i] != y.GetStrides()[i])
            return false;
    }
    return true;
}

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    auto anchor_dtype = aDesc.GetType();
    auto output_dtype = oDesc.GetType();
    auto size0        = aDesc.GetLengths()[0];
    auto size1        = aDesc.GetLengths()[1];

    std::ostringstream ss;

    ss << "tripletmarginloss_unreduced_fwd";
    ss << "a_dtype" << anchor_dtype;
    ss << "o_dtype" << output_dtype;
    ss << "size0" << size0;
    ss << "size1" << size1;

    return NetworkConfig{ss.str()};
}

} // namespace tripletmarginloss

} // namespace miopen
