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

#include <miopen/bcelogitsloss/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace bcelogitsloss {

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

bool checkContiguous(const TensorDescriptor& x)
{
    size_t s = 1;
    for(int i = x.GetSize() - 1; i >= 0; --i)
    {
        if(s != x.GetStrides()[i])
            return false;
        s *= x.GetLengths()[i];
    }
    return true;
}

NetworkConfig ReducedForwardProblemDescription::MakeNetworkConfig() const
{
    auto input_dtype      = iDesc.GetType();
    auto target_dtype     = tDesc.GetType();
    auto weight_dtype     = wDesc.GetType();
    auto pos_weight_dtype = pwDesc.GetType();
    auto output_dtype     = oDesc.GetType();
    auto size             = iDesc.GetElementSize();

    std::ostringstream ss;

    ss << "bceloss_reduced_fwd";
    ss << "input_dtype" << input_dtype << '&' << target_dtype << '&' << weight_dtype << '&'
       << pos_weight_dtype;
    ss << "output_dtype" << output_dtype;
    ss << "size" << size;

    return NetworkConfig{ss.str()};
}

NetworkConfig ReducedBackwardProblemDescription::MakeNetworkConfig() const
{
    auto input_dtype      = iDesc.GetType();
    auto target_dtype     = tDesc.GetType();
    auto weight_dtype     = wDesc.GetType();
    auto pos_weight_dtype = pwDesc.GetType();
    auto o_grad_dtype     = doDesc.GetType();
    auto i_grad_dtype     = diDesc.GetType();
    auto t_grad_dtype     = dtDesc.GetType();
    auto size             = iDesc.GetElementSize();

    std::ostringstream ss;

    ss << "bceloss_reduced_bwd";
    ss << "input_dtype" << input_dtype << '&' << target_dtype << '&' << weight_dtype << '&'
       << pos_weight_dtype << '&' << i_grad_dtype << '&' << t_grad_dtype;
    ss << "output_dtype" << o_grad_dtype;
    ss << "size" << size;

    return NetworkConfig{ss.str()};
}

} // namespace bcelogitsloss

} // namespace miopen
