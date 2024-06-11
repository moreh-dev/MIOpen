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

#include <cstddef>
#include <miopen/softmaxcrossentropywithlogits/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace softmaxcrossentropywithlogits {

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

NetworkConfig FwdProblemDescription::MakeNetworkConfig() const
{
    auto input_dims  = inputDesc.GetLengths();
    auto input_dtype = inputDesc.GetType();
    auto Si          = inputDesc.GetStrides();
    auto St          = targetDesc.GetStrides();
    auto So          = outputDesc.GetStrides();
    auto Sb          = backpropDesc.GetStrides();

    std::ostringstream ss;
    ss << "softmaxcrossentropywithlogits";
    ss << "is_fwd" << true;
    ss << "input_dtype" << input_dtype;
    ss << "input_dims" << input_dims;
    ss << "input1_stride" << Si;
    ss << "target_stride" << St;
    ss << "output_stride" << So;
    ss << "backprop_stride" << Sb;

    return NetworkConfig{ss.str()};
}

NetworkConfig BwdProblemDescription::MakeNetworkConfig() const
{
    auto input_dims  = inputDesc.GetLengths();
    auto input_dtype = inputDesc.GetType();
    auto Sdo         = outputGradDesc.GetStrides();
    auto Sb          = backpropDesc.GetStrides();
    auto Si          = inputDesc.GetStrides();
    auto Sdi         = inputGradDesc.GetStrides();
    auto Sdt         = targetGradDesc.GetStrides();

    std::ostringstream ss;
    ss << "softmaxcrossentropywithlogits";
    ss << "is_fwd" << false;
    ss << "input_dtype" << input_dtype;
    ss << "input_dims" << input_dims;
    ss << "output_grad_stride" << Sdo;
    ss << "backprop_stride" << Sb;
    ss << "input_stride" << Si;
    ss << "input_grad_stride" << Sdi;
    ss << "target_grad_stride" << Sdt;

    return NetworkConfig{ss.str()};
}

} // namespace softmaxcrossentropywithlogits

} // namespace miopen
