/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
 
#include <miopen/SGD/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {
namespace SGD {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    auto dtype = paramInDesc.GetType();
    int32_t total_dims = paramInDesc.GetLengths().size();

    int32_t param_size = 0;
    for(int32_t i = 0; i < total_dims; ++i)
    {
        param_size += paramInDesc.GetLengths()[i];
    }

    std::ostringstream ss;
    ss << "dtype" << dtype;
    ss << "total_dims" << total_dims;
    ss << "param_size" << param_size;

    return NetworkConfig{ss.str()};
}

} // namespace SGD
} // namespace miopen
