
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

#include <miopen/errors.hpp>
#include <miopen/gather/problem_description.hpp>
#include <miopen/tensor.hpp>

#include <cstddef>
#include <sstream>

namespace miopen {

namespace gather {

// M is size of shape
template <int M>
tensor_view_t<M> reshape(const TensorDescriptor& tensorDes, const std::vector<size_t>& shape)
{
    if(!tensorDes.IsContiguous())
    {
        MIOPEN_THROW("Cannot reshape non-contiguous tensor");
    }
    size_t numel = tensorDes.GetElementSize();

    size_t inferred_dim = -1;
    size_t new_numel    = 1;
    for(size_t i = 0; i < shape.size(); i++)
    {
        auto dim = shape[i];
        if(dim == -1)
        {
            if(inferred_dim != -1)
            {
                MIOPEN_THROW("only one dimension can be inferred");
            }
            inferred_dim = i;
        }
        else
        {
            if(dim <= 0)
            {
                MIOPEN_THROW("dimension must be positive");
            }
            new_numel *= dim;
        }
    }

    if(numel < new_numel)
    {
        MIOPEN_THROW("invalid shape size (can't reshape tensor into a larger size)");
    }

    tensor_view_t<M> new_tv;
    for(size_t i = 0; i < M; i++)
    {
        new_tv.size[i] = shape[i];
    }

    if(inferred_dim != -1)
    {
        if(numel % new_numel != 0)
        {
            MIOPEN_THROW("invalid shape size");
        }
        new_tv.size[inferred_dim] = numel / new_numel;
    }
    else
    {
        if(numel != new_numel)
        {
            MIOPEN_THROW("invalid shape size");
        }
    }

    new_tv.get_contiguous_stride();

    return new_tv;
}

template tensor_view_t<3> reshape(const TensorDescriptor& tensorDes,
                                  const std::vector<size_t>& shape);

template tensor_view_t<4> reshape(const TensorDescriptor& tensorDes,
                                  const std::vector<size_t>& shape);

NetworkConfig BwdProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;

    ss << "gatherv2";
    ss << "dtype" << paramGradDesc.GetType();
    ss << "index_type" << indicesDesc.GetType();
    ss << "param_size" << paramGradDesc.GetElementSize();
    ss << "outgrad_size" << outputGradDesc.GetElementSize();
    ss << "mode " << gatherDesc.getMode();
    ss << "dim " << gatherDesc.getDim();
    ss << "batch dim " << gatherDesc.getBatchDims();

    return NetworkConfig{ss.str()};
}

} // namespace gather

} // namespace miopen
