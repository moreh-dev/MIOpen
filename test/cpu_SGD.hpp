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

#ifndef GUARD_CPU_SGD_HPP
#define GUARD_CPU_SGD_HPP

#include "tensor_holder.hpp"

template <class T>
void cpu_SGD_forward(tensor<T> param_input,
                     tensor<T>& ref_param_output,
                     tensor<T> grad,
                     tensor<T> momentum_buffer_input,
                     tensor<T>& ref_momentum_buffer_output,
                     double lr,
                     double momentum,
                     double dampening,
                     double weight_decay,
                     char nesterov,
                     char momentum_initialized)
{
    auto dims         = param_input.desc.GetLengths();
    size_t param_size = std::accumulate(dims.begin(), dims.end(), 1ULL, std::multiplies<size_t>());

    for(int id = 0; id < param_size; ++id)
    {
        T param = param_input[id];
        T d_p   = grad[id];

        if(weight_decay != 0)
        {
            d_p += param * weight_decay;
        }

        if(momentum != 0)
        {
            T momentum_v;
            if(momentum_initialized)
            {
                momentum_v = momentum_buffer_input[id];
                momentum_v = momentum_v * momentum + d_p * (1 - dampening);
            }
            else
            {
                momentum_v = d_p;
            }
            ref_momentum_buffer_output[id] = momentum_v;

            if(nesterov)
            {
                d_p = d_p + momentum_v * momentum;
            }
            else
            {
                d_p = momentum_v;
            }
        }

        ref_param_output[id] = param - lr * d_p;
    }
}
#endif
