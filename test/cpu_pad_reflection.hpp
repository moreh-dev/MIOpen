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
#ifndef GUARD_CPU_PAD_REFLECTION_HPP
#define GUARD_CPU_PAD_REFLECTION_HPP

#include "tensor_holder.hpp"

template <class T>
void cpu_pad_reflection_fwd(tensor<T> input_tensor,
                        tensor<T>& ref_output_tensor,
                        int contiguous,
                        const std::vector<size_t> padding)
{
    auto input_size    = input_tensor.desc.GetSize();
    auto input_dims    = input_tensor.desc.GetLengths();
    auto output_dims   = ref_output_tensor.desc.GetLengths();
    auto input         = input_tensor.data.data();
    auto output        = ref_output_tensor.data.data();
    auto input_strides = input_tensor.desc.GetStrides();
    auto output_strides = ref_output_tensor.desc.GetStrides();
    auto output_size =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());

    if(input_size == 3 && contiguous == 1)
    {
        long padding_l = padding[0];
        size_t in_W    = input_dims[2];

        for(size_t gid = 0; gid < output_size; ++gid)
        {
            long n, c, w;
            ulong nc = gid / output_dims[2];
            w        = gid % output_dims[2];
            n        = nc / output_dims[1];
            c        = nc % output_dims[1];

            long in_start_x  = max(0L, -padding_l);
            long out_start_x = max(0L, padding_l);

            if(w < padding_l)
            {
                w = padding_l * 2 - w;
            }
            else if(padding_l <= w && w < in_W + padding_l)
            {
            }
            else
            {
                w = (in_W + padding_l - 1) * 2 - w;
            }
            w           = w - out_start_x + in_start_x;
            output[gid] = input[(input_strides[2] * (w)) + (input_strides[1] * (c)) +
                                (input_strides[0] * (n)) + 0];
        }
    }
    else if(input_size == 3 && contiguous == 0)
    {
        long padding_l = padding[0];
        size_t in_W    = input_dims[2];

        for(size_t gid = 0; gid < output_size; ++gid)
        {
            long n, c, w;
            ulong nc = gid / output_dims[2];
            w        = gid % output_dims[2];
            n        = nc / output_dims[1];
            c        = nc % output_dims[1];

            long in_start_x  = max(0L, -padding_l);
            long out_start_x = max(0L, padding_l);

            if(w < padding_l)
            {
                w = padding_l * 2 - w;
            }
            else if(padding_l <= w && w < in_W + padding_l)
            {
            }
            else
            {
                w = (in_W + padding_l - 1) * 2 - w;
            }
            w           = w - out_start_x + in_start_x;
            size_t output_idx = output_strides[0] * (gid / output_dims[2] / output_dims[1]) +
                                output_strides[1] * ((gid / output_dims[2]) % output_dims[1]) +
                                output_strides[2] * (gid % output_dims[2]) + 0;
            output[output_idx] = input[(input_strides[2] * (w)) + (input_strides[1] * (c)) +
                                    (input_strides[0] * (n)) + 0];
        }
    }
    // else if(input_size == 4)
    // {
    //     long padding_l = padding[0];
    //     long padding_t = padding[2];
    //     size_t in_H    = input_dims[2];
    //     size_t in_W    = input_dims[3];

    //     for(size_t gid = 0; gid < output_size; ++gid)
    //     {
    //         long n, c, h, w;
    //         // GET_NCHW(n, c, h, w, gid, output);
    //         ulong nch = (gid) / output_dims[3];
    //         w         = (gid) % output_dims[3];
    //         ulong nc  = nch / output_dims[2];
    //         h         = nch % output_dims[2];
    //         n         = nc / output_dims[1];
    //         c         = nc % output_dims[1];

    //         long in_start_x  = max(0L, -padding_l);
    //         long in_start_y  = max(0L, -padding_t);
    //         long out_start_x = max(0L, padding_l);
    //         long out_start_y = max(0L, padding_t);

    //         if(w < padding_l)
    //         {
    //             w = padding_l * 2 - w;
    //         }
    //         else if(padding_l <= w && w < in_W + padding_l)
    //         {
    //         }
    //         else
    //         {
    //             w = (in_W + padding_l - 1) * 2 - w;
    //         }
    //         w = w - out_start_x + in_start_x;

    //         if(h < padding_t)
    //         {
    //             h = padding_t * 2 - h;
    //         }
    //         else if(padding_t <= h && h < in_H + padding_t)
    //         {
    //         }
    //         else
    //         {
    //             h = (in_H + padding_t - 1) * 2 - h;
    //         }
    //         h = h - out_start_y + in_start_y;

    //         output[gid] = input[(input_strides[3] * (w)) + (input_strides[2] * (h)) +
    //                             (input_strides[1] * (c)) + (input_strides[0] * (n)) + 0];
    //     }
    // }
}

template <class T>
void cpu_pad_reflection_bwd(tensor<T>& ref_input_tensor,
                        tensor<T> output_tensor,
                        int contiguous,
                        const std::vector<size_t> padding)
{
    auto input_size    = ref_input_tensor.desc.GetSize();
    auto input_dims    = ref_input_tensor.desc.GetLengths();
    auto output_dims   = output_tensor.desc.GetLengths();
    auto input         = ref_input_tensor.data.data();
    auto output        = output_tensor.data.data();
    auto input_strides = ref_input_tensor.desc.GetStrides();
    auto output_strides = output_tensor.desc.GetStrides();
    auto output_size =
        std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());

    if(input_size == 3 && contiguous == 1)
    {
        long padding_l = padding[0];
        size_t in_W    = input_dims[2];
        long in_start_x  = max(0L, -padding_l);
        long out_start_x = max(0L, padding_l);
        for(size_t gid = 0; gid < output_size; ++gid)
        {
            long n, c, w;
            ulong nc = gid / output_dims[2];
            w        = gid % output_dims[2];
            n        = nc / output_dims[1];
            c        = nc % output_dims[1];

            if(w < padding_l)
            {
                w = padding_l * 2 - w;
            }
            else if(padding_l <= w && w < in_W + padding_l)
            {
            }
            else
            {
                w = (in_W + padding_l - 1) * 2 - w;
            }
            w           = w - out_start_x + in_start_x;
            input[(input_strides[2] * (w)) + (input_strides[1] * (c)) + (input_strides[0] * (n)) + 0] = 
            input[(input_strides[2] * (w)) + (input_strides[1] * (c)) + (input_strides[0] * (n)) + 0] + output[gid];
        }
    }
    else if(input_size == 3 && contiguous == 0)
    {
        long padding_l = padding[0];
        size_t in_W    = input_dims[2];
        long in_start_x  = max(0L, -padding_l);
        long out_start_x = max(0L, padding_l);
        for(size_t gid = 0; gid < output_size; ++gid)
        {
            long n, c, w;
            ulong nc = gid / output_dims[2];
            w        = gid % output_dims[2];
            n        = nc / output_dims[1];
            c        = nc % output_dims[1];

            if(w < padding_l)
            {
                w = padding_l * 2 - w;
            }
            else if(padding_l <= w && w < in_W + padding_l)
            {
            }
            else
            {
                w = (in_W + padding_l - 1) * 2 - w;
            }
            w           = w - out_start_x + in_start_x;
            size_t output_idx = output_strides[0] * (gid / output_dims[2] / output_dims[1]) +
                                output_strides[1] * ((gid / output_dims[2]) % output_dims[1]) +
                                output_strides[2] * (gid % output_dims[2]) + 0;
            input[(input_strides[2] * (w)) + (input_strides[1] * (c)) + (input_strides[0] * (n)) + 0] =
            input[(input_strides[2] * (w)) + (input_strides[1] * (c)) + (input_strides[0] * (n)) + 0] + output[output_idx];
        }
    }
}
#endif
