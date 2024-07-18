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

__device__ void GET_NCDHW(uint64_t& n,
                          uint64_t& c,
                          uint64_t& d,
                          uint64_t& h,
                          uint64_t& w,
                          uint64_t gid,
                          uint64_t output_dim0,
                          uint64_t output_dim1,
                          uint64_t output_dim2,
                          uint64_t output_dim3,
                          uint64_t output_dim4)
{
    uint64_t ncdh = (gid) / output_dim4;
    w             = (gid) % output_dim4;
    uint64_t ncd  = (ncdh) / output_dim3;
    h             = (ncdh) % output_dim3;
    uint64_t nc   = (ncd) / output_dim2;
    d             = (ncd) % output_dim2;
    n             = (nc) / output_dim1;
    c             = (nc) % output_dim1;
}

__device__ int GET_5D_INDEX(
    const uint64_t input_dimensions[5], uint64_t n, uint64_t c, uint64_t d, uint64_t h, uint64_t w)
{
    return (((n * input_dimensions[1] + c) * input_dimensions[2] + d) * input_dimensions[3] + h) *
               input_dimensions[4] +
           w;
}