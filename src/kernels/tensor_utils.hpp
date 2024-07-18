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

__device__ void GET_NCDHW(uint64_t ncdhw[5], uint64_t gid, const uint64_t output_dimensions[5])
{
    uint64_t ncdh = (gid) / output_dimensions[4];
    ncdhw[4]      = (gid) % output_dimensions[4];
    uint64_t ncd  = (ncdh) / output_dimensions[3];
    ncdhw[3]      = (ncdh) % output_dimensions[3];
    uint64_t nc   = (ncd) / output_dimensions[2];
    ncdhw[2]      = (ncd) % output_dimensions[2];
    uint64_t n    = (nc) / output_dimensions[1];
    ncdhw[1]      = (nc) % output_dimensions[1];
    ncdhw[0]      = n;
}

__device__ uint64_t GET_STRIDED_INDEX(const uint64_t indices[5], const uint64_t strides[5])
{
    return indices[0] * strides[0] + indices[1] * strides[1] + indices[2] * strides[2] +
           indices[3] * strides[3] + indices[4] * strides[4];
}