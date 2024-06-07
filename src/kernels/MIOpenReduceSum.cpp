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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

__device__ FLOAT_ACCUM warp_reduce_sum(FLOAT_ACCUM val)
{
    if(warpSize >= 64)
        val += __shfl_down(val, 32);
    if(warpSize >= 32)
        val += __shfl_down(val, 16);
    if(warpSize >= 16)
        val += __shfl_down(val, 8);
    if(warpSize >= 8)
        val += __shfl_down(val, 4);
    if(warpSize >= 4)
        val += __shfl_down(val, 2);
    if(warpSize >= 2)
        val += __shfl_down(val, 1);
    return val;
}

__device__ FLOAT_ACCUM block_reduce_sum(FLOAT_ACCUM val)
{
    static __shared__ FLOAT_ACCUM shared[REDUCE_SIZE / warpSize];
    auto lane = threadIdx.x % warpSize;
    auto wid  = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);

    if(lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = threadIdx.x < REDUCE_SIZE / warpSize ? shared[lane] : 0;
    if(wid == 0)
        val = warp_reduce_sum(val);

    return val;
}

template <typename TO>
__device__ void ReduceSum(const FLOAT_ACCUM* input, TO* output, size_t N)
{
    auto gid = blockIdx.x * blockDim.x + threadIdx.x;

    FLOAT_ACCUM val = gid < N ? input[gid] : CVT_FP32_2ACCUM(0.0f);
    if(blockDim.x == warpSize)
        val = warp_reduce_sum(val);
    else
        val = block_reduce_sum(val);

    if(threadIdx.x == 0)
        output[blockIdx.x] = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void
ReduceSum(const FLOAT_ACCUM* __restrict__ input, OUTPUT_TYPE* __restrict__ output, size_t N)
{
    // instantiate the kernel
    ReduceSum<OUTPUT_TYPE>(input, output, N);
}

extern "C" __global__ void ReduceSumFLOATACCUM(const FLOAT_ACCUM* __restrict__ input,
                                               FLOAT_ACCUM* __restrict__ output,
                                               size_t N)
{
    auto gid = blockIdx.x * blockDim.x + threadIdx.x;

    FLOAT_ACCUM val = gid < N ? input[gid] : 0.0f;
    if(blockDim.x == warpSize)
        val = warp_reduce_sum(val);
    else
        val = block_reduce_sum(val);

    if(threadIdx.x == 0)
        output[blockIdx.x] = val;
}

template <typename TO>
__device__ void Reduce1dSum(const FLOAT_ACCUM* __restrict__ input,
                            TO* __restrict__ output,
                            size_t output_numel,
                            size_t inner_size,
                            size_t outer_size)
{
    int tid  = threadIdx.x;
    int oidx = blockIdx.x;

    FLOAT_ACCUM sum = CVT_FP32_2ACCUM(0.0f);

    for(int i = 0; i < outer_size; ++i)
        for(int j = tid; j < inner_size; j += blockDim.x)
            sum += input[i * output_numel * inner_size + oidx * inner_size + j];

    if(blockDim.x == warpSize)
        sum = warp_reduce_sum(sum);
    else
        sum = block_reduce_sum(sum);

    if(tid == 0)
        output[oidx] = CVT_ACCUM2FLOAT(sum);
}

extern "C" __global__ void Reduce1dSum(const FLOAT_ACCUM* __restrict__ input,
                                       OUTPUT_TYPE* __restrict__ output,
                                       size_t output_numel,
                                       size_t inner_size,
                                       size_t outer_size)
{
    // instantiate the kernel
    Reduce1dSum<OUTPUT_TYPE>(input, output, output_numel, inner_size, outer_size);
}