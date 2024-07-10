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

# ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
# include <hip/hipf16.h>
# include <hip/hip_runtime.h>
# endif

# include "float_types.h"

extern "C" __global__ void MaskedFillForward(
	const FLOAT * __restrict__ input,
	FLOAT * __restrict__ output,
	const bool * mask
	const float value,
	const unsigned long numel
) {
	const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= numel) return;
	output[gid] = mask[gid]? value : input[gid];
}

extern "C" __global__ void MaskedFillBackward(
	const FLOAT * __restrict__ outputgradient,
	FLOAT * __restrict__ inputgradient,
	const bool * mask
	const float value,
	const unsigned long numel
) {
	const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= numel) return;
	inputgrad[gid] = mask[gid]? static_cast<FLOAT>(0) : outputgrad[gid];
}
