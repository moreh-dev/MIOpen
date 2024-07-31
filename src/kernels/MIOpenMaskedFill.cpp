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

template <typename T> __device__ void MaskedFillForwardContiguousImpl(
	T const * const __restrict__ input,
	T * const __restrict__ output,
	__hip_internal :: int8_t const * const __restrict__ mask,
	T const value,
	unsigned long const numel
) {
	uint64_t const gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= numel) return;
	output[gid] = mask[gid]? value : input[gid];
}
extern "C" __global__ void MaskedFillForwardContiguous(
	FLOAT const * const __restrict__ input,
	FLOAT * const __restrict__ output,
	__hip_internal :: int8_t const * const __restrict__ mask,
	FLOAT const value,
	unsigned long const numel
) {
	MaskedFillForwardContiguousImpl<FLOAT>(input, output, mask, value, numel);
}

template <typename T> __device__ void MaskedFillBackwardContiguousImpl(
	T const * const __restrict__ outputgradient,
	T * const __restrict__ inputgradient,
	__hip_internal :: int8_t const * const __restrict__ mask,
	unsigned long const numel
) {
	const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= numel) return;
	inputgradient[gid] = mask[gid]? static_cast<T>(0) : outputgradient[gid];
}
extern "C" __global__ void MaskedFillBackwardContiguous(
	FLOAT const * const __restrict__ outputgradient,
	FLOAT * const __restrict__ inputgradient,
	__hip_internal :: int8_t const * const __restrict__ mask,
	FLOAT const value,
	unsigned long const numel
) {
	MaskedFillBackwardContiguousImpl<FLOAT>(outputgradient, inputgradient, mask, numel);
}
