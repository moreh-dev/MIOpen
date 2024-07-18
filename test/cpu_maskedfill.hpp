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

# ifndef GUARD_CPU_MASKEDFILL_HPP
# define GUARD_CPU_MASKEDFILL_HPP

# include "tensor_holder.hpp"

template <class T> void cpu_maskedfill_forward(tensor<T> const & input, tensor<T> & output, tensor<int8_t> const & mask, T const value) {
	par_ford(std :: accumulate(input.desc.GetLengths().begin(), input.desc.GetLengths().end(), 1, std :: multiplies<> {}))([&] (size_t const i) {
		output[i] = mask[i]? value : input[i];
	} );
}

template <class T> void cpu_maskedfill_backward(tensor<T> const & outputgradient, tensor<T> & inputgradient, tensor<int8_t> const & mask) {
	par_ford(std :: accumulate(outputgradient.desc.GetLengths().begin(), outputgradient.desc.GetLengths().end(), 1, std :: multiplies<> {}))([&] (size_t const i) {
		inputgradient[i] = mask[i]? 0 : outputgradient[i];
	} );
}

# endif
