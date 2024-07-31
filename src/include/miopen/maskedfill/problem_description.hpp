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

# pragma once

# include <miopen/problem_description_base.hpp>

# include <miopen/tensor.hpp>

namespace miopen :: maskedfill {

	struct ProblemDescription: ProblemDescriptionBase {
		ProblemDescription(TensorDescriptor const & outputDesc_, miopenMaskedFillDirection_t const direction_): outputDesc(outputDesc_), direction(direction_) {}
		TensorDescriptor const & GetOutputDesc() const { return outputDesc; }
		bool const IsBackward() const { return direction == MIOPEN_MASKEDFILL_BACKWARD; }
		NetworkConfig MakeNetworkConfig() const override;
		private:
		TensorDescriptor outputDesc;
		miopenMaskedFillDirection_t direction;
		NetworkConfig MakeForwardNetworkConfig() const;
	};

}
